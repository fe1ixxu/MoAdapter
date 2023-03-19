from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from fairseq.modules import LayerNorm
from fairseq.utils import get_activation_fn


class CLSAGate(torch.nn.Module):
    def __init__(self, model_dim: int, p: float = 0.0):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, 1)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(
        self,
        input: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.wg(input)
        gates = logits.squeeze(-1).sigmoid()
        gates = self.dropout(gates)
        if input_mask is not None and input_mask.any():
            nonpadding = ~input_mask.bool()
            gates = gates * nonpadding.to(gates.dtype)
        return gates


class CLSALayer(torch.nn.Module):
    def __init__(
        self,
        moa_layer: torch.nn.Module,
        ffn_fn: Callable,
        model_dim: int,
        p: float = 0.0,
        lang_idx: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.moa_layer = moa_layer
        self.ffn_fn = ffn_fn
        self.gate = CLSAGate(model_dim, p)
        self.ffn_proj = torch.nn.Linear(model_dim, model_dim)
        if lang_idx is not None:
            self.register_buffer("lang_idx", lang_idx)
        else:
            self.lang_idx = None

    def forward(
        self,
        *input: torch.Tensor,
        residual: torch.Tensor,
        input_padding_mask=None,
        prefix_tokens=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(input) == 1, "only single input Tensor supported"

        gates = self.gate(input[0], input_padding_mask)
        x_ffn = self.ffn_fn(*input)
        residual = residual.transpose(0,1)
        x_ffn = x_ffn + residual
        x_ffn = self.ffn_proj(torch.nn.functional.relu(x_ffn))
        x_moa, l_aux = self.moa_layer(
            *input, input_padding_mask=input_padding_mask, prefix_tokens=prefix_tokens, source=kwargs["source"],
        )
        x_out = x_ffn * (1 - gates).unsqueeze(-1) + x_moa * gates.unsqueeze(-1)

        if input_padding_mask is None:
            input_padding_mask = torch.zeros_like(input[0][:, :, 0], dtype=torch.bool)

        used_budget = (gates * (~input_padding_mask)).sum()
        total_budget = (~input_padding_mask).sum()

        l_aux["cmr_gate_loss_num"] = used_budget
        l_aux["cmr_gate_loss_denom"] = total_budget

        self.moa_layer.metadata["cmr_lang_gates"] = 0
        if prefix_tokens is not None and self.lang_idx is not None:
            num_langs = self.lang_idx.shape[0]
            # map lang token indices to lang_idx
            batch_langs = prefix_tokens.new_zeros(gates.shape[0])
            # non-matches have value 0 in batch_langs
            lang_match = torch.where(
                prefix_tokens.expand(-1, num_langs) == self.lang_idx
            )
            batch_langs[lang_match[0]] = lang_match[1]

            out = gates.new_zeros(num_langs, gates.shape[0])
            out[batch_langs, torch.arange(gates.shape[0])] = 1
            out = F.normalize(out, p=1, dim=1, eps=1e-5)

            # per-lang, (soft) fraction of tokens routed to MoA layers
            self.moa_layer.metadata["cmr_lang_gates"] = out.mm(
                gates.mean(dim=1, keepdim=True)
            ).detach()
        return x_out, l_aux

# parallel:
class ParallelMoALayer(torch.nn.Module):
    def __init__(
        self,
        moa_layer: torch.nn.Module,
        ffn_fn: Callable,
    ) -> None:
        super().__init__()
        self.moa_layer = moa_layer
        self.ffn_fn = ffn_fn

    def forward(
        self,
        *input: torch.Tensor,
        residual: torch.Tensor,
        prefix_tokens=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(input) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x_ffn = self.ffn_fn(*input)
        x_moa, l_aux = self.moa_layer(*input, prefix_tokens=prefix_tokens, source=kwargs["source"])
        x_out = x_ffn + x_moa + residual

        return x_out, l_aux

## seq
class SeqMoALayer(torch.nn.Module):
    def __init__(
        self,
        moa_layer: torch.nn.Module,
        ffn_fn: Callable,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.moa_layer = moa_layer
        self.ffn_fn = ffn_fn
        self.layer_norm = LayerNorm(model_dim, elementwise_affine=True)

    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        prefix_tokens=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual
        shortcut = x
        
        x = self.layer_norm(x)
        x, l_aux = self.moa_layer(x, prefix_tokens=prefix_tokens, source=kwargs["source"])
        x = x + shortcut

        return x, l_aux

def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.0)
    return m

class NaiveAdapter(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        bottleneck_size: int,
    ):
        super().__init__()

        # reuse the transformer Linear layer to have consistent init with the rest of the model
        self.down = Linear(input_size, bottleneck_size)
        self.up = Linear(bottleneck_size, input_size)
        self.layer_norm = LayerNorm(input_size, elementwise_affine=True)
        self.activation_fn = get_activation_fn("relu")

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation_fn(x)
        x = self.up(x)

        return x + shortcut

class ADMoALayer(torch.nn.Module):
    def __init__(
        self,
        moa_layer: torch.nn.Module,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
        num_langs: List,
    ) -> None:
        super().__init__()
        self.moa_layer = moa_layer
        self.ffn_fn = ffn_fn
        self.layer_norm = LayerNorm(model_dim, elementwise_affine=True)
        self.lang_adapters=torch.nn.ModuleDict([])
        for lang in num_langs:
            self.lang_adapters[lang]= NaiveAdapter(
                model_dim,
                bottleneck_size,
                )
    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        prefix_tokens=None,
        lang_id=None,
        side=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual
        shortcut = x
        
        if side == "moa" or (not self.training):
            x = self.layer_norm(x)
            x, l_aux = self.moa_layer(x, prefix_tokens=prefix_tokens, source=kwargs["source"])
            x = x + shortcut
        else:
            x_ffn = self.lang_adapters[lang_id](x)
            x = self.layer_norm(x)
            x_moa, l_aux = self.moa_layer(x, prefix_tokens=prefix_tokens, source=kwargs["source"])
            x = x_moa + x_ffn
            
            # l_aux = {"moa_gate_loss": torch.tensor([0.]).to(x.device)} # dummy gate loss

        return x, l_aux

class SeqNaiveLayer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
        num_langs: List,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        self.layer_norm = LayerNorm(model_dim, elementwise_affine=True)
        self.lang_adapters=torch.nn.ModuleDict([])
        for lang in num_langs:
            self.lang_adapters[lang]= NaiveAdapter(
                model_dim,
                bottleneck_size,
                )
    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        prefix_tokens=None,
        lang_id=None,
        side=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual
        shortcut = x
        x = self.lang_adapters[lang_id](x)
        l_aux = {"moa_gate_loss": torch.tensor([0.]).to(x.device)} # dummy gate loss

        return x, l_aux