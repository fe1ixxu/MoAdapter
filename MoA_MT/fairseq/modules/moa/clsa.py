from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from fairseq.modules import LayerNorm
from fairseq.utils import get_activation_fn
from fairseq.modules.linear import Linear as FairLinear


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

class L0Adapter(torch.nn.Module):
    def __init__(
        self,
        input_size,
        bottleneck_size,
        num_langs,
        init_mean=0,
        init_sdev=0.01,
        init_beta=0.05,
        zeta=1.1,
        gamma=-0.1,
        epsilon=1e-6
        ):
        super().__init__()
        self.size = input_size
        self.down = Linear(input_size, bottleneck_size)
        self.up = Linear(bottleneck_size, input_size)
        self.layer_norm = LayerNorm(input_size, elementwise_affine=True)
        self.activation_fn = get_activation_fn("relu")
        self.zeta = zeta
        self.gamma = gamma
        self.epsilon = epsilon
        self.down_loga = torch.nn.Parameter(torch.zeros(num_langs, input_size).normal_(init_mean, init_sdev)
        )
        self.up_loga = torch.nn.Parameter(torch.zeros(num_langs, bottleneck_size).normal_(init_mean, init_sdev)
        )
        self.down_beta = init_beta #torch.nn.Parameter(torch.zeros(num_langs).fill_(init_beta))
        self.up_beta = init_beta #torch.nn.Parameter(torch.zeros(num_langs).fill_(init_beta))
        
    def forward(self, x, lang_id):
        shortcut = x
        x = self.layer_norm(x)

        ## language L0 for down
        if self.training:
            down_mask = self.sample_and_get_masks(x, lang_id, self.down_loga, self.down_beta)
        else:
            down_mask = F.hardtanh(torch.sigmoid(self.down_loga[lang_id]) * (self.zeta - self.gamma) + self.gamma, min_val=0, max_val=1)
        x = F.linear(x, down_mask.view(1, -1) * self.down.weight, self.down.bias)

        x = self.activation_fn(x)

        # language L0 for up
        if self.training:
            up_mask = self.sample_and_get_masks(x, lang_id, self.up_loga, self.up_beta)
        else:
            up_mask = F.hardtanh(torch.sigmoid(self.up_loga[lang_id]) * (self.zeta - self.gamma) + self.gamma, min_val=0, max_val=1)
        x = F.linear(x, up_mask.view(1, -1) * self.up.weight, self.up.bias)

        x = x + shortcut
        return x

    def sample_and_get_masks(self, x, lang_id, loga, beta):
        u = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device).uniform_(self.epsilon, 1-self.epsilon)
        s = torch.sigmoid((torch.log(u) - torch.log(1-u) + loga[lang_id]) / beta)
        s = s * (self.zeta - self.gamma) + self.gamma
        s = F.hardtanh(s, min_val=0, max_val=1)
        return s

class L0DropoutAdapter(torch.nn.Module):
    def __init__(
        self,
        input_size,
        bottleneck_size,
        num_langs,
        init_mean=0,
        init_sdev=0.01,
        init_beta=0.01,
        zeta=1.1,
        gamma=-0.1,
        epsilon=1e-6
        ):
        super().__init__()
        self.size = input_size
        self.down = Linear(input_size, bottleneck_size)
        self.up = Linear(bottleneck_size, input_size)
        self.layer_norm = LayerNorm(input_size, elementwise_affine=True)
        self.activation_fn = get_activation_fn("relu")
        self.zeta = zeta
        self.gamma = gamma
        self.epsilon = epsilon

        self.pair2ind = {num_langs[i]: i for i in range(len(num_langs))}
        self.loga = torch.nn.Parameter(torch.zeros(len(num_langs), input_size).normal_(init_mean, init_sdev))

        self.beta = init_beta
    def forward(self, x, lang_id):
        lang_id = self.pair2ind[lang_id]
        shortcut = x

        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation_fn(x)
        x = self.up(x)

        ## language L0 for down
        if self.training:
            mask = self.sample_and_get_masks(x, lang_id, self.loga, self.beta)
        else:
            mask = F.hardtanh(torch.sigmoid(self.loga[lang_id]) * (self.zeta - self.gamma) + self.gamma, min_val=0, max_val=1)

        x = mask.view(1, 1, -1) * x
        x = x + shortcut
        return x

    def sample_and_get_masks(self, x, lang_id, loga, beta):
        u = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device).uniform_(self.epsilon, 1-self.epsilon)
        s = torch.sigmoid((torch.log(u) - torch.log(1-u) + loga[lang_id]) / beta)
        s = s * (self.zeta - self.gamma) + self.gamma
        s = F.hardtanh(s, min_val=0, max_val=1)
        return s

# if __name__ == "__main__":
#     a =L0DropoutAdapter(20,10,10, init_beta=2)
#     b = torch.randn(1,20)
#     print(a(b,2))
    # print(a.training, a(b, 2))

class L0Layer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
        num_langs: List,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        self.adapter = L0Adapter(
            input_size=model_dim,
            bottleneck_size=bottleneck_size,
            num_langs=num_langs,
        )

    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        lang_id=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual

        x = self.adapter(x, lang_id)
        return x, None

class L0DropLayer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
        num_langs: List,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        self.adapter = L0DropoutAdapter(
            input_size=model_dim,
            bottleneck_size=bottleneck_size,
            num_langs=num_langs,
        )

    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        lang_id=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual

        x = self.adapter(x, lang_id)
        return x, None



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

    def forward(self, x: torch.Tensor, **kwargs: Any):
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
        self.bottleneck_size = bottleneck_size
        if bottleneck_size > 0:
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
            if self.bottleneck_size > 0:
                x_ffn = self.lang_adapters[lang_id](x)
            x = self.layer_norm(x)
            x_moa, l_aux = self.moa_layer(x, prefix_tokens=prefix_tokens, source=kwargs["source"])
            if self.bottleneck_size > 0:
                x = x_moa + x_ffn
            else:
                x = x + shortcut
            # x_ffn = self.lang_adapters[lang_id](x)
            
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


class LUALayer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size1: int,
        bottleneck_size2: int,
        num_langs: List,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        # self.layer_norm = LayerNorm(model_dim, elementwise_affine=True)
        self.adapter = NaiveAdapter(
            model_dim,
            bottleneck_size1,
            )
        self.bottleneck_size2 = bottleneck_size2
        if bottleneck_size2 > 0:
            self.lang_adapters=torch.nn.ModuleDict([])
            for lang in num_langs:
                self.lang_adapters[lang]= NaiveAdapter(
                    model_dim,
                    bottleneck_size2,
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
 
        if side == "moa" or (not self.training):
            x = self.adapter(x)
            # x = self.lang_adapters[lang_id](x)
        else:
            if self.bottleneck_size2 > 0:
                x = self.lang_adapters[lang_id](x)
            else:
                x = self.adapter(x)

        return x, None

class SingleAdapterLayer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        self.adapter = NaiveAdapter(
            model_dim,
            bottleneck_size,
            )
    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual
        x = self.adapter(x)
        return x, None

class LangMoALayer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
        num_adapters: int,
        num_langs: int,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        self.adapters=torch.nn.ModuleList([])
        self.model_dim = model_dim
        for _ in range(num_adapters):
            self.adapters.append(
                NaiveAdapter(
                    model_dim,
                    bottleneck_size,
                    )
                )
        self.gate = FairLinear(model_dim, num_adapters)
        self.lang_classifer = FairLinear(model_dim, num_langs)
    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        lang_id=None,
        side=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual

        mean_x = torch.mean(x.reshape(-1, self.model_dim), dim=0) #[dim]
        gate_logit = F.softmax(self.gate(mean_x), dim=-1) # [num_gate]
        gate_id = torch.argmax(gate_logit)
        gate_score = gate_logit[gate_id]
        x = self.adapters[gate_id](x) * (gate_score / gate_score.data.detach())

        mean_x = torch.mean(x.reshape(-1, self.model_dim), dim=0) #[dim]
        lang_logit = self.lang_classifer(mean_x) # [num_lang]
        lid = F.cross_entropy(lang_logit, lang_id.squeeze(0), label_smoothing=0.1)
        l_aux = {"lid_loss": lid}

        return x, l_aux

class LUAPLUSLayer(torch.nn.Module):
    def __init__(
        self,
        ffn_fn: Callable,
        model_dim: int,
        bottleneck_size: int,
        num_langs: int,
    ) -> None:
        super().__init__()
        self.ffn_fn = ffn_fn
        self.layer_norm = LayerNorm(model_dim, elementwise_affine=True)
        self.model_dim = model_dim
        self.adapter = NaiveAdapter(
            model_dim,
            bottleneck_size,
            )
        self.lang_embedding = torch.nn.Embedding(num_langs, model_dim)
        self.lang_classifer = FairLinear(model_dim, num_langs + 1)
        self.num_langs = num_langs
    def forward(
        self,
        *x: torch.Tensor,
        residual: torch.Tensor,
        lang_id=None,
        side=None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(x) == 1, "only single input Tensor supported"
        residual = residual.transpose(0,1)
        x = self.ffn_fn(*x)
        x = x + residual
 
        if side == "moa" or (not self.training):
            lang_embed = self.lang_embedding(lang_id)
            x = self.adapter(x + lang_embed)
            mean_x = torch.mean(x.reshape(-1, self.model_dim), dim=0) #[dim]
            lang_logit = self.lang_classifer(mean_x) # [num_lang]
            lid = F.cross_entropy(lang_logit, lang_id.squeeze(0))
            l_aux = {"lid_loss": lid}
        else:
            x = self.adapter(x)
            mean_x = torch.mean(x.reshape(-1, self.model_dim), dim=0) #[dim]
            lang_logit = self.lang_classifer(mean_x) # [num_lang]
            lid = F.cross_entropy(lang_logit, torch.ones_like(lang_id).squeeze(0) * self.num_langs)
            l_aux = {"lid_loss": lid}

        return x, l_aux

# Original LUA version
# class LUALayer(torch.nn.Module):
#     def __init__(
#         self,
#         ffn_fn: Callable,
#         model_dim: int,
#         bottleneck_size1: int,
#         bottleneck_size2: int,
#         num_langs: List,
#     ) -> None:
#         super().__init__()
#         self.ffn_fn = ffn_fn
#         self.layer_norm = LayerNorm(model_dim, elementwise_affine=True)
#         self.lang_adapters=torch.nn.ModuleDict([])
#         self.adapter = NaiveAdapter(
#             model_dim,
#             bottleneck_size1,
#             )
#         for lang in num_langs:
#             self.lang_adapters[lang]= NaiveAdapter(
#                 model_dim,
#                 bottleneck_size2,
#                 )
#     def forward(
#         self,
#         *x: torch.Tensor,
#         residual: torch.Tensor,
#         prefix_tokens=None,
#         lang_id=None,
#         side=None,
#         **kwargs: Any
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         assert len(x) == 1, "only single input Tensor supported"
#         residual = residual.transpose(0,1)
#         x = self.ffn_fn(*x)
#         x = x + residual
 
#         if side == "moa" or (not self.training):
#             x = self.adapter(x)
#         else:
#             x = self.lang_adapters[lang_id](x)
#         return x, None


# v1:
class L0Adapter(torch.nn.Module):
    def __init__(
        self,
        input_size,
        bottleneck_size,
        num_langs,
        init_mean=0,
        init_sdev=0.01,
        init_beta=2/3,
        zeta=1.1,
        gamma=-0.1,
        epsilon=1e-6
        ):
        super().__init__()
        self.size = input_size
        self.down = Linear(input_size, bottleneck_size)
        self.up = Linear(bottleneck_size, input_size)
        self.layer_norm = LayerNorm(input_size, elementwise_affine=True)
        self.activation_fn = get_activation_fn("relu")
        self.zeta = zeta
        self.gamma = gamma
        self.epsilon = epsilon
        self.down_loga = torch.nn.Parameter(torch.zeros(num_langs, input_size).normal_(init_mean, init_sdev)
        )
        self.up_loga = torch.nn.Parameter(torch.zeros(num_langs, bottleneck_size).normal_(init_mean, init_sdev)
        )
        self.down_beta = init_beta #torch.nn.Parameter(torch.zeros(num_langs).fill_(init_beta))
        self.up_beta = init_beta #torch.nn.Parameter(torch.zeros(num_langs).fill_(init_beta))
        
    def forward(self, x, lang_id):
        shortcut = x
        x = self.layer_norm(x)

        ## language L0 for down
        if self.training:
            down_mask = self.sample_and_get_masks(x, lang_id, self.down_loga, self.down_beta)
        else:
            down_mask = F.hardtanh(torch.sigmoid(self.down_loga[lang_id]) * (self.zeta - self.gamma) + self.gamma, min_val=0, max_val=1)
        x = F.linear(x, down_mask.view(1, -1) * self.down.weight, self.down.bias)

        x = self.activation_fn(x)

        # language L0 for up
        if self.training:
            up_mask = self.sample_and_get_masks(x, lang_id, self.up_loga, self.up_beta)
        else:
            up_mask = F.hardtanh(torch.sigmoid(self.up_loga[lang_id]) * (self.zeta - self.gamma) + self.gamma, min_val=0, max_val=1)
        x = F.linear(x, up_mask.view(1, -1) * self.up.weight, self.up.bias)

        x = x + shortcut
        return x

    def sample_and_get_masks(self, x, lang_id, loga, beta):
        u = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device).uniform_(self.epsilon, 1-self.epsilon)
        s = torch.sigmoid((torch.log(u) - torch.log(1-u) + loga[lang_id]) / beta)
        s = s * (self.zeta - self.gamma) + self.gamma
        s = F.hardtanh(s, min_val=0, max_val=1)
        return s