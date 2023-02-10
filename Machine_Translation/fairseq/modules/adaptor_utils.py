import ast
from collections.abc import Mapping
import json
import hashlib
import re
import os
from os.path import basename, isdir, isfile, join
from typing import Callable, Dict, List, Optional, Tuple, Union

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
HEAD_CONFIG_NAME = "head_config.json"
HEAD_WEIGHTS_NAME = "pytorch_model_head.bin"
ADAPTERFUSION_CONFIG_NAME = "adapter_fusion_config.json"
ADAPTERFUSION_WEIGHTS_NAME = "pytorch_model_adapter_fusion.bin"
EMBEDDING_FILE = "embedding.pt"
TOKENIZER_PATH = "tokenizer"

ADAPTER_HUB_URL = "https://raw.githubusercontent.com/Adapter-Hub/Hub/master/dist/v2/"
ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + "index/{}.json"
ADAPTER_HUB_CONFIG_FILE = ADAPTER_HUB_URL + "architectures.json"
ADAPTER_HUB_ALL_FILE = ADAPTER_HUB_URL + "all.json"
ADAPTER_HUB_ADAPTER_ENTRY_JSON = ADAPTER_HUB_URL + "adapters/{}/{}.json"

torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
# the download cache
ADAPTER_CACHE = join(torch_cache_home, "adapters")

# these keys are ignored when calculating the config hash
ADAPTER_CONFIG_HASH_IGNORE = []

# old: new
ACTIVATION_RENAME = {
    "gelu": "gelu_new",
    "gelu_orig": "gelu",
}
# HACK: To keep config hashs consistent with v2, remove default values of keys introduced in v3 from hash computation
ADAPTER_CONFIG_HASH_IGNORE_DEFAULT = {
    "phm_layer": True,
    "phm_dim": 4,
    "factorized_phm_W": True,
    "shared_W_phm": False,
    "shared_phm_rule": True,
    "factorized_phm_rule": False,
    "phm_c_init": "normal",
    "phm_init_range": 0.0001,
    "learn_phm": True,
    "hypercomplex_nonlinearity": "glorot-uniform",
    "phm_rank": 1,
    "phm_bias": True,
    "init_weights": "bert",
    "scaling": 1.0,
}
ADAPTER_CONFIG_STRING_PATTERN = re.compile(r"^(?P<name>[^\[\]\|\n]+)(?:\[(?P<kvs>.*)\])?$")


def _minimize_dict(d):
    if isinstance(d, Mapping):
        return {k: _minimize_dict(v) for (k, v) in d.items() if v}
    else:
        return d


def get_adapter_config_hash(config, length=16):
    """
    Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The resulting hash of the given config dict.
    """
    minimized_config = _minimize_dict({k: v for (k, v) in config.items() if k not in ADAPTER_CONFIG_HASH_IGNORE})
    # ensure hash is kept consistent to previous versions
    for name, default in ADAPTER_CONFIG_HASH_IGNORE_DEFAULT.items():
        if minimized_config.get(name, None) == default:
            del minimized_config[name]
    dict_str = json.dumps(minimized_config, sort_keys=True)
    h = hashlib.sha1()
    h.update(dict_str.encode(encoding="utf-8"))
    return h.hexdigest()[:length]

def parse_adapter_config_string(config_string: str) -> List[Tuple[str, dict]]:
    """
    Parses an adapter configuration string into a list of tuples. Each tuple constists of an adapter config identifier
    and dictionary.
    """
    # First split by "|" into individual adapter configs
    config_string_chunks = config_string.split("|")
    # Now match each adapter config against the regex
    adapter_configs = []
    for config_string_chunk in config_string_chunks:
        match = re.match(ADAPTER_CONFIG_STRING_PATTERN, config_string_chunk.strip())
        if not match or not match.group("name"):
            raise ValueError(f"Invalid adapter config string format: '{config_string_chunk}'.")
        name = match.group("name")
        if kvs := match.group("kvs"):
            # Replace "=" with ":" in key-value pairs for valid Python dict
            kvs = re.sub(r"(\w+)=", r"'\1':", kvs)
        else:
            kvs = ""
        # Now evaluate key-value pairs as Python dict
        try:
            config_kwargs = ast.literal_eval("{" + kvs + "}")
        except Exception:
            raise ValueError(f"Invalid adapter configguration '{kvs}' in '{name}'.")
        adapter_configs.append((name, config_kwargs))

    return adapter_configs


def resolve_adapter_config(config: Union[dict, str], local_map=None, **kwargs) -> dict:
    """
    Resolves a given adapter configuration specifier to a full configuration dictionary.

    Args:
        config (Union[dict, str]): The configuration to resolve. Can be either:

            - a dictionary: returned without further action
            - an identifier string available in local_map
            - the path to a file containing a full adapter configuration
            - an identifier string available in Adapter-Hub

    Returns:
        dict: The resolved adapter configuration dictionary.
    """
    # already a dict, so we don't have to do anything
    if isinstance(config, Mapping):
        return config
    # first, look in local map
    if local_map and config in local_map:
        return local_map[config]
    # load from file system if it's a local file
    if isfile(config):
        with open(config, "r") as f:
            loaded_config = json.load(f)
            # search for nested config if the loaded dict has the form of a config saved with an adapter module
            if "config" in loaded_config:
                return loaded_config["config"]
            else:
                return loaded_config
    # parse the config string
    config_pairs = parse_adapter_config_string(config)
    if len(config_pairs) > 0:
        full_configs = []
        for name, config_kwargs in config_pairs:
            # first, look in local map
            if local_map and name in local_map:
                config_obj = local_map[name]
                full_configs.append(config_obj.replace(**config_kwargs))
            else:
                raise ValueError("Could not identify '{}' as a valid adapter configuration.".format(name))
        # Case 1: only one config, return it directly
        if len(full_configs) == 1:
            return full_configs[0]
        # Case 2: multiple configs, return a config union
        elif len(full_configs) > 1:
            return {"architecture": "union", "configs": full_configs}

    raise ValueError("Could not identify '{}' as a valid adapter configuration.".format(config))

