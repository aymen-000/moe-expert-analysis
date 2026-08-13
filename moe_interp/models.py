import os
import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "switch_base_8": {
        "hf_name": "google/switch-base-8",
        "trust_remote_code": False,
        "n_layers": 12,
        "n_routed_experts": 8,
        "n_shared_experts": 0,
        "top_k": 1,  
        "hidden_size": 768,
        "moe_module_name_pattern": "mlp",  
        "gate_attr": "router",             
        "experts_attr": "experts",        
        "architecture": "switch",
        "encoder_decoder": True,
    },
    "switch_base_32": {
        "hf_name": "google/switch-base-32",
        "trust_remote_code": False,
        "n_layers": 12,
        "n_routed_experts": 32,
        "n_shared_experts": 0,
        "top_k": 1,
        "hidden_size": 768,
        "moe_module_name_pattern": "mlp",
        "gate_attr": "router",
        "experts_attr": "experts",
        "architecture": "switch",
        "encoder_decoder": True,
    },
    "switch_large_128": {
        "hf_name": "google/switch-large-128",
        "trust_remote_code": False,
        "n_layers": 24,
        "n_routed_experts": 128,
        "n_shared_experts": 0,
        "top_k": 1,
        "hidden_size": 1024,
        "moe_module_name_pattern": "mlp",
        "gate_attr": "router",
        "experts_attr": "experts",
        "architecture": "switch",
        "encoder_decoder": True,
    },
}


def load_model(model_key: str, device_map: str = "auto", dtype=torch.float16):
    """Load tokenizer + model for a registered model key.

    Returns (tokenizer, model, cfg).
    """
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"], trust_remote_code=cfg["trust_remote_code"])
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        cfg["hf_name"],
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=cfg["trust_remote_code"],
    )
    model.eval()
    return tokenizer, model, cfg


def get_moe_layers(model, cfg):
    out = []
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        for i, block in enumerate(model.encoder.block):
            ff_layer = block.layer[-1]  # SwitchTransformersLayerFF
            moe_mod = getattr(ff_layer, cfg["moe_module_name_pattern"], None)
            if moe_mod is not None and hasattr(moe_mod, cfg["experts_attr"]):
                out.append((f"enc_{i}", moe_mod))
    if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        for i, block in enumerate(model.decoder.block):
            ff_layer = block.layer[-1]
            moe_mod = getattr(ff_layer, cfg["moe_module_name_pattern"], None)
            if moe_mod is not None and hasattr(moe_mod, cfg["experts_attr"]):
                out.append((f"dec_{i}", moe_mod))
    return out


def make_dirs(base_dir: str):
    cache_dir = os.path.join(base_dir, "cache")
    sae_dir = os.path.join(base_dir, "sae_checkpoints")
    fig_dir = os.path.join(base_dir, "figures")
    autointerp_dir = os.path.join(base_dir, "autointerp")
    for d in (cache_dir, sae_dir, fig_dir, autointerp_dir):
        os.makedirs(d, exist_ok=True)
    return {
        "base": base_dir,
        "cache": cache_dir,
        "sae": sae_dir,
        "figures": fig_dir,
        "autointerp": autointerp_dir,
    }
