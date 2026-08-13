import os
import torch
import numpy as np

from .models import get_moe_layers


class RoutingRecorder:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.records = {}
        self.handles = []
        for layer_key, moe_mod in get_moe_layers(model, cfg):
            gate = getattr(moe_mod, cfg["gate_attr"])
            h = gate.register_forward_hook(self._make_hook(layer_key))
            self.handles.append(h)

    def _make_hook(self, layer_key):
        arch = self.cfg["architecture"]

        def hook(module, inputs, output):
            if arch == "switch":
                router_probs, expert_index_onehot, _ = output
                was_routed = expert_index_onehot.sum(dim=-1) > 0
                idx = expert_index_onehot.argmax(dim=-1)
                idx = torch.where(was_routed, idx, torch.full_like(idx, -1))

                topk_idx = idx.unsqueeze(-1).detach().to("cpu")
                topk_weight = router_probs.detach().to("cpu")
            elif arch == "deepseek":
                topk_idx, topk_weight = output[0], output[1]
            elif arch == "mixtral":
                logits = output
                weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
                topk_weight, topk_idx = torch.topk(weights, self.cfg["top_k"], dim=-1)
            else:
                raise ValueError(f"Unknown architecture: {arch}")

            self.records[layer_key] = {
                "topk_idx": topk_idx,
                "topk_weight": topk_weight,
            }

        return hook

    def clear(self):
        self.records = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class ExpertActivationRecorder:
    def __init__(self, model, cfg, layer_key):
        self.cfg = cfg
        self.layer_key = layer_key  
        self.buffers = {e: [] for e in range(cfg["n_routed_experts"])}
        self.handles = []
        moe_layers = dict(get_moe_layers(model, cfg))
        moe_mod = moe_layers[layer_key]
        experts = getattr(moe_mod, cfg["experts_attr"])

        if hasattr(experts, "items"): 
            for e_idx in range(cfg["n_routed_experts"]):
                expert = experts[f"expert_{e_idx}"]
                h = expert.register_forward_pre_hook(self._make_hook(e_idx))
                self.handles.append(h)
        else:  # nn.ModuleList (other architectures)
            for e_idx, expert in enumerate(experts):
                h = expert.register_forward_pre_hook(self._make_hook(e_idx))
                self.handles.append(h)

    def _make_hook(self, expert_idx):
        def hook(module, inputs):
            x = inputs[0]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.shape[0] > 0:
                self.buffers[expert_idx].append(x.detach().to("cpu", torch.float16))
        return hook

    def flush_to_disk(self, cache_dir, tag=""):
        out_dir = os.path.join(cache_dir, f"layer_{self.layer_key}")
        os.makedirs(out_dir, exist_ok=True)
        for e_idx, chunks in self.buffers.items():
            if not chunks:
                continue
            arr = torch.cat(chunks, dim=0).numpy()
            fname = os.path.join(out_dir, f"expert_{e_idx}{tag}.npy")
            if os.path.exists(fname):
                prev = np.load(fname)
                arr = np.concatenate([prev, arr], axis=0)
            np.save(fname, arr)
            self.buffers[e_idx] = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
