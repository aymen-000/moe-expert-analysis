import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_interp.models import load_model, get_moe_layers, make_dirs
from moe_interp.hooks import RoutingRecorder, ExpertActivationRecorder
from moe_interp.data import PILE_DOMAINS, load_pile_domains
from moe_interp.routing import collect_and_cache_expert_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="./moe_interp_out")
    parser.add_argument("--model", default="switch_base_8")
    parser.add_argument("--layers", nargs="+", default=None,
                         help="Layer keys to collect, e.g. enc_0 enc_5 enc_11. "
                              "Defaults to [first, middle, last] encoder MoE layers.")
    parser.add_argument("--max-chunks-per-domain", type=int, default=150)
    args = parser.parse_args()

    dirs = make_dirs(args.base_dir)
    tokenizer, model, cfg = load_model(args.model)

    enc_layer_keys = [k for k, _ in get_moe_layers(model, cfg) if k.startswith("enc_")]
    layers = args.layers or [enc_layer_keys[0], enc_layer_keys[len(enc_layer_keys) // 2], enc_layer_keys[-1]]
    print("Collecting activations for layers:", layers)

    domain_data = load_pile_domains(PILE_DOMAINS)

    for layer_key in layers:
        print(f"\n=== {layer_key} ===")
        routing_recorder = RoutingRecorder(model, cfg)
        act_recorder = ExpertActivationRecorder(model, cfg, layer_key)
        collect_and_cache_expert_activations(
            model, tokenizer, cfg, domain_data, layer_key,
            routing_recorder, act_recorder, dirs["cache"],
            max_chunks=args.max_chunks_per_domain,
        )
        routing_recorder.remove()
        act_recorder.remove()


if __name__ == "__main__":
    main()
