import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_interp.models import load_model, make_dirs
from moe_interp.hooks import RoutingRecorder
from moe_interp.data import load_language_samples
from moe_interp.routing import collect_language_routing
from moe_interp.viz import plot_language_radar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="./moe_interp_out")
    parser.add_argument("--model", default="switch_base_8")
    parser.add_argument("--layer", required=True, help="e.g. enc_5")
    parser.add_argument("--n-docs-per-language", type=int, default=100)
    args = parser.parse_args()

    dirs = make_dirs(args.base_dir)
    tokenizer, model, cfg = load_model(args.model)

    language_data = load_language_samples(n_docs=args.n_docs_per_language)

    recorder = RoutingRecorder(model, cfg)
    lang_expert_counts = collect_language_routing(model, tokenizer, language_data, args.layer, recorder)
    recorder.remove()

    plot_language_radar(
        lang_expert_counts, args.layer, cfg["n_routed_experts"],
        out_path=os.path.join(dirs["figures"], f"language_radar_{args.layer}.png"),
        json_path=os.path.join(dirs["figures"], f"language_radar_{args.layer}.json"),
    )


if __name__ == "__main__":
    main()
