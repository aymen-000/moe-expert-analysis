import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_interp.models import load_model, get_moe_layers, make_dirs
from moe_interp.hooks import RoutingRecorder
from moe_interp.data import PILE_DOMAINS, load_pile_domains
from moe_interp.routing import collect_domain_routing_stats
from moe_interp.viz import plot_expert_distribution_by_domain, render_token_routing_multilayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="./moe_interp_out")
    parser.add_argument("--model", default="switch_base_8")
    parser.add_argument("--n-docs-per-domain", type=int, default=60)
    args = parser.parse_args()

    dirs = make_dirs(args.base_dir)
    tokenizer, model, cfg = load_model(args.model)

    enc_layer_keys = [k for k, _ in get_moe_layers(model, cfg) if k.startswith("enc_")]
    print("Encoder MoE layers:", enc_layer_keys)

    domain_data = load_pile_domains(PILE_DOMAINS, n_docs=args.n_docs_per_domain)

    recorder = RoutingRecorder(model, cfg)
    expert_counts, repeat_counts = collect_domain_routing_stats(
        model, tokenizer, cfg, domain_data, enc_layer_keys, recorder,
    )
    recorder.remove()

    n_experts = cfg["n_routed_experts"]
    plot_layers = [enc_layer_keys[0], enc_layer_keys[len(enc_layer_keys) // 2], enc_layer_keys[-1]]
    plot_expert_distribution_by_domain(expert_counts, PILE_DOMAINS, plot_layers, n_experts, dirs["figures"])

    print(f"\nExpected random repetition rate (top-1, {n_experts} experts): {100/n_experts:.1f}%")
    for domain in PILE_DOMAINS:
        row = []
        for lk in enc_layer_keys:
            n_same, n_total = repeat_counts[domain][lk]
            row.append(f"{lk}={100*n_same/n_total:.1f}%" if n_total > 0 else f"{lk}=n/a")
        print(f"  {domain}: " + ", ".join(row))

    # Token-level colored routing visualization across a few representative layers
    fig8_layers = [
        enc_layer_keys[0],
        enc_layer_keys[len(enc_layer_keys) // 3],
        enc_layer_keys[2 * len(enc_layer_keys) // 3],
        enc_layer_keys[-1],
    ]
    long_text = (
        "def forward(self, x):\n"
        "    hidden = self.linear1(x)\n"
        "    hidden = torch.relu(hidden)\n"
        "    return self.linear2(hidden)\n\n"
        "Question: What is the derivative of x^2 with respect to x? "
        "Answer: The derivative is 2x, obtained by applying the power rule.\n\n"
        "The mitochondria is the organelle responsible for producing ATP through "
        "oxidative phosphorylation, and is often described as the powerhouse of the cell."
    )
    recorder = RoutingRecorder(model, cfg)
    render_token_routing_multilayer(
        model, tokenizer, long_text, fig8_layers, recorder, n_experts,
        max_len=256, save_path=os.path.join(dirs["figures"], "moe_routing_visualization.html"),
    )
    recorder.remove()


if __name__ == "__main__":
    main()
