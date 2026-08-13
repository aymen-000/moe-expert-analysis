import os
import re
import html
import json
import numpy as np
import matplotlib.pyplot as plt

from .utils import expert_colors
from .routing import run_encoder_and_record_all_layers


def plot_expert_distribution_by_domain(expert_counts, domains, plot_layers, n_experts, out_dir):
    fig, axes = plt.subplots(len(plot_layers), 1, figsize=(9, 3 * len(plot_layers)), sharex=True)
    if len(plot_layers) == 1:
        axes = [axes]

    x = np.arange(n_experts)
    width = 0.09
    json_results = {}

    for ax, lk in zip(axes, plot_layers):
        json_results[lk] = {}
        for i, domain in enumerate(domains):
            counts = expert_counts[domain][lk]
            total = counts.sum()
            prop = counts / total if total > 0 else counts
            ax.bar(x + i * width, prop, width=width, label=domain)
            json_results[lk][domain] = {
                "total_tokens": int(total),
                "experts": [
                    {"expert_id": int(e), "count": int(counts[e]), "selection_proportion": float(prop[e])}
                    for e in range(n_experts)
                ],
            }
        ax.axhline(1 / n_experts, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Layer: {lk}")
        ax.set_ylabel("Selection proportion")

    axes[-1].set_xlabel("Expert ID")
    axes[-1].set_xticks(x + width * (len(domains) - 1) / 2)
    axes[-1].set_xticklabels(x)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "expert_distribution_by_domain.png")
    plt.savefig(png_path, dpi=150)
    plt.show()

    json_path = os.path.join(out_dir, "expert_distribution_by_domain.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"Saved JSON to: {json_path}")
    return png_path, json_path


def _token_span(tok, e, colors):
    tok_str = html.escape(tok.replace("▁", " "))
    if e is None or e < 0:
        return f'<span style="padding:1px 2px;">{tok_str}</span>'
    r, g, b = colors[e]
    return (
        f'<span style="background-color: rgba({r},{g},{b},0.55); '
        f'padding:1px 2px; border-radius:2px;" title="expert {e}">{tok_str}</span>'
    )


def render_token_routing_multilayer(model, tokenizer, text, layer_keys, recorder,
                                     n_experts, max_len=256, save_path=None):
    colors = expert_colors(n_experts)
    ids = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]

    all_layer_idx = run_encoder_and_record_all_layers(model, ids, recorder)
    tokens = tokenizer.convert_ids_to_tokens(ids[0])

    rows_html = []
    for lk in layer_keys:
        layer_idx = all_layer_idx[lk]
        if layer_idx.numel() != len(tokens):
            raise ValueError(
                f"Mismatch: {len(tokens)} tokens vs {layer_idx.numel()} routing entries for layer {lk}"
            )
        spans = [_token_span(tok, int(e), colors) for tok, e in zip(tokens, layer_idx.tolist())]
        rows_html.append(f"""
            <div style="margin-bottom:18px;">
                <div style="font-size:12px; font-weight:bold; color:#444; margin-bottom:4px;">{lk}</div>
                <div style="font-family:monospace; font-size:13px; line-height:1.8;
                            max-width:1000px; white-space:pre-wrap; word-break:break-word;">
                    {''.join(spans)}
                </div>
            </div>
        """)

    body = "".join(rows_html)
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>MoE Routing Visualization</title></head>
    <body style="padding:20px; font-family:Arial;">
        <h2>MoE Expert Routing</h2>
        {body}
    </body>
    </html>
    """

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"Saved HTML to {save_path}")

    return html_doc


def get_token_expert_ratios(token_expert_counts, token, n_experts):
    counts = token_expert_counts.get(token, {})
    total = sum(counts.values())
    if total == 0:
        return np.zeros(n_experts)
    return np.array([counts.get(e, 0) / total for e in range(n_experts)])


def plot_token_routing_radar(token_expert_counts, tokens_to_plot, layer_key, n_experts, out_path=None):
    angles = np.linspace(0, 2 * np.pi, n_experts, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for token in tokens_to_plot:
        ratios = get_token_expert_ratios(token_expert_counts, token, n_experts)
        values = ratios.tolist()
        values += values[:1]
        label = token.replace("▁", " ").strip() or "(space)"
        ax.plot(angles, values, linewidth=1.3, label=label)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"E_{i}" for i in range(n_experts)], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(f"Routing decision at different Token IDs — {layer_key}", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, frameon=False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_language_radar(lang_expert_counts, layer_key, n_experts, out_path=None, json_path=None):
    angles = np.linspace(0, 2 * np.pi, n_experts, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    json_output = {"layer": layer_key, "n_experts": n_experts, "languages": {}}

    for lang_name, counts in lang_expert_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        ratios = [counts.get(e, 0) / total for e in range(n_experts)]
        top_expert = max(counts.items(), key=lambda kv: kv[1])[0]
        json_output["languages"][lang_name] = {
            "total_tokens": total,
            "expert_counts": {f"E_{e}": counts.get(e, 0) for e in range(n_experts)},
            "expert_ratios": {f"E_{e}": round(r, 5) for e, r in enumerate(ratios)},
            "top_expert": f"E_{top_expert}",
            "top_expert_share": round(counts[top_expert] / total, 5),
        }
        ratios_closed = ratios + ratios[:1]
        ax.plot(angles, ratios_closed, linewidth=1.2, label=lang_name)
        ax.fill(angles, ratios_closed, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"E_{i}" for i in range(n_experts)], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(f"Routing decision by language — {layer_key}", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8, frameon=False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"JSON saved: {json_path}")

    return json_output


_FNAME_RE = re.compile(r"^(?P<layer>.+)_expert(?P<expert>\d+)_features\.json$")


def discover_autointerp_files(autointerp_dir):
    """Scans autointerp_dir for `{layer}_expert{N}_features.json` files.
    Returns {layer_key: {expert_id: filepath}}.
    """
    available = {}
    for fname in sorted(os.listdir(autointerp_dir)):
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        available.setdefault(m.group("layer"), {})[int(m.group("expert"))] = os.path.join(autointerp_dir, fname)
    return available


def load_layer_features(available, layer_key, n_experts, metric="mean_activation"):

    out = {}
    present = available.get(layer_key, {})
    skipped = [e for e in range(n_experts) if e not in present]

    for expert_id, path in present.items():
        with open(path) as f:
            d = json.load(f)
        if not d.get("features"):
            skipped.append(expert_id)
            continue
        records = []
        for feat in d["features"]:
            acts = [ex["activation"] for ex in feat["examples"]]
            mean_act = float(np.mean(acts)) if acts else 0.0
            records.append({
                "feature_id": feat["feature_id"],
                "mean_activation": mean_act,
                "density": feat["density"],
            })
        out[expert_id] = records  

    return out, sorted(skipped)


def plot_top_features_per_expert(feats_by_expert, layer_key, n_experts, out_dir, out_stub,
                                  top_k=6, metric="mean_activation"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
    })

    cmap = plt.get_cmap("tab10" if n_experts <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n_experts)]

    experts_with_data = sorted(feats_by_expert.keys())
    if not experts_with_data:
        print(f"No experts with data for layer {layer_key} — skipping plot.")
        return None

    ncols = min(4, len(experts_with_data))
    nrows = int(np.ceil(len(experts_with_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), dpi=150, squeeze=False)
    axes = axes.reshape(-1)

    for ax_i, e in enumerate(experts_with_data):
        ax = axes[ax_i]
        records = sorted(feats_by_expert[e], key=lambda r: r[metric], reverse=True)[:top_k]
        values = [r[metric] for r in records]
        labels = [f"F{r['feature_id']}" for r in records]

        bars = ax.bar(range(len(records)), values, color=colors[e],
                       edgecolor="black", linewidth=0.6, width=0.65, zorder=3)
        for rect, v in zip(bars, values):
            ax.annotate(f"{v:.2f}" if metric == "mean_activation" else f"{v:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color="#222222")

        ax.set_xticks(range(len(records)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Expert {e}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        if ax_i % ncols == 0:
            ax.set_ylabel("Mean activation" if metric == "mean_activation" else "Density", fontsize=9)

    for j in range(len(experts_with_data), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Top-{top_k} Firing Feature IDs per Expert — Layer {layer_key}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_pdf = os.path.join(out_dir, f"{out_stub}.pdf")
    out_png = os.path.join(out_dir, f"{out_stub}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    return fig
