import os
import html
import json
from collections import Counter


def generate_interactive_dashboard(dashboard_data, out_path, title="SAE Feature Dashboard"):
    payload = json.dumps(dashboard_data)

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#f7f7f8; color:#1a1a1a; }}
  #layout {{ display:flex; height:100vh; }}
  #sidebar {{ width:340px; border-right:1px solid #ddd; background:#fff; display:flex; flex-direction:column; }}
  #sidebar-header {{ padding:12px; border-bottom:1px solid #eee; }}
  #search {{ width:100%; padding:6px 8px; border:1px solid #ccc; border-radius:6px; font-size:13px; }}
  #sort-select {{ width:100%; margin-top:6px; padding:5px; font-size:12px; }}
  #feature-list {{ overflow-y:auto; flex:1; }}
  .feature-row {{ padding:10px 12px; border-bottom:1px solid #f0f0f0; cursor:pointer; font-size:12px; }}
  .feature-row:hover {{ background:#f2f6ff; }}
  .feature-row.active {{ background:#e3edff; border-left:3px solid #3b6fe0; }}
  .feature-row .fid {{ font-weight:600; font-size:13px; }}
  .feature-row .meta {{ color:#666; margin-top:2px; }}
  .badge {{ display:inline-block; background:#eee; border-radius:4px; padding:1px 6px; margin-right:4px; font-size:11px; }}
  #main {{ flex:1; overflow-y:auto; padding:24px 32px; }}
  #feature-title {{ font-size:20px; font-weight:700; margin-bottom:4px; }}
  #feature-stats {{ color:#555; font-size:13px; margin-bottom:20px; }}
  .example-row {{ background:#fff; border:1px solid #eee; border-radius:8px; padding:10px 14px; margin-bottom:8px; }}
  .example-domain {{ display:inline-block; color:#888; font-size:11px; width:130px; }}
  .example-text {{ font-family: ui-monospace, Menlo, monospace; font-size:13px; line-height:1.8; }}
  .example-act {{ float:right; color:#aaa; font-size:11px; }}
  #placeholder {{ color:#999; padding:40px; text-align:center; }}
  #count-label {{ font-size:11px; color:#888; padding:8px 12px 0; }}
</style>
</head>
<body>
<div id="layout">
  <div id="sidebar">
    <div id="sidebar-header">
      <input id="search" type="text" placeholder="Search feature id or domain...">
      <select id="sort-select">
        <option value="domain_share">Sort: domain concentration</option>
        <option value="density_asc">Sort: rarest first (density)</option>
        <option value="density_desc">Sort: most frequent first</option>
        <option value="max_act">Sort: strongest activation</option>
        <option value="feature_id">Sort: feature id</option>
      </select>
    </div>
    <div id="count-label"></div>
    <div id="feature-list"></div>
  </div>
  <div id="main">
    <div id="placeholder">Select a feature on the left to inspect its activating tokens.</div>
  </div>
</div>

<script>
const DATA = {payload};
let filtered = DATA.slice();
let activeFeature = null;

function domainColor(domain) {{
  let hash = 0;
  for (let i = 0; i < domain.length; i++) hash = domain.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${{hue}}, 55%, 45%)`;
}}

function renderList() {{
  const listEl = document.getElementById('feature-list');
  listEl.innerHTML = '';
  document.getElementById('count-label').textContent = `${{filtered.length}} features`;
  filtered.forEach(f => {{
    const row = document.createElement('div');
    row.className = 'feature-row' + (activeFeature === f.feature ? ' active' : '');
    row.innerHTML = `
      <div class="fid">Feature ${{f.feature}}</div>
      <div class="meta">
        <span class="badge">${{f.top_domain}} ${{Math.round(f.top_domain_share*100)}}%</span>
        <span class="badge">density ${{f.density.toFixed(4)}}</span>
        <span class="badge">${{f.n_examples}} ex.</span>
      </div>`;
    row.onclick = () => {{ activeFeature = f.feature; renderList(); renderDetail(f); }};
    listEl.appendChild(row);
  }});
}}

function renderDetail(f) {{
  const main = document.getElementById('main');
  let html = `
    <div id="feature-title">Feature ${{f.feature}}</div>
    <div id="feature-stats">
      density: ${{f.density.toFixed(5)}} &nbsp;|&nbsp;
      max activation: ${{f.max_act.toFixed(2)}} &nbsp;|&nbsp;
      top domain: ${{f.top_domain}} (${{Math.round(f.top_domain_share*100)}}%)
    </div>`;

  f.examples.forEach(ex => {{
    let toks = '';
    ex.context_tokens.forEach((t, i) => {{
      const isCenter = i === ex.center_idx;
      toks += `<span class="tok${{isCenter ? ' center' : ''}}" style="${{isCenter ? 'background: rgba(255,120,0,0.35);' : ''}}">${{t}}</span>`;
    }});
    html += `
      <div class="example-row">
        <span class="example-domain">[${{ex.domain}}]</span>
        <span class="example-act">act=${{ex.activation.toFixed(3)}}</span>
        <div class="example-text">${{toks}}</div>
      </div>`;
  }});

  main.innerHTML = html;
}}

function applyFilterSort() {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  const sortMode = document.getElementById('sort-select').value;

  filtered = DATA.filter(f =>
    q === '' || String(f.feature).includes(q) || f.top_domain.toLowerCase().includes(q)
  );

  const sorters = {{
    domain_share: (a, b) => b.top_domain_share - a.top_domain_share,
    density_asc: (a, b) => a.density - b.density,
    density_desc: (a, b) => b.density - a.density,
    max_act: (a, b) => b.max_act - a.max_act,
    feature_id: (a, b) => a.feature - b.feature,
  }};
  filtered.sort(sorters[sortMode]);
  renderList();
}}

document.getElementById('search').addEventListener('input', applyFilterSort);
document.getElementById('sort-select').addEventListener('change', applyFilterSort);
applyFilterSort();
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Dashboard saved: {out_path}  ({len(dashboard_data)} features, {os.path.getsize(out_path)/1024:.0f} KB)")


def build_dashboard_payload(top_by_expert, layer_key, max_examples_in_dashboard=20, max_tokens_in_cloud=12):
    payload = []
    for expert_id in sorted(top_by_expert.keys()):
        for feat in top_by_expert[expert_id]:
            examples = feat["examples"][:max_examples_in_dashboard]
            tok_counts = Counter(ex["target_token"].strip() for ex in feat["examples"])
            top_tokens = [{"token": t, "count": c} for t, c in tok_counts.most_common(max_tokens_in_cloud)]
            payload.append({
                "layer": layer_key,
                "expert_id": expert_id,
                "feature_id": feat["feature_id"],
                "density": feat["density"],
                "mean_activation": round(feat.get("mean_activation", 0.0), 4),
                "n_examples_total": feat.get("n_examples", len(feat["examples"])),
                "top_tokens": top_tokens,
                "examples": examples,
            })
    return payload


def generate_expert_feature_dashboard(payload, out_path, title):
    """Multi-expert dashboard (grouped sidebar by expert) showing top
    features per expert with a token-frequency cloud + example contexts.
    `payload` comes from build_dashboard_payload.
    """
    data_json = json.dumps(payload)
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#f7f7f8; color:#1a1a1a; }}
  #layout {{ display:flex; height:100vh; }}
  #sidebar {{ width:320px; border-right:1px solid #ddd; background:#fff; display:flex; flex-direction:column; }}
  #sidebar-header {{ padding:12px; border-bottom:1px solid #eee; }}
  #search {{ width:100%; padding:6px 8px; border:1px solid #ccc; border-radius:6px; font-size:13px; }}
  #feature-list {{ overflow-y:auto; flex:1; }}
  .expert-group-header {{ padding:8px 12px; background:#f0f2f5; font-weight:700; font-size:12px;
                           color:#444; position:sticky; top:0; border-bottom:1px solid #e5e5e5; }}
  .feature-row {{ padding:9px 12px; border-bottom:1px solid #f0f0f0; cursor:pointer; font-size:12px; }}
  .feature-row:hover {{ background:#f2f6ff; }}
  .feature-row.active {{ background:#e3edff; border-left:3px solid #3b6fe0; }}
  .feature-row .fid {{ font-weight:600; font-size:13px; }}
  .feature-row .meta {{ color:#666; margin-top:2px; }}
  .badge {{ display:inline-block; background:#eee; border-radius:4px; padding:1px 6px; margin-right:4px; font-size:11px; }}
  #main {{ flex:1; overflow-y:auto; padding:24px 32px; }}
  #feature-title {{ font-size:20px; font-weight:700; margin-bottom:4px; }}
  #feature-stats {{ color:#555; font-size:13px; margin-bottom:16px; }}
  #token-cloud {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:22px; }}
  .token-chip {{ background:#eef2ff; border:1px solid #dbe3fb; border-radius:14px;
                 padding:4px 10px; font-size:13px; font-family: ui-monospace, Menlo, monospace; }}
  .token-chip b {{ color:#3b6fe0; }}
  .example-row {{ background:#fff; border:1px solid #eee; border-radius:8px; padding:10px 14px; margin-bottom:8px; }}
  .example-domain {{ display:inline-block; color:#888; font-size:11px; width:140px; }}
  .example-text {{ font-family: ui-monospace, Menlo, monospace; font-size:13px; line-height:1.7; }}
  .example-act {{ float:right; color:#aaa; font-size:11px; }}
  .target-tok {{ background: rgba(255,120,0,0.35); font-weight:700; padding:1px 3px; border-radius:3px; }}
  #placeholder {{ color:#999; padding:40px; text-align:center; }}
  #count-label {{ font-size:11px; color:#888; padding:8px 12px 0; }}
</style>
</head>
<body>
<div id="layout">
  <div id="sidebar">
    <div id="sidebar-header">
      <input id="search" type="text" placeholder="Search feature id, expert, or token...">
    </div>
    <div id="count-label"></div>
    <div id="feature-list"></div>
  </div>
  <div id="main">
    <div id="placeholder">Select a feature on the left to inspect its most-fired tokens and examples.</div>
  </div>
</div>

<script>
const DATA = {data_json};
let filtered = DATA.slice();
let activeKey = null;

function keyOf(f) {{ return f.expert_id + "_" + f.feature_id; }}

function escapeHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function renderList() {{
  const listEl = document.getElementById('feature-list');
  listEl.innerHTML = '';
  document.getElementById('count-label').textContent = `${{filtered.length}} features`;

  let lastExpert = null;
  filtered.forEach(f => {{
    if (f.expert_id !== lastExpert) {{
      const header = document.createElement('div');
      header.className = 'expert-group-header';
      header.textContent = `Expert ${{f.expert_id}}`;
      listEl.appendChild(header);
      lastExpert = f.expert_id;
    }}
    const row = document.createElement('div');
    row.className = 'feature-row' + (activeKey === keyOf(f) ? ' active' : '');
    row.innerHTML = `
      <div class="fid">Feature ${{f.feature_id}}</div>
      <div class="meta">
        <span class="badge">density ${{f.density.toFixed(4)}}</span>
        <span class="badge">act ${{f.mean_activation.toFixed(2)}}</span>
        <span class="badge">${{f.n_examples_total}} ex.</span>
      </div>`;
    row.onclick = () => {{ activeKey = keyOf(f); renderList(); renderDetail(f); }};
    listEl.appendChild(row);
  }});
}}

function renderDetail(f) {{
  const main = document.getElementById('main');
  let html = `
    <div id="feature-title">Expert ${{f.expert_id}} — Feature ${{f.feature_id}}</div>
    <div id="feature-stats">
      layer: <b>${{f.layer}}</b> &nbsp;|&nbsp;
      density: ${{f.density.toFixed(5)}} &nbsp;|&nbsp;
      mean activation: ${{f.mean_activation.toFixed(3)}} &nbsp;|&nbsp;
      ${{f.n_examples_total}} examples collected (showing ${{f.examples.length}})
    </div>
    <div id="token-cloud">`;

  f.top_tokens.forEach(t => {{
    html += `<span class="token-chip">${{escapeHtml(t.token)}} <b>x${{t.count}}</b></span>`;
  }});
  html += `</div>`;

  f.examples.forEach(ex => {{
    const escCtx = escapeHtml(ex.context);
    const escTarget = escapeHtml(ex.target_token);
    const idx = escCtx.indexOf(escTarget);
    let marked;
    if (idx >= 0) {{
      marked = escCtx.slice(0, idx) +
               `<span class="target-tok">${{escTarget}}</span>` +
               escCtx.slice(idx + escTarget.length);
    }} else {{
      marked = escCtx;
    }}
    html += `
      <div class="example-row">
        <span class="example-domain">[${{escapeHtml(ex.domain)}}]</span>
        <span class="example-act">act=${{ex.activation.toFixed(3)}}</span>
        <div class="example-text">${{marked}}</div>
      </div>`;
  }});

  main.innerHTML = html;
}}

function applyFilter() {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  filtered = DATA.filter(f =>
    q === '' ||
    String(f.feature_id).includes(q) ||
    String(f.expert_id).includes(q) ||
    f.top_tokens.some(t => t.token.toLowerCase().includes(q))
  );
  renderList();
}}

document.getElementById('search').addEventListener('input', applyFilter);
applyFilter();
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Dashboard saved: {out_path}  ({len(payload)} features, {os.path.getsize(out_path)/1024:.0f} KB)")
