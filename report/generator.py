import json
import datetime
from pathlib import Path

DIMENSION_LABELS = [
    "Dialogue", "Pacing", "Char. Consistency", "Thematic Resonance", "Dramatic Tension"
]
DIMENSION_KEYS = ["dialogue", "pacing", "character_consistency", "theme", "tension"]

ITER_COLORS = [
    ("rgba(124,58,237,0.35)", "rgba(124,58,237,1)"),   # purple
    ("rgba(6,182,212,0.35)",  "rgba(6,182,212,1)"),    # cyan
    ("rgba(234,179,8,0.35)",  "rgba(234,179,8,1)"),    # yellow
    ("rgba(249,115,22,0.35)", "rgba(249,115,22,1)"),   # orange
    ("rgba(34,197,94,0.35)",  "rgba(34,197,94,1)"),    # green
]


def _score_color(score: int) -> str:
    if score >= 8:
        return "#22c55e"
    if score >= 6:
        return "#eab308"
    return "#ef4444"


def generate_report(
    prompt: str,
    final_script: str,
    iteration_history: list,
    final_score: int,
    timestamp: str,
    output_dir: str = ".",
    save: bool = True,
) -> str:
    """Generate the HTML report.

    Args:
        save: If True, writes report_{timestamp}.html to output_dir and returns
              the filename. If False, returns the raw HTML string.
    """
    radar_datasets = []
    for i, rec in enumerate(iteration_history):
        bg, border = ITER_COLORS[i % len(ITER_COLORS)]
        radar_datasets.append({
            "label": f"Iteration {i + 1}",
            "data": [rec["scores"][k] for k in DIMENSION_KEYS],
            "backgroundColor": bg,
            "borderColor": border,
            "borderWidth": 2,
            "pointBackgroundColor": border,
            "pointRadius": 4,
        })

    overall_scores = [r["scores"]["overall"] for r in iteration_history]
    line_labels = [f"Iter {i+1}" for i in range(len(iteration_history))]

    # Escape user-supplied text to prevent XSS in the HTML report
    def _html_escape(text: str) -> str:
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    prompt_escaped = _html_escape(prompt)

    iteration_cards_html = ""
    for i, rec in enumerate(iteration_history):
        s = rec["scores"]
        bars = ""
        for key, label in zip(DIMENSION_KEYS, DIMENSION_LABELS):
            v = s[key]
            col = _score_color(v)
            bars += f"""
            <div class="bar-row">
              <span class="bar-label">{label}</span>
              <div class="bar-track">
                <div class="bar-fill" style="width:{v*10}%;background:{col}"></div>
              </div>
              <span class="bar-val" style="color:{col}">{v}/10</span>
            </div>"""

        imp_items = "".join(
            f'<li>{_html_escape(imp)}</li>' for imp in rec.get("improvements", [])
        )

        defense_notes_html = ""
        if rec.get('defense_notes'):
            defense_html = _html_escape(rec['defense_notes']).replace(chr(10), '<br>')
            defense_notes_html = f"""
            <div class="defense-box">
              <h4>Writer's Defense Notes</h4>
              <p>{defense_html}</p>
            </div>"""

        iteration_cards_html += f"""
        <div class="iter-card">
          <div class="iter-header">
            <span class="iter-badge">Iteration {i+1}</span>
            <span class="overall-pill" style="background:{_score_color(s['overall'])}22;
                  color:{_score_color(s['overall'])};border:1px solid {_score_color(s['overall'])}">
              Overall {s['overall']}/10
            </span>
          </div>
          <div class="dim-bars">{bars}</div>
          <div class="critique-box">
            <h4>Critique</h4>
            <p>{_html_escape(rec.get('critique_text','')).replace(chr(10),'<br>')}</p>
          </div>
          <div class="improvements-box">
            <h4>Required Improvements</h4>
            <ul>{imp_items}</ul>
          </div>
          {defense_notes_html}
        </div>"""

    script_escaped = (
        final_script
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Script Doctor — Run Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0c12;color:#e2e8f0;font-family:'Inter',sans-serif;min-height:100vh;padding:2rem}}
.page-wrap{{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:2rem}}

/* ── header ── */
.report-header{{background:linear-gradient(135deg,rgba(124,58,237,.15),rgba(6,182,212,.08));
  border:1px solid rgba(124,58,237,.3);border-radius:16px;padding:2rem 2.5rem}}
.report-title{{font-size:1.7rem;font-weight:700;background:linear-gradient(90deg,#7c3aed,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.4rem}}
.report-meta{{display:flex;gap:2rem;flex-wrap:wrap;margin-top:1rem}}
.meta-chip{{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
  border-radius:8px;padding:.35rem .8rem;font-size:.82rem;color:#94a3b8}}
.meta-chip strong{{color:#e2e8f0}}
.prompt-box{{margin-top:1.2rem;background:rgba(0,0,0,.3);border-left:3px solid #7c3aed;
  border-radius:0 8px 8px 0;padding:.8rem 1rem;font-size:.9rem;color:#cbd5e1;font-style:italic}}

/* ── pipeline diagram ── */
.pipeline{{display:flex;align-items:center;justify-content:center;gap:0;
  background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:1.5rem}}
.pipe-node{{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
  border-radius:10px;padding:.8rem 1.4rem;text-align:center;min-width:110px}}
.pipe-node .icon{{font-size:1.4rem;margin-bottom:.3rem}}
.pipe-node .label{{font-size:.78rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.pipe-arrow{{color:#4b5563;font-size:1.2rem;padding:0 .5rem;flex-shrink:0}}
.pipe-loop{{font-size:.7rem;color:#7c3aed;margin-top:.2rem}}
.final-score-node{{background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(6,182,212,.1));
  border-color:rgba(124,58,237,.4)}}
.final-score-num{{font-size:1.5rem;font-weight:700;color:{_score_color(final_score)}}}

/* ── section titles ── */
.section-title{{font-size:1rem;font-weight:600;color:#94a3b8;text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}}
.section-title::before{{content:'';display:block;width:3px;height:1em;
  background:linear-gradient(#7c3aed,#06b6d4);border-radius:2px}}

/* ── charts ── */
.charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
@media(max-width:700px){{.charts-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:14px;padding:1.5rem}}
.chart-card h3{{font-size:.9rem;color:#94a3b8;margin-bottom:1rem;font-weight:500}}
.chart-wrap{{position:relative;height:280px}}

/* ── iteration cards ── */
.iter-card{{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:14px;padding:1.5rem;display:flex;flex-direction:column;gap:1.2rem}}
.iter-header{{display:flex;align-items:center;gap:1rem}}
.iter-badge{{background:rgba(124,58,237,.2);color:#a78bfa;border:1px solid rgba(124,58,237,.4);
  border-radius:20px;padding:.3rem .9rem;font-size:.82rem;font-weight:600}}
.overall-pill{{border-radius:20px;padding:.3rem .9rem;font-size:.82rem;font-weight:600}}
.dim-bars{{display:flex;flex-direction:column;gap:.55rem}}
.bar-row{{display:grid;grid-template-columns:140px 1fr 52px;align-items:center;gap:.75rem}}
.bar-label{{font-size:.8rem;color:#94a3b8;text-align:right}}
.bar-track{{height:7px;background:rgba(255,255,255,.08);border-radius:4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;transition:width .6s ease}}
.bar-val{{font-size:.82rem;font-weight:600;text-align:right}}
.critique-box,.improvements-box,.defense-box{{background:rgba(0,0,0,.25);border-radius:10px;padding:1rem}}
.critique-box h4,.improvements-box h4,.defense-box h4{{font-size:.78rem;text-transform:uppercase;
  letter-spacing:.07em;color:#64748b;margin-bottom:.6rem;font-weight:600}}
.defense-box h4{{color:#a78bfa}}
.critique-box p,.defense-box p{{font-size:.88rem;color:#cbd5e1;line-height:1.65}}
.improvements-box ul{{padding-left:1.2rem;display:flex;flex-direction:column;gap:.4rem}}
.improvements-box li{{font-size:.87rem;color:#cbd5e1;line-height:1.55}}

/* ── final script ── */
.script-card{{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:2rem}}
.script-pre{{font-family:'JetBrains Mono',monospace;font-size:.82rem;line-height:1.75;
  color:#cbd5e1;white-space:pre-wrap;word-break:break-word}}

/* ── footer ── */
.report-footer{{text-align:center;font-size:.78rem;color:#374151;padding:1rem 0}}
</style>
</head>
<body>
<div class="page-wrap">

  <!-- Header -->
  <div class="report-header">
    <div class="report-title">Script Doctor — Run Report</div>
    <div class="report-meta">
      <span class="meta-chip">Generated <strong>{timestamp}</strong></span>
      <span class="meta-chip">Iterations <strong>{len(iteration_history)}</strong></span>
      <span class="meta-chip">Final Score <strong style="color:{_score_color(final_score)}">{final_score}/10</strong></span>
      <span class="meta-chip">Exit Condition <strong>{"Score &ge; 8" if final_score >= 8 else "Iteration cap"}</strong></span>
    </div>
    <div class="prompt-box">{prompt_escaped}</div>
  </div>

  <!-- Pipeline Diagram -->
  <div>
    <div class="section-title">Pipeline Flow</div>
    <div class="pipeline">
      <div class="pipe-node"><div class="icon">&#128221;</div><div class="label">Init RAG</div></div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-node"><div class="icon">&#9999;&#65039;</div><div class="label">Writer</div></div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-node">
        <div class="icon">&#128269;</div><div class="label">Critic</div>
        <div class="pipe-loop">&#8635; loops if score &lt; 8</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-node final-score-node">
        <div class="final-score-num">{final_score}/10</div>
        <div class="label">Final</div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div>
    <div class="section-title">Score Analysis</div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Dimension Radar — All Iterations</h3>
        <div class="chart-wrap"><canvas id="radarChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>Overall Score Progression</h3>
        <div class="chart-wrap"><canvas id="lineChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Iteration Cards -->
  <div>
    <div class="section-title">Iteration Breakdown</div>
    <div style="display:flex;flex-direction:column;gap:1.25rem">
      {iteration_cards_html}
    </div>
  </div>

  <!-- Final Script -->
  <div class="script-card">
    <div class="section-title">Final Approved Script</div>
    <pre class="script-pre">{script_escaped}</pre>
  </div>

  <div class="report-footer">Generated by Script Doctor &mdash; Multi-Agent LangGraph Pipeline</div>
</div>

<script>
const radarData = {{
  labels: {json.dumps(DIMENSION_LABELS)},
  datasets: {json.dumps(radar_datasets)}
}};
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: radarData,
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{ r: {{
      min: 0, max: 10,
      ticks: {{ stepSize: 2, color: '#64748b', backdropColor: 'transparent', font: {{size: 10}} }},
      grid: {{ color: 'rgba(255,255,255,0.06)' }},
      pointLabels: {{ color: '#94a3b8', font: {{size: 11}} }},
      angleLines: {{ color: 'rgba(255,255,255,0.06)' }}
    }} }},
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{size: 11}}, boxWidth: 12 }} }} }}
  }}
}});

const lineData = {{
  labels: {json.dumps(line_labels)},
  datasets: [{{
    label: 'Overall Score',
    data: {json.dumps(overall_scores)},
    borderColor: 'rgba(124,58,237,1)',
    backgroundColor: 'rgba(124,58,237,0.15)',
    borderWidth: 2.5,
    pointBackgroundColor: 'rgba(124,58,237,1)',
    pointRadius: 6,
    fill: true,
    tension: 0.3
  }}, {{
    label: 'Threshold (8)',
    data: Array({len(iteration_history)}).fill(8),
    borderColor: 'rgba(34,197,94,0.5)',
    borderWidth: 1.5,
    borderDash: [6,4],
    pointRadius: 0,
    fill: false,
  }}]
}};
new Chart(document.getElementById('lineChart'), {{
  type: 'line',
  data: lineData,
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      y: {{ min: 0, max: 10, ticks: {{ stepSize: 2, color:'#64748b' }},
             grid: {{ color:'rgba(255,255,255,0.06)' }} }},
      x: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'rgba(255,255,255,0.04)' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color:'#94a3b8', font:{{size:11}}, boxWidth:12 }} }} }}
  }}
}});
</script>
</body>
</html>"""

    if not save:
        return html

    ts_safe = timestamp.replace(":", "-").replace(" ", "_")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"report_{ts_safe}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return str(filename)
