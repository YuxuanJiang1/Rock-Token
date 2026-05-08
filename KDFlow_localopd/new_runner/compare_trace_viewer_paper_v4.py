#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def load_trace(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return esc(x)


def clean_display_text(x: Any) -> str:
    """Clean common tokenizer artifacts for readable display."""
    if x is None:
        return ""
    s = str(x)
    # SentencePiece / BPE whitespace markers
    s = s.replace("Ġ", " ").replace("▁", " ")
    # GPT-style newline/tab byte markers
    s = s.replace("Ċ", "\n").replace("ĉ", "\t")
    # Avoid visually noisy special token wrappers when they appear in text
    s = s.replace("<｜end▁of▁sentence｜>", "")
    s = s.replace("<|endoftext|>", "")
    # Normalize repeated spaces but preserve newlines
    lines = [re.sub(r"[ \t]+", " ", line) for line in s.splitlines()]
    return "\n".join(lines)


def token_text_for_html(tok: Any) -> str:
    """Readable token text for inline highlighted view.

    Pure newline tokens become <br>; pure spaces are kept as a visible thin gap,
    but we do not show symbols like ↵ or Ġ.
    """
    s = clean_display_text(tok)
    if s == "":
        return ""
    # If a token is only newlines, render line breaks instead of glyphs.
    if s.strip(" \t\n\r") == "":
        if "\n" in s:
            return "<br>"
        return " "
    # Preserve internal newlines as HTML line breaks.
    return esc(s).replace("\n", "<br>")


def token_df(method: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "idx": t.get("idx"),
            "token": clean_display_text(t.get("token_text")),
            "loss": t.get("token_loss"),
            "gap": t.get("logprob_gap"),
            "candidate": t.get("is_candidate"),
            "selected": t.get("is_selected"),
            "notes": t.get("notes"),
        }
        for t in method.get("tokens", [])
    ])


BASE_CSS = """
<style>
:root { --fg:#2f3340; --muted:#6b7280; --line:#e6e8ef; --bg:#ffffff; --soft:#f7f8fb; --red:#e74c3c; --blue:#2563eb; --green:#148a4a; --amber:#8a6200; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color: var(--fg); background: var(--bg); margin: 26px; line-height: 1.45; }
h1 { font-size: 30px; margin: 0 0 18px; }
h2 { font-size: 24px; margin: 34px 0 12px; border-top: 1px solid var(--line); padding-top: 24px; }
h3 { font-size: 18px; margin: 22px 0 10px; }
h4 { font-size: 15px; margin: 16px 0 8px; }
.caption { color: var(--muted); font-size: 13px; margin: 4px 0 14px; }
.code { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; background: var(--soft); border: 1px solid #eef0f6; border-radius: 10px; padding: 14px 16px; overflow-x: auto; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.small { font-size: 13px; color: var(--muted); }
.metric { font-size: 30px; font-weight: 700; margin-top: 4px; }
.badge { display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; font-weight:700; margin-right:8px; }
.badge.teacher { background:#e8f0ff; color:#174ea6; border:1px solid #aac3ff; }
.badge.student { background:#fff3e8; color:#9a4b00; border:1px solid #f2c28f; }
.badge.score { background:#eef8f1; color:#137044; border:1px solid #a7e0bf; }
.token-box { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; line-height:2.05; word-break: break-word; white-space: normal; }
.tok { display:inline; padding:2px 4px; margin:1px; border-radius:4px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space:pre-wrap; }
.tok.candidate { outline: 1.5px dashed #555; }
.tok.selected { outline: 2.5px solid #174ea6; box-shadow: 0 0 0 2px rgba(23,78,166,0.12); }
table { border-collapse: collapse; width:100%; font-size: 13px; margin: 10px 0 18px; }
th, td { border: 1px solid var(--line); padding: 7px 8px; vertical-align: top; }
th { background:#f6f7fb; text-align:left; font-weight:650; }
.step { border-left: 4px solid var(--blue); padding-left: 12px; margin: 18px 0; }
.step-title { font-weight: 750; font-size: 17px; margin-bottom: 8px; }
.sample { margin-bottom: 14px; }
.sample .margin { font-weight: 700; color:#111827; }
.explain { background:#fbfcff; border:1px solid #e4e9ff; border-radius:12px; padding:12px 14px; margin:12px 0; }
.note { background:#fffdf4; border:1px solid #f0df9c; border-radius:12px; padding:12px 14px; margin:12px 0; color:#4f3b00; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } body { margin: 16px; } }
</style>
"""


def loss_alpha(loss: Optional[float], min_loss: float, max_loss: float) -> float:
    if loss is None:
        return 0.10
    denom = max(max_loss - min_loss, 1e-8)
    return 0.10 + 0.78 * ((float(loss) - min_loss) / denom)


def highlighted_tokens_html(method: Dict[str, Any], title: str, caption: str = "") -> str:
    tokens = method.get("tokens", [])
    losses = [t.get("token_loss") for t in tokens if t.get("token_loss") is not None]
    min_loss, max_loss = (min(losses), max(losses)) if losses else (0.0, 1.0)
    parts = []
    for t in tokens:
        rendered = token_text_for_html(t.get("token_text"))
        if rendered == "":
            continue
        # line-break-only tokens should not carry background blocks.
        if rendered == "<br>":
            parts.append("<br>")
            continue
        loss = t.get("token_loss")
        a = loss_alpha(loss, min_loss, max_loss)
        classes = ["tok"]
        labels = []
        if t.get("is_candidate"):
            classes.append("candidate")
            labels.append("candidate")
        if t.get("is_selected"):
            classes.append("selected")
            labels.append("selected")
        tip = f"idx={t.get('idx')} | loss={fmt(loss)} | gap={fmt(t.get('logprob_gap'))}"
        if labels:
            tip += " | " + ", ".join(labels)
        parts.append(
            f'<span class="{" ".join(classes)}" title="{esc(tip)}" '
            f'style="background:rgba(231,76,60,{a:.3f});">{rendered}</span>'
        )
    return f"""
    <h3>{esc(title)}</h3>
    {f'<div class="caption">{esc(caption)}</div>' if caption else ''}
    <div class="token-box">{''.join(parts)}</div>
    <div class="caption">Darker red = higher student token loss. Dashed border = top-C candidate. Blue border = selected point. Whitespace markers are cleaned for readability.</div>
    """


def table_html(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "<div class='caption'>No rows.</div>"
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for r in rows:
        body.append("<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def original_opd_html(method: Dict[str, Any]) -> str:
    repair = method.get("teacher_repair")
    selected = method.get("selected_index")
    cand = method.get("candidate_indices") or []
    html_parts = [
        "<h2>1. Original OPD-style signal</h2>",
        highlighted_tokens_html(
            method,
            "Token loss visualization",
            "This view highlights where the student assigned low probability to its own generated token. Original OPD-style selection uses this token-level signal on the student trajectory."
        ),
        f"<div class='card'><b>Top candidate indices:</b> {esc(cand)} &nbsp; <b>Selected index:</b> {esc(selected)}</div>",
    ]
    if not repair:
        html_parts.append("<div class='explain'>Teacher repair information is unavailable. The token-loss visualization is still usable.</div>")
        return "\n".join(html_parts)

    top_rows = []
    for r in repair.get("teacher_top_tokens") or []:
        top_rows.append({
            "rank": r.get("rank"),
            "token_text": clean_display_text(r.get("token_text")),
            "teacher_prob": fmt(r.get("teacher_prob"), 4),
            "teacher_logprob": fmt(r.get("teacher_logprob"), 4),
            "token_id": r.get("token_id"),
        })
    chosen = repair.get("student_chosen") or {}
    html_parts.append("""
    <h3>How original OPD would repair the selected high-loss token</h3>
    <div class="explain">
      <b>Interpretation:</b> this panel shows a <b>single-token training signal</b>, not an edited final answer. At the selected prefix, original OPD/KD would increase the probability of teacher-preferred next tokens and reduce the mismatch between the student distribution and teacher distribution.
    </div>
    """)
    html_parts.append(f"""
    <div class="grid2">
      <div>
        <h4>Prefix before the selected token</h4>
        <div class="code">{esc(clean_display_text(repair.get('prefix_text', '')))}</div>
      </div>
      <div>
        <h4><span class="badge student">Student token</span> token being repaired</h4>
        <div class="card">
          <b>idx:</b> {esc(repair.get('selected_idx'))}<br>
          <b>token:</b> <code>{esc(clean_display_text(repair.get('selected_token_text')))}</code><br>
          <b>student logprob:</b> {fmt(chosen.get('student_logprob'))}<br>
          <b>teacher logprob on this same token:</b> {fmt(chosen.get('teacher_logprob_on_student_token'))}<br>
          <b>token loss:</b> {fmt(chosen.get('token_loss'))}<br>
          <b>teacher-student gap:</b> {fmt(chosen.get('logprob_gap'))}
        </div>
      </div>
    </div>
    <h4><span class="badge teacher">Teacher distribution</span> preferred next-token alternatives at this prefix</h4>
    {table_html(top_rows, ['rank','token_text','teacher_prob','teacher_logprob','token_id'])}
    """)
    return "\n".join(html_parts)


def local_opd_html(method: Dict[str, Any]) -> str:
    steps = method.get("candidate_steps") or []
    selected = method.get("divergence_index")
    threshold = method.get("dynamic_threshold")
    html_parts = [
        "<h2>2. Local OPD divergence detection</h2>",
        "<div class='explain'><b>Goal:</b> Local OPD starts from high-loss candidates, but it does not assume every high-loss token is a true reasoning divergence. It probes each candidate locally, compares teacher/student continuation behavior, and then uses a trajectory-level distance.</div>",
    ]

    rows = []
    for s in steps:
        rows.append({
            "candidate_idx": s.get("candidate_idx"),
            "token": clean_display_text(s.get("token_text")),
            "token_loss": fmt(s.get("token_loss")),
            "rollback_start_idx": s.get("rollback_start_idx"),
            "probe_gap": fmt(s.get("probe_gap")),
            "threshold": fmt(s.get("threshold")),
            "decision": "divergent" if s.get("is_divergent") else "not divergent",
        })
    html_parts.append("<div class='step'><div class='step-title'>Step 1. Select top-C high-loss candidate tokens</div>")
    html_parts.append("<div class='caption'>These are suspicious points passed from the token-loss signal into Local OPD.</div>")
    html_parts.append(table_html(rows, ["candidate_idx", "token", "token_loss", "rollback_start_idx", "probe_gap", "threshold", "decision"]))
    html_parts.append("</div>")

    selected_step = next((s for s in steps if s.get("candidate_idx") == selected), None)
    html_parts.append("<div class='step'><div class='step-title'>Step 2. Roll back, generate teacher probe, and test divergence</div>")
    html_parts.append(f"<div class='caption'>Dynamic threshold = {fmt(threshold)}. Selected divergence index = <b>{esc(selected)}</b>. A candidate is confirmed when its divergence score is above the threshold.</div>")
    html_parts.append("""
    <div class="explain">
      For each high-loss candidate, Local OPD rolls back to a shared prefix. From that same prefix,
      it compares two short futures: <b>the student continuation</b> from the original rollout and
      <b>a teacher-generated continuation</b>. Each continuation receives a margin score
      (mean teacher logprob − mean student logprob). The divergence score is:
      <b>teacher-probe margin − student-probe margin</b>.
    </div>
    """)
    steps = method.get("candidate_steps") or []
    if steps:
        for step in steps:
            is_sel = step.get("candidate_idx") == selected
            border = " style='border-left:4px solid #2563eb; padding-left:10px;'" if is_sel else ""
            html_parts.append(f"<div class='card'{border}>")
            html_parts.append(
                f"<h4>Candidate idx {esc(step.get('candidate_idx'))}: "
                f"<code>{esc(clean_display_text(step.get('token_text') or ''))}</code> "
                f"{'<span class=\"badge teacher\">selected divergent point</span>' if is_sel else ''}</h4>"
            )
            html_parts.append(f"<div class='small'>token loss = {fmt(step.get('token_loss'))}; rollback start idx = {esc(step.get('rollback_start_idx'))}</div>")
            html_parts.append("<div class='grid2'>")
            html_parts.append(f"""
              <div>
                <h4>Rollback prefix</h4>
                <div class="caption"><span class="badge student">Student trajectory</span> Shared prefix before both continuations.</div>
                <div class="code">{esc(clean_display_text(step.get('rollback_prefix_text') or ''))}</div>
              </div>
              <div>
                <h4>Probe continuations from the same prefix</h4>
                <div class="caption"><span class="badge student">Student continuation</span> Short future from the original student rollout.</div>
                <div class="code">{esc(clean_display_text(step.get('student_probe_continuation_text') or step.get('probe_continuation_text') or ''))}</div>
                <div class="caption"><span class="badge teacher">Teacher-generated continuation</span> One short continuation generated by the teacher from the same rollback prefix.</div>
                <div class="code">{esc(clean_display_text(step.get('teacher_probe_continuation_text') or ''))}</div>
              </div>
            """)
            html_parts.append("</div>")
            html_parts.append(f"""
              <div class="card">
                <span class="badge student">student probe margin</span> {fmt(step.get('student_probe_margin_score'))}
                &nbsp;&nbsp;
                <span class="badge teacher">teacher probe margin</span> {fmt(step.get('teacher_probe_margin_score'))}
                &nbsp;&nbsp;
                <span class="badge score">divergence score</span> {fmt(step.get('divergence_score') if step.get('divergence_score') is not None else step.get('probe_gap'))}
                &nbsp;&nbsp;
                <b>threshold:</b> {fmt(step.get('threshold'))}
                <br><span class="small">Decision: {'confirmed divergent' if step.get('is_divergent') else 'not above threshold'}</span>
              </div>
            """)
            html_parts.append("</div>")
    elif selected_step:
        html_parts.append(f"""
        <div class="grid2">
          <div>
            <h4>Rollback prefix</h4>
            <div class="caption"><span class="badge student">Student trajectory</span> Prefix is taken from the original student rollout.</div>
            <div class="code">{esc(clean_display_text(selected_step.get('rollback_prefix_text') or ''))}</div>
          </div>
          <div>
            <h4>Student probe continuation</h4>
            <div class="caption"><span class="badge student">Student continuation</span> This older trace only stored the student probe.</div>
            <div class="code">{esc(clean_display_text(selected_step.get('probe_continuation_text') or ''))}</div>
            <div class="card">
              <b>probe gap:</b> {fmt(selected_step.get('probe_gap'))} &nbsp; <b>threshold:</b> {fmt(selected_step.get('threshold'))}
            </div>
          </div>
        </div>
        """)
    html_parts.append("</div>")

    teacher_samples = method.get("teacher_samples_sorted") or []
    student_samples = method.get("student_samples_sorted") or []
    html_parts.append("<div class='step'><div class='step-title'>Step 3. Generate local continuations from the same rollback prefix</div>")
    html_parts.append("<div class='caption'>After confirming a divergent point, both models generate multiple continuations from the selected rollback prefix. These samples estimate the local future trajectory distributions.</div>")
    html_parts.append("<div class='grid2'><div><h4><span class='badge teacher'>Teacher-generated continuations</span></h4>")
    for s in teacher_samples:
        html_parts.append(f"<div class='sample'><div class='margin'>margin = {fmt(s.get('margin_score'))}</div><div class='code'>{esc(clean_display_text(s.get('text') or ''))}</div></div>")
    html_parts.append("</div><div><h4><span class='badge student'>Student-generated continuations</span></h4>")
    for s in student_samples:
        html_parts.append(f"<div class='sample'><div class='margin'>margin = {fmt(s.get('margin_score'))}</div><div class='code'>{esc(clean_display_text(s.get('text') or ''))}</div></div>")
    html_parts.append("</div></div></div>")

    t_scores = [float(s.get("margin_score")) for s in teacher_samples if s.get("margin_score") is not None]
    s_scores = [float(s.get("margin_score")) for s in student_samples if s.get("margin_score") is not None]
    html_parts.append("<div class='step'><div class='step-title'>Step 4. Sort margin scores and compute final local distance</div>")
    html_parts.append("""
    <div class="explain">
      <b>Margin score:</b> mean teacher logprob − mean student logprob on one sampled continuation. Higher margin means the continuation is more teacher-preferred relative to the student.<br>
      <b>Final local distance:</b> after sorting the teacher-generated and student-generated margin scores, Local OPD computes a one-dimensional Wasserstein/OT distance between the two score distributions. A larger distance means the two local continuation distributions are more separated.
    </div>
    """)
    html_parts.append(f"""
    <div class="grid2">
      <div class="card"><span class="badge teacher">Teacher samples</span><br><b>Sorted margins:</b><br>{esc([round(x, 4) for x in sorted(t_scores, reverse=True)])}</div>
      <div class="card"><span class="badge student">Student samples</span><br><b>Sorted margins:</b><br>{esc([round(x, 4) for x in sorted(s_scores, reverse=True)])}</div>
    </div>
    <div class="card"><div class="small">{esc(method.get('local_distance_type') or '1D OT distance over margin scores')}</div><div class="metric">{fmt(method.get('local_distance_value'))}</div></div>
    """)
    html_parts.append("</div>")
    return "\n".join(html_parts)


def build_full_html(trace: Dict[str, Any]) -> str:
    baseline = trace.get("baseline", {})
    local = trace.get("local_opd", {})
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Original OPD vs Local OPD Trace</title>
{BASE_CSS}
</head>
<body>
<h1>Original OPD vs Local OPD: compact trace</h1>
<h2>Prompt</h2>
<div class="code">{esc(clean_display_text(trace.get('prompt', '')))}</div>
<h2>Original student answer</h2>
<div class="code">{esc(clean_display_text(trace.get('original_response_text') or baseline.get('response_text', '')))}</div>
{original_opd_html(baseline)}
{local_opd_html(local)}
<h2>Meta</h2>
<div class="code">{esc(json.dumps(trace.get('meta', {}), indent=2, ensure_ascii=False))}</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--html_out", type=str, default=None, help="Optional path to save a static standalone HTML page.")
    args, _ = parser.parse_known_args()

    trace = load_trace(args.trace)
    full_html = build_full_html(trace)

    if args.html_out:
        Path(args.html_out).write_text(full_html, encoding="utf-8")

    st.set_page_config(layout="wide", page_title="OPD Trace Viewer")
    st.title("Original OPD vs Local OPD: compact trace viewer")
    if args.html_out:
        st.success(f"Saved static HTML to {args.html_out}")
    st.download_button(
        "Download static HTML",
        data=full_html.encode("utf-8"),
        file_name="opd_local_trace.html",
        mime="text/html",
    )
    components.html(full_html, height=4600, scrolling=True)

    with st.expander("Raw token tables"):
        st.markdown("**Original OPD token table**")
        st.dataframe(token_df(trace.get("baseline", {})), use_container_width=True)
        st.markdown("**Local OPD token table**")
        st.dataframe(token_df(trace.get("local_opd", {})), use_container_width=True)


if __name__ == "__main__":
    main()
