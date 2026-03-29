"""
app/ui/gradio_app.py
====================
Gradio web interface for the Multi-Agent AI Research System.

Provides a four-tab browser UI mounted onto the FastAPI application at /ui.
Runs inside the same process as FastAPI — no separate server needed.

Tabs:
    1. New Research    — submit a research query and get a job_id
    2. Job Status      — monitor progress, view plan, approve/refine/reject
    3. Job History     — browse all past jobs from SQLite (persisted across restarts)
    4. Download Report — download the .docx report for any completed job

Mounting:
    Call build_gradio_app() to get the gr.Blocks instance, then mount it:
        import gradio as gr
        demo = build_gradio_app()
        app = gr.mount_gradio_app(app, demo, path="/ui")

Dependencies:
    gradio>=4.0.0
    requests (for calling POST /research and POST /approve endpoints)
"""

from __future__ import annotations

import os
import requests
import gradio as gr

from app.config import API_BASE_URL, API_REQUEST_TIMEOUT, UI_POLL_INTERVAL_SECONDS
from app.utils.job_store import job_store


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_status(job: dict) -> str:
    """
    Render a job dict as rich Markdown with colour-coded status emoji.

    Args:
        job: A job record dict returned by job_store.get_job().

    Returns:
        str: Multi-line Markdown string suitable for a gr.Markdown component.
    """
    # Map each status value to a descriptive emoji
    status_emoji = {
        "queued": "🕐",
        "planning": "🔄",
        "running": "⏳",
        "awaiting_approval": "👤",
        "complete": "✅",
        "failed": "⚠️",
        "cancelled": "❌",
    }.get(job["status"], "❓")

    lines = [
        f"## {status_emoji} Status: `{job['status']}`",
        f"**Query:** {job['query']}",
        f"**Depth:** {job['depth']}",
        f"**Phase:** {job.get('phase', '-')}",
    ]

    # Show live sub-agent activity during running/planning phases
    if job.get("progress") or job.get("sub_agents_active"):
        lines.append(f"**Sub-agents active:** {job.get('sub_agents_active', 0)}")

    # Show rich completion metrics when the job finishes successfully
    if job["status"] == "complete":
        lines.append(f"**Sources:** {job.get('source_count', 0)}")
        lines.append(f"**Citations:** {job.get('citation_count', 0)}")
        lines.append(f"**Duration:** {job.get('duration_seconds', 0):.1f}s")
        if job.get("summary_snippet"):
            lines.append(f"\n**Summary:** {job['summary_snippet']}")

    # Surface any error message so the user can act on it
    if job.get("error"):
        lines.append(f"\n⚠️ **Error:** {job['error']}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1 — New Research
# ---------------------------------------------------------------------------


def submit_research(query: str, depth: str) -> tuple[str, str]:
    """
    Submit a new research job via the POST /research REST endpoint.

    Args:
        query: The research question entered by the user.
        depth: One of "simple", "moderate", or "deep".

    Returns:
        tuple[str, str]: (job_id string, status markdown message).
            On error both values carry a descriptive error string.
    """
    # Validate inputs before hitting the API
    if not query or not query.strip():
        return "", "⚠️ Please enter a research query before submitting."

    try:
        resp = requests.post(
            f"{API_BASE_URL}/research",
            json={"query": query.strip(), "depth": depth},
            timeout=API_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("job_id", "")
        return (
            job_id,
            "✅ Job submitted! Copy your Job ID and go to the **Job Status** tab.",
        )
    except requests.exceptions.ConnectionError:
        return "", "❌ Cannot reach the API server. Is it running on port 8000?"
    except requests.exceptions.Timeout:
        return "", "❌ Request timed out. The server may be overloaded."
    except requests.exceptions.HTTPError as exc:
        return "", f"❌ API error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:  # noqa: BLE001 — UI must never crash
        return "", f"❌ Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Tab 2 — Job Status helpers
# ---------------------------------------------------------------------------


def check_status(job_id: str) -> tuple[str, dict | None]:
    """
    Look up a job by ID and return formatted status markdown plus the research plan.

    The research plan JSON is only surfaced when the job is awaiting_approval
    so the user can review it before deciding.

    Args:
        job_id: UUID string of the job to look up.

    Returns:
        tuple[str, dict | None]:
            - Markdown status string (always populated with a meaningful message).
            - research_plan dict if status is awaiting_approval, else None.
    """
    if not job_id or not job_id.strip():
        return "Enter a Job ID above to check its status.", None

    job = job_store.get_job(job_id.strip())
    if job is None:
        return f"❌ No job found with ID: `{job_id.strip()}`", None

    # Expose the research plan only during the HITL approval step
    plan = job.get("research_plan") if job["status"] == "awaiting_approval" else None
    return _format_status(job), plan


def handle_approval(job_id: str, decision: str, feedback: str) -> str:
    """
    Send an approval decision to the POST /approve/{job_id} endpoint.

    Args:
        job_id:   UUID of the job being reviewed.
        decision: One of "approved", "refine", or "rejected".
        feedback: Optional free-text guidance for the refinement case.

    Returns:
        str: Markdown string describing the result of the action.
    """
    if not job_id or not job_id.strip():
        return "⚠️ Please enter a Job ID before approving."

    payload: dict = {"decision": decision}
    if feedback and feedback.strip():
        payload["feedback"] = feedback.strip()

    try:
        resp = requests.post(
            f"{API_BASE_URL}/approve/{job_id.strip()}",
            json=payload,
            timeout=API_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        # Surface the API's own message if it provides one
        data = resp.json()
        api_msg = data.get("message", "")
        return f"✅ Decision **{decision}** recorded. {api_msg}".strip()
    except requests.exceptions.ConnectionError:
        return "❌ Cannot reach the API server."
    except requests.exceptions.Timeout:
        return "❌ Request timed out."
    except requests.exceptions.HTTPError as exc:
        return f"❌ API error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error: {exc}"


# Thin wrappers so each button can pass its own decision constant
def approve(job_id: str, feedback: str) -> str:
    """Wrapper: send 'approved' decision for the given job."""
    return handle_approval(job_id, "approved", feedback)


def refine(job_id: str, feedback: str) -> str:
    """Wrapper: send 'refine' decision with feedback for the given job."""
    return handle_approval(job_id, "refine", feedback)


def reject(job_id: str, feedback: str) -> str:
    """Wrapper: send 'rejected' decision for the given job."""
    return handle_approval(job_id, "rejected", feedback)


def load_activity(job_id: str) -> str:
    """
    Load all activity events for a job and format them as a readable log.

    Called by the gr.Timer to auto-refresh the activity feed.

    Args:
        job_id: UUID of the job to load events for.

    Returns:
        str: Newline-separated log lines with timestamps, or a placeholder message.
    """
    if not job_id or not job_id.strip():
        return "Enter a Job ID to see activity..."

    events = job_store.get_events(job_id.strip())
    if not events:
        return "No activity yet — job may still be starting up."

    lines = []
    for evt in events:
        # Show only time portion of the ISO timestamp (HH:MM:SS)
        ts = (evt.get("timestamp") or "")
        time_part = ts[11:19] if len(ts) >= 19 else ts
        lines.append(f"[{time_part}] {evt['message']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3 — Job History helpers
# ---------------------------------------------------------------------------



def load_history() -> list[list]:
    """
    Fetch all jobs from the store and format them as a list-of-lists for gr.Dataframe.

    Queries are truncated to 60 characters so the table stays readable.
    Timestamps are shortened to the date portion (first 10 chars of ISO string).

    Returns:
        list[list]: Each inner list is one table row:
            [job_id, query_truncated, depth, status, created_date, duration_seconds]
    """
    jobs = job_store.list_jobs()
    rows = []
    for job in jobs:
        # Truncate long queries with an ellipsis for compact display
        query_display = job["query"]
        if len(query_display) > 60:
            query_display = query_display[:57] + "..."

        # Show only the date portion of the ISO timestamp
        created_display = (job.get("created_at") or "")[:10]

        # Round duration to one decimal place; default to 0 if missing
        duration = round(job.get("duration_seconds") or 0, 1)

        rows.append([
            job["job_id"],
            query_display,
            job.get("depth", "-"),
            job.get("status", "-"),
            created_display,
            duration,
        ])
    return rows


# ---------------------------------------------------------------------------
# Tab 4 — Eval History helpers
# ---------------------------------------------------------------------------


def load_eval_runs() -> list[list]:
    """
    Fetch all eval runs from SQLite and format for gr.Dataframe.

    Returns:
        list[list]: Each row is [run_id_short, date, cases, passed, pass_rate, avg_score, total_cost, judge_model]
    """
    runs = job_store.list_eval_runs()
    rows = []
    for run in runs:
        run_id = run.get("run_id", "")
        date = (run.get("timestamp") or "")[:16].replace("T", " ")
        cases = run.get("total_cases", 0)
        passed = run.get("passed", 0)
        pass_rate = f"{run.get('pass_rate', 0)*100:.0f}%"
        avg_score = f"{run.get('avg_weighted_score', 0):.3f}"
        cost = f"${run.get('total_cost', 0):.4f}"
        judge = run.get("judge_model", "-")
        rows.append([run_id, date, cases, passed, pass_rate, avg_score, cost, judge])
    return rows


def load_eval_run_detail(run_id: str) -> tuple[list[list], str]:
    """
    Fetch per-case results for a specific eval run.

    Args:
        run_id: The eval run ID to load.

    Returns:
        tuple: (rows for dataframe, markdown summary text)
    """
    if not run_id or not run_id.strip():
        return [], "Enter a Run ID to see details."

    run_id = run_id.strip()
    results = job_store.get_eval_results(run_id)
    if not results:
        return [], f"No results found for run `{run_id}`."

    rows = []
    for r in results:
        verdict = r.get("verdict") or ("pipeline_err" if r.get("pipeline_error") else "—")
        score = f"{r['weighted_score']:.3f}" if r.get("weighted_score") is not None else "ERR"
        query_short = (r.get("query") or "")[:50] + ("…" if len(r.get("query", "")) > 50 else "")
        rows.append([
            r.get("case_id", ""),
            r.get("tier", ""),
            query_short,
            score,
            verdict,
            f"${r.get('cost', 0):.4f}",
            f"{r.get('duration_seconds', 0):.0f}s",
        ])

    # Build criteria summary markdown
    criteria = ["factual_accuracy", "citation_quality", "completeness",
                "source_quality", "structure_clarity", "efficiency"]
    scored = [r for r in results if r.get("weighted_score") is not None]
    md_lines = [f"### Run `{run_id[:20]}…` — {len(results)} cases\n"]
    if scored:
        passed = sum(1 for r in results if r.get("passed"))
        md_lines.append(f"**Pass rate:** {passed}/{len(results)} ({passed/len(results)*100:.0f}%)  ")
        md_lines.append(f"**Avg score:** {sum(r['weighted_score'] for r in scored)/len(scored):.3f}\n")
        md_lines.append("**Avg criteria scores:**")
        for c in criteria:
            vals = [r[c] for r in scored if r.get(c) is not None]
            avg = sum(vals) / len(vals) if vals else 0
            bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
            md_lines.append(f"- `{c:<22}` {bar} {avg:.2f}")
    else:
        md_lines.append("No scored results in this run.")

    return rows, "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Tab 5 — Download Report helpers
# ---------------------------------------------------------------------------


def download_report(job_id: str) -> tuple[str | None, str]:
    """
    Resolve the .docx report path for a completed job.

    Args:
        job_id: UUID of the job whose report should be downloaded.

    Returns:
        tuple[str | None, str]:
            - Absolute path to the .docx file (or None on error).
            - Markdown message describing success or the reason for failure.
    """
    if not job_id or not job_id.strip():
        return None, "⚠️ Please enter a Job ID."

    job = job_store.get_job(job_id.strip())
    if not job:
        return None, "❌ Job not found."

    if job["status"] != "complete":
        return None, f"⏳ Job is not complete yet (status: `{job['status']}`)"

    path = job.get("document_path")
    if not path or not os.path.exists(path):
        return None, "❌ Report file not found on disk."

    return path, "✅ Ready to download!"


# ---------------------------------------------------------------------------
# Tab 6 — Observability helpers
# ---------------------------------------------------------------------------


def load_obs_metrics() -> tuple[str, list[list], list[list]]:
    """
    Build the System Dashboard from SQLite (last 24h) + live /metrics endpoint.

    Returns:
        tuple:
            - header_md:   Header bar + legend Markdown.
            - requests_rows: list-of-lists for Requests & Performance table.
            - cost_rows:   list-of-lists for Cost & Token Health table.
    """
    from datetime import datetime as _dt
    from app.config import MAX_COST_PER_QUERY, COST_WARNING_THRESHOLD

    now_str = _dt.utcnow().strftime("%H:%M:%S UTC")

    # Live uptime from /metrics endpoint
    uptime_str = "—"
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics", timeout=3)
        if resp.ok:
            uptime_str = f"{resp.json().get('uptime_seconds', 0):.0f}s"
    except Exception:
        pass

    # SQLite-backed 24h stats (survives restarts)
    s = job_store.get_dashboard_stats(hours=24)

    # ── Header bar ────────────────────────────────────────────────────────────
    server_icon = "🟢 Server Online" if uptime_str != "—" else "🔴 Server Offline"
    header_md = (
        f"**📊 Last 24 Hours** &nbsp;|&nbsp; "
        f"**Refreshed:** {now_str} &nbsp;|&nbsp; "
        f"{server_icon} &nbsp;|&nbsp; "
        f"**Uptime:** {uptime_str}\n\n"
        f"**Legend:** &nbsp; ✅ Normal &nbsp; ⚠️ Warning (failures > 0 or cost near threshold) "
        f"&nbsp; 🔴 Critical (budget exceeded or multiple failures) &nbsp; — Not applicable"
    )

    # ── Requests & Performance table ─────────────────────────────────────────
    total = s["total"]
    completed = s["completed"]
    failed = s["failed"]
    success_rate = s["success_rate"]

    requests_rows = [
        ["Total Requests",  total,                                           "—"],
        ["Completed",       completed,                                       "✅" if failed == 0 else "⚠️"],
        ["Failed",          failed,                                          "✅" if failed == 0 else ("🔴" if failed > 2 else "⚠️")],
        ["Success Rate",    f"{success_rate:.0f}%",                         "✅" if success_rate >= 80 else ("⚠️" if success_rate >= 50 else "🔴")],
        ["Avg Duration",    f"{s['avg_duration']:.1f}s" if s["avg_duration"] else "—", "—"],
        ["Fastest Job",     f"{s['fastest']:.1f}s" if s["fastest"] else "—", "—"],
        ["Slowest Job",     f"{s['slowest']:.1f}s" if s["slowest"] else "—", "—"],
    ]

    # ── Cost & Token Health table ─────────────────────────────────────────────
    total_cost = s["total_cost"]
    avg_cost = s["avg_cost"]
    budget_exceeded = s["budget_exceeded_count"]

    cost_rows = [
        ["Total Cost (24h)",        f"${total_cost:.4f}",              "—"],
        ["Avg Cost / Query",        f"${avg_cost:.4f}",                "✅" if avg_cost < COST_WARNING_THRESHOLD else ("⚠️" if avg_cost < MAX_COST_PER_QUERY else "🔴")],
        ["Cost Budget (per query)", f"${MAX_COST_PER_QUERY:.2f}",      "—"],
        ["Warning Threshold",       f"${COST_WARNING_THRESHOLD:.2f}",  "—"],
        ["Total Tokens Used",       f"{s['total_tokens']:,}",          "—"],
        ["Avg Tokens / Query",      f"{s['avg_tokens']:,.0f}",         "—"],
        ["Budget Exceeded",         budget_exceeded,                   "✅" if budget_exceeded == 0 else "🔴"],
    ]

    return header_md, requests_rows, cost_rows


def load_obs_job_choices() -> list[tuple[str, str]]:
    """
    Return (label, job_id) pairs for the 10 most recently completed traced jobs.

    Used to populate the Observability job selector dropdown.
    """
    traces = job_store.list_recent_job_traces(limit=10)
    choices = []
    for t in traces:
        short_id = (t.get("job_id") or "")[:8]
        short_q = (t.get("query") or "")[:45]
        date = (t.get("saved_at") or "")[:10]
        label = f"{short_id}… [{date}] {short_q}"
        choices.append((label, t["job_id"]))
    return choices


def load_obs_job_detail(job_id: str) -> tuple[str, list[list], str, str]:
    """
    Load budget summary, combined pipeline health table, and activity log for a job.

    Args:
        job_id: UUID of the job to inspect.

    Returns:
        tuple:
            - budget_md:       Markdown budget summary bar.
            - health_rows:     list-of-lists for the combined health Dataframe (11 cols).
            - activity_header: Markdown header showing Job ID + query preview.
            - activity_text:   multi-line string for the activity Textbox.
    """
    if not job_id or not job_id.strip():
        return "", [], "", "Select a job from the dropdown or enter a Job ID."

    job_id = job_id.strip()
    trace = job_store.get_job_trace(job_id)

    if not trace:
        msg = f"No trace found for job `{job_id}`. Only completed jobs are stored."
        return "", [], "", msg

    import datetime as _dt
    from app.config import MAX_COST_PER_QUERY, COST_WARNING_THRESHOLD

    cs = trace.get("cost_summary") or {}
    spans = {s["name"]: s for s in (trace.get("timeline") or []) if s.get("name")}
    breakdown = cs.get("breakdown", {})
    call_stats = trace.get("call_stats") or {}
    total_cost = cs.get("total", 0.0)
    total_tokens = cs.get("total_input_tokens", 0) + cs.get("total_output_tokens", 0)

    # ── Budget summary Markdown ───────────────────────────────────────────────
    budget_pct = (total_cost / MAX_COST_PER_QUERY * 100) if MAX_COST_PER_QUERY else 0
    remaining = max(0.0, MAX_COST_PER_QUERY - total_cost)
    if total_cost >= MAX_COST_PER_QUERY:
        status_icon = "🔴 Over Budget"
    elif total_cost >= COST_WARNING_THRESHOLD:
        status_icon = "⚠️ Warning"
    else:
        status_icon = "✅ In Budget"

    budget_md = (
        f"**Budget:** ${MAX_COST_PER_QUERY:.2f} &nbsp;|&nbsp; "
        f"**Used:** ${total_cost:.6f} &nbsp;|&nbsp; "
        f"**Remaining:** ${remaining:.6f} &nbsp;|&nbsp; "
        f"**{status_icon}** &nbsp;|&nbsp; "
        f"**Warning:** ${COST_WARNING_THRESHOLD:.2f} &nbsp;|&nbsp; "
        f"**Used:** {budget_pct:.2f}%"
    )

    # ── Combined pipeline health table ────────────────────────────────────────
    # Canonical step order with (display_name, agent_key_in_breakdown)
    _STEP_ORDER = [
        ("plan_research",       "lead_researcher"),
        ("spawn_subagents",     "sub_agents"),
        ("evaluate_sufficiency", None),
        ("synthesize",          "lead_researcher"),
        ("review_synthesis",    None),
        ("cite",                "citation_agent"),
        ("generate_document",   None),
    ]

    # Total pipeline duration for Time % calculation
    pipeline_span = spans.get("pipeline_run")
    total_dur_ms = pipeline_span.get("duration_ms") or 1.0 if pipeline_span else 1.0

    health_rows: list[list] = []
    total_calls = total_retries = total_failed = 0

    for order, (step_name, _) in enumerate(_STEP_ORDER, start=1):
        span = spans.get(step_name)
        dur_ms = span.get("duration_ms") if span else None
        dur_str = f"{dur_ms:,.0f}ms" if dur_ms is not None else "—"
        time_pct = f"{dur_ms / total_dur_ms * 100:.1f}%" if dur_ms else "—"

        # Cost: sum breakdown entries that map to this step
        step_cost = 0.0
        step_tokens = 0
        if step_name == "spawn_subagents" and "sub_agents" in breakdown:
            step_cost = breakdown["sub_agents"].get("cost", 0.0)
            step_tokens = breakdown["sub_agents"].get("input_tokens", 0)
        elif step_name == "cite" and "citation_agent" in breakdown:
            step_cost = breakdown["citation_agent"].get("cost", 0.0)
            step_tokens = breakdown["citation_agent"].get("input_tokens", 0)
        elif step_name in ("plan_research", "evaluate_sufficiency", "synthesize", "review_synthesis"):
            lead = breakdown.get("lead_researcher", {})
            lead_total = lead.get("cost", 0.0)
            lead_tokens_total = lead.get("input_tokens", 0)
            step_cost = lead_total / 4
            step_tokens = lead_tokens_total // 4

        cost_pct = f"{step_cost / MAX_COST_PER_QUERY * 100:.2f}%" if MAX_COST_PER_QUERY and step_cost else "—"
        cost_str = f"${step_cost:.6f}" if step_cost else "—"
        tokens_str = f"{step_tokens:,}" if step_tokens else "—"

        # LLM call stats for this step
        step_stat = call_stats.get(step_name, {})
        calls = step_stat.get("calls", 0)
        retries = step_stat.get("retries", 0)
        failed = step_stat.get("failed", 0)
        total_calls += calls
        total_retries += retries
        total_failed += failed

        if failed > 0:
            health = "🔴"
        elif retries > 0:
            health = "⚠️"
        else:
            health = "✅"

        health_rows.append([
            step_name, order, dur_str, time_pct, tokens_str, cost_str, cost_pct,
            calls or "—", retries or "—", failed or "—", health,
        ])

    # TOTAL row
    health_rows.append([
        "TOTAL (pipeline_run)", "—",
        f"{total_dur_ms:,.0f}ms", "100%",
        f"{total_tokens:,}",
        f"${total_cost:.6f}",
        f"{budget_pct:.2f}%",
        total_calls or "—", total_retries or "—", total_failed or "—",
        status_icon,
    ])

    # ── Activity log ─────────────────────────────────────────────────────────
    events = job_store.get_events(job_id)
    short_id = job_id[:8]
    query_preview = (trace.get("query") or "")[:80]
    activity_header = f"**Job:** `{short_id}…` &nbsp;|&nbsp; **Query:** {query_preview}"

    if events:
        lines = [f"[{(e.get('timestamp') or '')[11:19]}] {e['message']}" for e in events]
        activity_text = "\n".join(lines[-60:])
    else:
        activity_text = "No activity events recorded for this job."

    return budget_md, health_rows, activity_header, activity_text


def refresh_obs_dropdown():
    """Return an updated Dropdown with the latest 10 traced jobs as choices."""
    choices = load_obs_job_choices()
    return gr.update(choices=choices, value=None)


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------


with gr.Blocks(title="AI Research System") as demo:
    # -----------------------------------------------------------------------
    # Page header
    # -----------------------------------------------------------------------
    gr.Markdown(
        """
        # 🔬 Multi-Agent AI Research System
        Powered by LangGraph · Groq · FastAPI
        """,
    )

    # -----------------------------------------------------------------------
    # Tab 1: New Research
    # -----------------------------------------------------------------------
    with gr.Tab("🔬 New Research"):
        gr.Markdown("### Submit a New Research Query")
        gr.Markdown(
            "Enter your research question below and choose the depth of analysis. "
            "A job will be queued immediately and you can track its progress on the "
            "**Job Status** tab."
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Main query input — multi-line to accommodate longer questions
                new_query_input = gr.Textbox(
                    label="Research Query",
                    placeholder="e.g. top AI agent frameworks in 2025",
                    lines=3,
                )
                # Depth selector with "simple" as a sensible default
                new_depth_input = gr.Dropdown(
                    choices=["simple", "moderate", "deep"],
                    value="simple",
                    label="Research Depth",
                )
                new_submit_btn = gr.Button("Start Research", variant="primary")

            with gr.Column(scale=2):
                # Read-only output fields populated after a successful submission
                new_job_id_output = gr.Textbox(
                    label="Job ID",
                    interactive=False,
                    placeholder="Your job ID will appear here after submission",
                )
                new_status_md = gr.Markdown(
                    value="",
                    label="Submission Status",
                )

        # Wire up the submit button to the API call
        new_submit_btn.click(
            fn=submit_research,
            inputs=[new_query_input, new_depth_input],
            outputs=[new_job_id_output, new_status_md],
        )

    # -----------------------------------------------------------------------
    # Tab 2: Job Status
    # -----------------------------------------------------------------------
    with gr.Tab("📊 Job Status"):
        gr.Markdown("### Monitor Research Progress")
        gr.Markdown(
            "Paste a Job ID to see live status. "
            "The panel auto-refreshes every **5 seconds** while a job ID is present."
        )

        with gr.Row():
            # Job ID entry — wide to accommodate UUIDs
            status_job_id_input = gr.Textbox(
                label="Job ID",
                placeholder="Paste your job ID here",
                scale=4,
            )
            status_check_btn = gr.Button("Check Status", variant="secondary", scale=1)

        # Rich Markdown display — shows status emoji, metrics, summary
        status_md_output = gr.Markdown(value="", label="Current Status")

        gr.Markdown("#### 📡 Live Activity Feed")
        activity_output = gr.Textbox(
            label="Agent Activity Log",
            lines=12,
            max_lines=20,
            interactive=False,
            placeholder="Activity will appear here once a job is running...",
        )

        # Research plan JSON — only rendered when job is awaiting_approval
        plan_json_output = gr.JSON(
            label="Research Plan (review before approving)",
            visible=True,
        )

        gr.Markdown("#### Human-in-the-Loop Decision")
        gr.Markdown(
            "When a job reaches **awaiting approval** status the research plan above "
            "will be populated. Review it, optionally add feedback, then click one of "
            "the action buttons."
        )

        # Optional free-text feedback used for the 'refine' decision
        hitl_feedback_input = gr.Textbox(
            label="Feedback for Refinement",
            placeholder="e.g. Focus only on open-source frameworks",
            lines=2,
        )

        with gr.Row():
            # Three HITL action buttons in a single row for visual clarity
            approve_btn = gr.Button("✅ Approve", variant="primary")
            refine_btn = gr.Button("🔄 Refine with Feedback", variant="secondary")
            reject_btn = gr.Button("❌ Reject", variant="stop")

        # Outcome message shown after an approval action
        hitl_result_md = gr.Markdown(value="")

        # ------------------------------------------------------------------
        # Auto-refresh every 5 seconds via gr.Timer
        # ------------------------------------------------------------------
        timer = gr.Timer(value=UI_POLL_INTERVAL_SECONDS)
        timer.tick(
            fn=check_status,
            inputs=[status_job_id_input],
            outputs=[status_md_output, plan_json_output],
        )
        timer.tick(
            fn=load_activity,
            inputs=[status_job_id_input],
            outputs=[activity_output],
        )

        # Manual check button uses the same handler
        status_check_btn.click(
            fn=check_status,
            inputs=[status_job_id_input],
            outputs=[status_md_output, plan_json_output],
        )
        status_check_btn.click(
            fn=load_activity,
            inputs=[status_job_id_input],
            outputs=[activity_output],
        )

        # HITL action buttons — each passes a fixed decision string and clears feedback on success
        approve_btn.click(
            fn=lambda job_id, feedback: (approve(job_id, feedback), ""),
            inputs=[status_job_id_input, hitl_feedback_input],
            outputs=[hitl_result_md, hitl_feedback_input],
        )
        refine_btn.click(
            fn=lambda job_id, feedback: (refine(job_id, feedback), ""),
            inputs=[status_job_id_input, hitl_feedback_input],
            outputs=[hitl_result_md, hitl_feedback_input],
        )
        reject_btn.click(
            fn=lambda job_id, feedback: (reject(job_id, feedback), ""),
            inputs=[status_job_id_input, hitl_feedback_input],
            outputs=[hitl_result_md, hitl_feedback_input],
        )

    # -----------------------------------------------------------------------
    # Tab 3: Job History
    # -----------------------------------------------------------------------
    with gr.Tab("📋 Job History"):
        gr.Markdown("### All Research Jobs")
        gr.Markdown(
            "Browse every job that has been submitted to this system. "
            "Click **Refresh** to pull the latest data from the database."
        )

        history_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")

        history_df = gr.Dataframe(
            headers=["Job ID", "Query", "Depth", "Status", "Created", "Duration(s)"],
            interactive=False,
            wrap=True,
        )

        # Populated automatically when user clicks any row in the table
        history_selected_id = gr.Textbox(
            label="📋 Selected Job ID — copy this and paste into Job Status or Download tabs",
            placeholder="Click any row above to copy its Job ID here",
            interactive=True,
        )

        # Populate table on refresh button click
        history_refresh_btn.click(
            fn=load_history,
            inputs=[],
            outputs=[history_df],
        )

        # Wire row-click: extract job_id (column 0) from the clicked row
        def _on_row_select(evt: gr.SelectData, df_data):
            """Copy the job_id of the clicked row into the Selected Job ID textbox."""
            if df_data is None:
                return ""
            row_idx = evt.index[0]
            try:
                # Gradio 6 passes a pandas DataFrame; earlier versions pass list of lists
                if hasattr(df_data, "iloc"):
                    return str(df_data.iloc[row_idx, 0])
                return str(df_data[row_idx][0])
            except Exception:
                return ""

        history_df.select(
            fn=_on_row_select,
            inputs=[history_df],
            outputs=[history_selected_id],
        )

        # Also load history immediately when the app first loads
        demo.load(
            fn=load_history,
            inputs=[],
            outputs=[history_df],
        )

    # -----------------------------------------------------------------------
    # Tab 4: Eval History
    # -----------------------------------------------------------------------
    with gr.Tab("🧪 Eval History"):
        gr.Markdown("### Evaluation Run History")
        gr.Markdown(
            "Browse all evaluation runs stored in SQLite. "
            "Click **Refresh** to load the latest runs, then click a row to see per-case scores."
        )

        eval_runs_refresh_btn = gr.Button("🔄 Refresh Runs", variant="secondary")

        eval_runs_df = gr.Dataframe(
            headers=["Run ID", "Date", "Cases", "Passed", "Pass Rate", "Avg Score", "Cost", "Judge Model"],
            interactive=False,
            wrap=True,
        )

        eval_selected_run_id = gr.Textbox(
            label="Selected Run ID",
            placeholder="Click a row above to select a run",
            interactive=True,
        )

        eval_load_detail_btn = gr.Button("Load Case Details", variant="secondary")

        with gr.Row():
            with gr.Column(scale=2):
                eval_results_df = gr.Dataframe(
                    headers=["Case ID", "Tier", "Query", "Score", "Verdict", "Cost", "Duration"],
                    interactive=False,
                    wrap=True,
                    label="Per-Case Results",
                )
            with gr.Column(scale=1):
                eval_criteria_md = gr.Markdown(value="Select a run to see criteria breakdown.")

        # Refresh button loads all runs
        eval_runs_refresh_btn.click(
            fn=load_eval_runs,
            inputs=[],
            outputs=[eval_runs_df],
        )

        # Row click copies run_id into the textbox
        def _on_eval_run_select(evt: gr.SelectData, df_data):
            if df_data is None:
                return ""
            row_idx = evt.index[0]
            try:
                if hasattr(df_data, "iloc"):
                    return str(df_data.iloc[row_idx, 0])
                return str(df_data[row_idx][0])
            except Exception:
                return ""

        eval_runs_df.select(
            fn=_on_eval_run_select,
            inputs=[eval_runs_df],
            outputs=[eval_selected_run_id],
        )

        # Load detail button or row click fetches per-case results
        eval_load_detail_btn.click(
            fn=load_eval_run_detail,
            inputs=[eval_selected_run_id],
            outputs=[eval_results_df, eval_criteria_md],
        )

        # Also auto-load detail when run_id textbox changes via row select
        eval_selected_run_id.change(
            fn=load_eval_run_detail,
            inputs=[eval_selected_run_id],
            outputs=[eval_results_df, eval_criteria_md],
        )

        # Load runs on app startup
        demo.load(
            fn=load_eval_runs,
            inputs=[],
            outputs=[eval_runs_df],
        )

    # -----------------------------------------------------------------------
    # Tab 6: Observability  (3 nested sub-tabs)
    # -----------------------------------------------------------------------
    with gr.Tab("🔭 Observability"):
        gr.Markdown("### Pipeline Observability — system health · cost · timing · activity logs")

        with gr.Tabs():

            # ── Sub-tab A: System Dashboard ──────────────────────────────
            with gr.Tab("📈 System Dashboard"):
                with gr.Row():
                    obs_metrics_md = gr.Markdown(value="Loading…")
                    obs_metrics_refresh_btn = gr.Button(
                        "🔄 Refresh", variant="secondary", scale=0, min_width=100
                    )
                with gr.Row():
                    obs_requests_df = gr.Dataframe(
                        headers=["Metric", "Value", "Status"],
                        label="Requests & Performance",
                        interactive=False,
                        wrap=True,
                        scale=1,
                    )
                    obs_cost_df = gr.Dataframe(
                        headers=["Metric", "Value", "Status"],
                        label="Cost & Token Health",
                        interactive=False,
                        wrap=True,
                        scale=1,
                    )

            # ── Sub-tab B: Job Inspector ──────────────────────────────────
            with gr.Tab("🔍 Job Inspector"):
                with gr.Row():
                    obs_job_dropdown = gr.Dropdown(
                        label="Recent Jobs",
                        choices=load_obs_job_choices(),
                        value=None,
                        scale=3,
                    )
                    obs_manual_job_id = gr.Textbox(
                        label="Or enter Job ID",
                        placeholder="Paste UUID here",
                        scale=3,
                    )
                    obs_load_btn = gr.Button("Load", variant="primary", scale=1)
                    obs_refresh_dropdown_btn = gr.Button("🔄", scale=0, min_width=60)

                obs_selected_job_id = gr.State(value="")

                obs_budget_md = gr.Markdown(value="")

                gr.Markdown("#### Pipeline Health (Time + Cost + LLM Calls per Step)")
                obs_health_df = gr.Dataframe(
                    headers=["Step", "Ord", "Duration", "Time %", "Tokens", "Cost $", "Cost %",
                             "LLM Calls", "Retries", "Failed", "Health"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown(
                    "_**Legend:** &nbsp; "
                    "**LLM Calls** = total Groq API calls per step &nbsp;|&nbsp; "
                    "**Tokens** = prompt + completion combined &nbsp;|&nbsp; "
                    "**Cost $** = USD, priced at input rate (actual ~10–20% higher) &nbsp;|&nbsp; "
                    "**Retries** = rate-limit retries (call recovered) &nbsp;|&nbsp; "
                    "**Failed** = permanent failures &nbsp;|&nbsp; "
                    "**Health:** ✅ clean &nbsp; ⚠️ retries &nbsp; 🔴 failures_"
                )

            # ── Sub-tab C: Activity Log ───────────────────────────────────
            with gr.Tab("📋 Activity Log"):
                obs_activity_header_md = gr.Markdown(value="")
                obs_activity_log = gr.Textbox(
                    label="Agent Activity Events",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    placeholder="Select a job in Job Inspector to see activity here.",
                )
                obs_reload_activity_btn = gr.Button("🔄 Reload Activity", variant="secondary")

        # ── Wire all cross-tab events here (after ALL components defined) ─────

        def _obs_load_all(job_id):
            budget_md, health_rows, activity_header, activity = load_obs_job_detail(job_id)
            return job_id, budget_md, health_rows, activity_header, activity

        _inspector_outputs = [obs_selected_job_id, obs_budget_md, obs_health_df,
                              obs_activity_header_md, obs_activity_log]

        obs_job_dropdown.change(
            fn=_obs_load_all,
            inputs=[obs_job_dropdown],
            outputs=_inspector_outputs,
        )
        obs_load_btn.click(
            fn=lambda mid: _obs_load_all(mid),
            inputs=[obs_manual_job_id],
            outputs=_inspector_outputs,
        )
        obs_refresh_dropdown_btn.click(
            fn=refresh_obs_dropdown,
            inputs=[],
            outputs=[obs_job_dropdown],
        )
        obs_reload_activity_btn.click(
            fn=lambda job_id: load_obs_job_detail(job_id)[2:],
            inputs=[obs_selected_job_id],
            outputs=[obs_activity_header_md, obs_activity_log],
        )
        obs_selected_job_id.change(
            fn=lambda job_id: load_obs_job_detail(job_id)[2:],
            inputs=[obs_selected_job_id],
            outputs=[obs_activity_header_md, obs_activity_log],
        )
        _dashboard_outputs = [obs_metrics_md, obs_requests_df, obs_cost_df]

        obs_metrics_refresh_btn.click(
            fn=load_obs_metrics,
            inputs=[],
            outputs=_dashboard_outputs,
        )

        # ── Auto-refresh system metrics every 5s ─────────────────────────────
        obs_timer = gr.Timer(value=UI_POLL_INTERVAL_SECONDS)
        obs_timer.tick(fn=load_obs_metrics, inputs=[], outputs=_dashboard_outputs)

        # Load on page startup
        demo.load(fn=load_obs_metrics, inputs=[], outputs=_dashboard_outputs)
        demo.load(fn=refresh_obs_dropdown, inputs=[], outputs=[obs_job_dropdown])

    # -----------------------------------------------------------------------
    # Tab 5: Download Report
    # -----------------------------------------------------------------------
    with gr.Tab("📥 Download Report"):
        gr.Markdown("### Download Completed Research Report")
        gr.Markdown(
            "Enter the Job ID of a **completed** research job to download the "
            "generated `.docx` report."
        )

        with gr.Row():
            download_job_id_input = gr.Textbox(
                label="Job ID",
                placeholder="Paste job ID of a completed research job",
                scale=4,
            )
            download_btn = gr.Button("Download Report", variant="primary", scale=1)

        # gr.File renders the downloaded file in the browser when path is set
        download_file_output = gr.File(label="Report (.docx)")

        # Feedback message — success or descriptive error
        download_status_md = gr.Markdown(value="")

        download_btn.click(
            fn=download_report,
            inputs=[download_job_id_input],
            outputs=[download_file_output, download_status_md],
        )


# ---------------------------------------------------------------------------
# Public factory function used by app/main.py
# ---------------------------------------------------------------------------


def build_gradio_app() -> gr.Blocks:
    """
    Return the configured Gradio Blocks app for mounting onto FastAPI.

    The returned instance is the module-level ``demo`` object built above.
    Mount it with:

        app = gr.mount_gradio_app(app, build_gradio_app(), path="/ui")

    Returns:
        gr.Blocks: The fully-configured Gradio application.
    """
    return demo
