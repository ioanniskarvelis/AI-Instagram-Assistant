"""Serves the trace viewer page.

Deliberately not on the `/admin` router: that router requires a Bearer
token on every request, which works for the JSON API (called by the
website's Netlify proxy) but not for a page loaded by direct browser
navigation, since a GET navigation can't carry a custom Authorization
header. This page ships with no data baked in — it's a static shell that
asks for the admin key once, stores it in localStorage, and then calls the
/admin/traces JSON API itself with that key attached.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Assistant trace viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --border: #ddd;
    --row-hover: #f2f2f2; --accent: #2563eb; --mono: ui-monospace, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #16181d; --fg: #e6e6e6; --muted: #9a9a9a; --border: #333; --row-hover: #1f2229; --accent: #6ea8fe; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; font-size: 14px; }
  header { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0 16px 0 0; }
  input, select, button { font: inherit; padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--fg); }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); }
  main { display: flex; height: calc(100vh - 53px); }
  #list { width: 46%; overflow: auto; border-right: 1px solid var(--border); }
  #detail { flex: 1; overflow: auto; padding: 16px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { position: sticky; top: 0; background: var(--bg); font-size: 12px; color: var(--muted); text-transform: uppercase; }
  tr.row { cursor: pointer; }
  tr.row:hover { background: var(--row-hover); }
  tr.row.selected { background: var(--row-hover); outline: 1px solid var(--accent); }
  .muted { color: var(--muted); }
  .mono { font-family: var(--mono); white-space: pre-wrap; word-break: break-word; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border); font-size: 12px; }
  .pill.suppressed { border-color: #b45309; color: #b45309; }
  .pill.canned { border-color: #b91c1c; color: #b91c1c; }
  .pill.generated { border-color: #15803d; color: #15803d; }
  section { margin-bottom: 20px; }
  section h2 { font-size: 12px; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }
  .turn { padding: 6px 10px; border-radius: 6px; margin-bottom: 4px; background: var(--row-hover); }
  .turn .role { font-size: 11px; color: var(--muted); }
  .hit { border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; }
  .hit .score { float: right; font-family: var(--mono); color: var(--accent); }
  pre { white-space: pre-wrap; word-break: break-word; background: var(--row-hover); padding: 10px; border-radius: 6px; }
  #empty { padding: 40px; color: var(--muted); text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Assistant trace viewer</h1>
  <input id="key" type="password" placeholder="Admin key" style="width:220px">
  <input id="sender" type="text" placeholder="Filter by sender_id">
  <select id="intent">
    <option value="">All intents</option>
    <option value="price">price</option>
    <option value="booking">booking</option>
    <option value="design">design</option>
    <option value="aftercare">aftercare</option>
    <option value="complaint">complaint</option>
    <option value="general">general</option>
  </select>
  <button id="refresh">Refresh</button>
  <span id="status" class="muted"></span>
</header>
<main>
  <div id="list">
    <table>
      <thead><tr><th>Time</th><th>Sender</th><th>Intent</th><th>Result</th><th>ms</th><th>Message</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div id="detail"><div id="empty">Select a trace to see details</div></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const state = { traces: [], selected: null };

$("key").value = localStorage.getItem("assistant_admin_key") || "";
$("key").addEventListener("change", () => localStorage.setItem("assistant_admin_key", $("key").value));

function authHeaders() {
  return { "Authorization": "Bearer " + $("key").value };
}

async function loadList() {
  $("status").textContent = "Loading…";
  const params = new URLSearchParams({ limit: "100" });
  if ($("sender").value.trim()) params.set("sender_id", $("sender").value.trim());
  if ($("intent").value) params.set("intent", $("intent").value);
  try {
    const resp = await fetch("/admin/traces?" + params, { headers: authHeaders() });
    if (!resp.ok) { $("status").textContent = "Error: " + resp.status; return; }
    state.traces = await resp.json();
    renderList();
    $("status").textContent = state.traces.length + " traces";
  } catch (e) {
    $("status").textContent = "Network error";
  }
}

function escapeHtml(s) {
  return (s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function preview(s, n) {
  s = s ?? "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function renderList() {
  const rows = state.traces.map(t => `
    <tr class="row${state.selected === t.id ? ' selected' : ''}" data-id="${t.id}">
      <td class="mono muted">${(t.created_at || "").replace("T", " ").slice(0, 19)}</td>
      <td class="mono">${escapeHtml(t.sender_id)}</td>
      <td>${escapeHtml(t.intent)}</td>
      <td><span class="pill ${t.reply_source}">${t.reply_source}</span></td>
      <td class="mono">${Math.round(t.total_latency_ms)}</td>
      <td>${escapeHtml(preview(t.incoming_text, 60))}</td>
    </tr>`).join("");
  $("rows").innerHTML = rows;
  document.querySelectorAll("tr.row").forEach(tr => {
    tr.addEventListener("click", () => selectTrace(parseInt(tr.dataset.id, 10)));
  });
}

async function selectTrace(id) {
  state.selected = id;
  renderList();
  const detail = $("detail");
  detail.innerHTML = '<div id="empty">Loading…</div>';
  const resp = await fetch(`/admin/traces/${id}`, { headers: authHeaders() });
  if (!resp.ok) { detail.innerHTML = `<div id="empty">Error ${resp.status}</div>`; return; }
  const t = await resp.json();

  const history = t.history_window.map(turn => `
    <div class="turn"><div class="role">${escapeHtml(turn.role)}</div>${escapeHtml(turn.text)}</div>
  `).join("") || '<span class="muted">empty</span>';

  const hits = t.retrieval_hits.length
    ? t.retrieval_hits.map(h => `
        <div class="hit">
          <span class="score">${h.score.toFixed(4)}</span>
          <div><strong>Q:</strong> ${escapeHtml(h.question)}</div>
          <div><strong>A:</strong> ${escapeHtml(h.reply)}</div>
        </div>`).join("")
    : '<span class="muted">no examples retrieved</span>';

  detail.innerHTML = `
    <section>
      <h2>Summary</h2>
      <div>sender: <span class="mono">${escapeHtml(t.sender_id)}</span></div>
      <div>intent: <strong>${escapeHtml(t.intent)}</strong> (${t.intent_latency_ms != null ? Math.round(t.intent_latency_ms) + " ms" : "n/a"})</div>
      <div>result: <span class="pill ${t.reply_source}">${t.reply_source}</span></div>
      <div>timing: retrieval ${t.retrieval_latency_ms != null ? Math.round(t.retrieval_latency_ms) + " ms" : "n/a"}, llm ${t.llm_latency_ms != null ? Math.round(t.llm_latency_ms) + " ms" : "n/a"}, total ${Math.round(t.total_latency_ms)} ms</div>
    </section>
    <section>
      <h2>Incoming message</h2>
      <pre>${escapeHtml(t.incoming_text)}</pre>
    </section>
    <section>
      <h2>History window sent to the model (${t.history_window.length})</h2>
      ${history}
    </section>
    <section>
      <h2>Retrieved style examples (top-${t.retrieval_hits.length})</h2>
      ${hits}
    </section>
    <section>
      <h2>System prompt actually sent</h2>
      ${t.system_prompt ? `<pre>${escapeHtml(t.system_prompt)}</pre>` : '<span class="muted">not generated (suppressed)</span>'}
    </section>
    <section>
      <h2>Reply</h2>
      <pre>${t.reply ? escapeHtml(t.reply) : '<span class="muted">none</span>'}</pre>
    </section>
  `;
}

$("refresh").addEventListener("click", loadList);
$("sender").addEventListener("keydown", (e) => { if (e.key === "Enter") loadList(); });
if ($("key").value) loadList();
</script>
</body>
</html>
"""


@router.get("/admin-ui/traces", response_class=HTMLResponse)
async def traces_page() -> str:
    return _PAGE
