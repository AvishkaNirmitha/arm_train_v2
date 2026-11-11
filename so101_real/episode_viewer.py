"""Browser viewer for recorded episodes (click_goal_real.py --record): open viewer.html, see the three camera frames of
any row of the table, play the episode, and follow the joints and your inputs over time.

    python so101_real/episode_viewer.py recordings/11_11_2026_pick_white_cube              # every episode + index
    python so101_real/episode_viewer.py recordings/11_11_2026_pick_white_cube/episode_000  # one episode
    python so101_real/episode_viewer.py recordings --open                                 # all, then open the index

Per episode it writes
    viewer.html        the page (data built in; opens from disk, no server needed)
    web/<camera>.mp4   H.264 copies of the videos (browsers can't play OpenCV's mp4v), made with ffmpeg
    web/overlay_<camera>.mp4   the same of overlay/<camera>.mp4 (the windows as shown), if recorded: the page's
                       "clean pictures / as shown" switch
and recordings/index.html lists every episode. New episodes get them automatically when they are saved.
"""
import os
import re
import sys
import json
import html
import shutil
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recorder import RECORD_ROOT, _fmt_event, write_table   # noqa: E402

CAMS = ("top", "wrist", "base")


def convert_videos(ep_dir):
    """web/<cam>.mp4: H.264, one frame per recorded frame (same numbering), a keyframe every 10 frames (fast seeking)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found - the viewer needs it to make browser-playable videos")
    out = []
    os.makedirs(os.path.join(ep_dir, "web"), exist_ok=True)
    for cam in CAMS:
        src = os.path.join(ep_dir, f"{cam}.mp4")
        if not os.path.exists(src):
            continue
        # web/<cam>.mp4 from <cam>.mp4, web/overlay_<cam>.mp4 from overlay/<cam>.mp4 (if recorded)
        for s, d in ((src, f"{cam}.mp4"), (os.path.join(ep_dir, "overlay", f"{cam}.mp4"), f"overlay_{cam}.mp4")):
            dst = os.path.join(ep_dir, "web", d)
            if os.path.exists(s) and not (os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(s)):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", s, "-vsync", "passthrough",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                "-g", "10", "-movflags", "+faststart", "-an", dst], check=True)
        out.append(cam)
    return out


def _group(e):
    """The filter group of an event (the table's dropdown): 'click top', 'gripper close (X)', ..."""
    k = e["kind"]
    if k == "click":
        return f"click {e.get('camera')}"
    if k == "plan":
        return f"plan {e.get('plan')}" + ("" if e.get("ok") else " (refused)")
    if k in ("gripper_open", "gripper_close"):
        return k.replace("_", " ")
    if k == "gripper":                                       # recordings from before 11-11-2026
        return "gripper close" if e.get("key") == "x" else "gripper open"
    if k == "move_done":
        return "move STOPPED" if e.get("stopped") else "move done"
    if k == "hold":
        return f"HOLD {e.get('motor')} {'on' if e.get('on') else 'off'}"
    return {"goal": "goal", "execute": "ENTER (execute plan)", "move_start": "move started", "surface": "surface (P)",
            "clear": "clear (C)", "episode_start": "episode start"}.get(k, k)


def _data(ep_dir, cams):
    rows = [json.loads(line) for line in open(os.path.join(ep_dir, "frames.jsonl")) if line.strip()]
    ev_path = os.path.join(ep_dir, "events.jsonl")
    events = [json.loads(line) for line in open(ev_path) if line.strip()] if os.path.exists(ev_path) else []
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    t0 = meta.get("t0_unix") or datetime.fromisoformat(meta["started"]).timestamp()
    motors = list(rows[0]["action"]) if rows else []
    ev_out, j = [], 0
    for e in sorted(events, key=lambda e: e["t"]):
        while j < len(rows) - 1 and rows[j]["t"] < e["t"]:
            j += 1
        ev_out.append({"t": e["t"], "f": j, "k": e["kind"], "x": _fmt_event(e), "g": _group(e)})
    frames = [{"t": r["t"], "c": datetime.fromtimestamp(t0 + r["t"]).strftime("%H:%M:%S.%f")[:-3],
               "s": [round(r["state"].get(m, float("nan")), 2) for m in motors],
               "a": [round(r["action"].get(m, float("nan")), 2) for m in motors], "m": int(bool(r.get("moving")))}
              for r in rows]
    fps = meta.get("fps_nominal") or 10.0
    return {"title": f"{os.path.basename(os.path.dirname(os.path.abspath(ep_dir)))} / {os.path.basename(ep_dir)}",
            "meta": {k: meta.get(k) for k in ("started", "duration_s", "frames", "fps_actual", "speed_deg_s",
                                              "allow_down", "dry_run")},
            "fps": fps, "cams": cams, "motors": motors, "frames": frames, "events": ev_out,
            "clean": bool(meta.get("clean_videos")),
            "overlay": bool(cams) and all(os.path.exists(os.path.join(ep_dir, "web", f"overlay_{c}.mp4")) for c in cams)}


def build(ep_dir, quiet=False):
    """web/ videos + viewer.html for one episode (+ table.md / table.csv if missing). Returns the viewer path."""
    if not os.path.exists(os.path.join(ep_dir, "table.md")):
        write_table(ep_dir)
    cams = convert_videos(ep_dir)
    data = json.dumps(_data(ep_dir, cams), separators=(",", ":")).replace("</", "<\\/")
    path = os.path.join(ep_dir, "viewer.html")
    with open(path, "w") as f:
        f.write(PAGE.replace("__TITLE__", html.escape(os.path.basename(ep_dir))).replace("__DATA__", data))
    if not quiet:
        print(f"{ep_dir}: viewer.html ({len(cams)} videos)")
    return path


def build_index(root=RECORD_ROOT):
    """recordings/index.html: every dataset and episode, linking to the viewers."""
    items = []
    for ds in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        dpath = os.path.join(root, ds)
        if not os.path.isdir(dpath):
            continue
        eps = []
        for ep in sorted(d for d in os.listdir(dpath) if re.fullmatch(r"episode_\d+", d)):
            mp = os.path.join(dpath, ep, "meta.json")
            if not os.path.exists(mp):
                continue
            m = json.load(open(mp))
            link = f'<a href="{ds}/{ep}/viewer.html">{ep}</a>' if os.path.exists(
                os.path.join(dpath, ep, "viewer.html")) else ep
            eps.append(f"<tr><td>{link}</td>"
                       f"<td>{m.get('started', '')}</td><td>{m.get('duration_s', '')} s</td>"
                       f"<td>{m.get('frames', '')}</td><td>{m.get('events', '')}</td>"
                       f"<td>{', '.join(m.get('videos', []))}</td></tr>")
        if eps:
            items.append(f"<h2>{html.escape(ds)}</h2><table><tr><th>episode</th><th>started</th><th>length</th>"
                         f"<th>frames</th><th>inputs</th><th>cameras</th></tr>{''.join(eps)}</table>")
    path = os.path.join(root, "index.html")
    with open(path, "w") as f:
        f.write(INDEX.replace("__BODY__", "".join(items) or "<p>No episodes yet.</p>"))
    return path


INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 Recordings</title><style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2c3038;--text:#e6e8eb;--muted:#9aa1ab;--accent:#5aa9ff}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
main{max-width:960px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px;color:var(--muted)}
table{border-collapse:collapse;width:100%;background:var(--panel);border-radius:8px;overflow:hidden}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:500}a{color:var(--accent)}p{color:var(--muted)}
</style></head><body><main><h1>SO-101 recordings</h1><p>Open an episode to see its camera frames, joints and inputs.
Rebuild with <code>python so101_real/episode_viewer.py recordings</code>.</p>__BODY__</main></body></html>
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ Viewer</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--panel2:#23272e;--line:#2c3038;--text:#e6e8eb;--muted:#9aa1ab;--accent:#5aa9ff;
--state:#5aa9ff;--action:#f0a35a;--move:rgba(90,169,255,.10);--cur:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
header h1{font-size:16px;margin:0}header .m{color:var(--muted)}header a{color:var(--accent)}
main{padding:12px 16px;max-width:1700px;margin:0 auto}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.cam{background:var(--panel);border-radius:8px;overflow:hidden}
.cam .lbl{padding:5px 9px;color:var(--muted);display:flex;justify-content:space-between}
.cam video{display:block;width:100%;aspect-ratio:4/3;background:#000}
.ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:10px 0 4px}
button,select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}button.on{background:var(--accent);color:#000;border-color:var(--accent)}
#slider{flex:1 1 300px;accent-color:var(--accent)}
.now{font-variant-numeric:tabular-nums;color:var(--muted);min-width:260px}
.now b{color:var(--text)}
canvas{display:block;width:100%;background:var(--panel);border-radius:8px;cursor:crosshair}
.row2{display:grid;grid-template-columns:minmax(280px,420px) 1fr;gap:10px;margin:8px 0 10px}
@media (max-width:800px){.row2{grid-template-columns:1fr}}
.card{background:var(--panel);border-radius:8px;padding:10px 12px}
.card h3{margin:0 0 6px;font-size:13px;color:var(--muted);font-weight:500}
.jt{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
.jt td,.jt th{padding:3px 6px;text-align:right;border-bottom:1px solid var(--line)}.jt td:first-child,.jt th:first-child{text-align:left}
.jt th{color:var(--muted);font-weight:500}
.evs{max-height:170px;overflow:auto;position:relative}.ev{padding:2px 0;display:flex;gap:8px;cursor:pointer}.ev:hover{color:var(--accent)}
.ev .t{color:var(--muted);min-width:62px;font-variant-numeric:tabular-nums}.ev.here{color:#fff;font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-top:4px;flex:none}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);margin:6px 0}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px}
.tbar{display:flex;gap:10px;align-items:center;margin:14px 0 6px;flex-wrap:wrap}
.tbar h2{font-size:14px;margin:0}label{color:var(--muted)}
.ms{position:relative;display:inline-block}
.ms>button{min-width:260px;text-align:left;display:flex;justify-content:space-between;gap:10px}
.ms .pan{display:none;position:absolute;top:calc(100% + 4px);left:0;z-index:5;min-width:300px;max-height:380px;overflow:auto;
 background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:6px;box-shadow:0 8px 24px rgba(0,0,0,.45)}
.ms.open .pan{display:block}
.ms .quick{display:flex;gap:6px;padding:2px 2px 6px;border-bottom:1px solid var(--line);margin-bottom:4px}
.ms .quick button{padding:3px 8px;font-size:12px}
.ms .opt{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:5px;cursor:pointer;color:var(--text)}
.ms .opt:hover{background:#2c313a}.ms .opt .n{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
.ms .grp{color:var(--muted);font-size:11px;padding:6px 6px 2px;text-transform:uppercase;letter-spacing:.04em}
.tw{max-height:420px;overflow:auto;border-radius:8px;background:var(--panel);position:relative}
table.big{border-collapse:collapse;width:100%;min-width:900px;table-layout:fixed;font-variant-numeric:tabular-nums}
table.big th{position:sticky;top:0;background:var(--panel2);color:var(--muted);font-weight:500;text-align:left;padding:6px 8px;z-index:1}
table.big td{padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
table.big td.a{white-space:normal}
table.big tr{cursor:pointer}table.big tr:hover td{background:#262a31}
table.big tr.hasev td{background:rgba(240,163,90,.08)}table.big tr.cur td{background:rgba(90,169,255,.28)}
table.big td .in{color:var(--action)}
.help{color:var(--muted);font-size:12px}
</style></head><body>
<header><h1 id="title"></h1><span class="m" id="metaline"></span>
<span class="help">Space play / pause &nbsp; ← → one frame &nbsp; Shift+← → 10 frames &nbsp; [ ] previous / next input (of the table filter) &nbsp; click a row, the timeline or the chart to jump</span></header>
<main>
<div class="cams" id="cams"></div>
<div class="ctl">
  <button id="b10" title="10 frames back (Shift+←)">−10</button><button id="b1" title="1 frame back (←)">−1</button>
  <button id="play" title="Space">▶ Play</button>
  <button id="f1" title="1 frame on (→)">+1</button><button id="f10" title="10 frames on (Shift+→)">+10</button>
  <button id="pe" title="previous input ([)">◀ input</button><button id="ne" title="next input (])">input ▶</button>
  <select id="rate" title="playback speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>
  <select id="pics" title="clean = the pictures a model is trained on; as shown = with the overlays you saw"></select>
  <input type="range" id="slider" min="0" value="0">
  <span class="now" id="now"></span>
</div>
<div class="row2">
  <div class="card"><h3>This frame</h3><table class="jt" id="jt"></table></div>
  <div class="card"><h3>Inputs (click to jump)</h3><div class="evs" id="evs"></div></div>
</div>
<div class="tbar"><h2>Table</h2>
  <label for="msbtn">show</label>
  <div class="ms" id="ms"><button id="msbtn" type="button" title="which rows the table (and the inputs list, [ ] keys) show"><span id="mslbl">all rows</span><span>▾</span></button>
    <div class="pan" id="mspan"></div></div>
  <span class="help" id="fcount"></span>
  <label><input type="checkbox" id="follow" checked> follow the current frame</label></div>
<div class="tw" id="tw"><table class="big"><colgroup><col style="width:190px"><col style="width:125px"><col style="width:140px"><col style="width:135px"><col></colgroup><thead><tr><th>time stamp</th><th>Top Frame</th><th>Wrist Frame</th><th>Base Frame</th><th>Action</th></tr></thead><tbody id="tb"></tbody></table></div>

<canvas id="chart" height="300" style="margin-top:12px"></canvas>
<div class="legend"><span><i style="background:var(--state)"></i>state (where the joint was)</span>
<span><i style="background:var(--action)"></i>action (what it was told)</span>
<span><i style="background:rgba(90,169,255,.35);height:10px"></i>moving</span><span id="evlegend"></span></div>
</main>
<script>
const D = __DATA__;
const N = D.frames.length, FPS = D.fps, M = D.motors;
const SHORT = {shoulder_pan:"pan",shoulder_lift:"lift",elbow_flex:"elbow",wrist_flex:"wrist",wrist_roll:"roll",gripper:"grip"};
const EVC = {click:"#ffd84d",goal:"#5ad16a",plan:"#e070e0",execute:"#5ad1d1",move_start:"#5ad1d1",move_done:"#5ad1d1",
             gripper:"#ff6b6b",gripper_close:"#ff6b6b",gripper_open:"#ff9f43",hold:"#ffffff",surface:"#9aa1ab",clear:"#9aa1ab",episode_start:"#9aa1ab"};
const $ = id => document.getElementById(id);
$("title").textContent = D.title;
const mm = D.meta;
$("metaline").textContent = `started ${mm.started} · ${mm.duration_s} s · ${mm.frames} frames at ${mm.fps_actual} fps · speed ${mm.speed_deg_s} deg/s` + (mm.dry_run ? " · DRY RUN" : "");
// ── videos
const vids = {};
for (const c of D.cams) {
  const d = document.createElement("div"); d.className = "cam";
  d.innerHTML = `<div class="lbl"><span>${c}</span><span id="fl_${c}"></span></div><video id="v_${c}" src="web/${c}.mp4" preload="auto" muted playsinline></video>`;
  $("cams").appendChild(d); vids[c] = d.querySelector("video");
}
if (!D.cams.length) $("cams").innerHTML = "<p>No videos.</p>";
const master = vids[D.cams[0]];
// ── clean pictures (training data) or as shown (overlays); old recordings only have the shown ones
$("pics").innerHTML = D.clean ? `<option value="">clean pictures</option>` + (D.overlay ? `<option value="overlay_">as shown (overlays)</option>` : "")
                              : `<option value="">as shown (overlays) - old recording, no clean pictures</option>`;
$("pics").disabled = $("pics").options.length < 2;
$("pics").onchange = e => {
  pause(); const i = cur;
  for (const c of D.cams) { const v = vids[c]; v.src = `web/${e.target.value}${c}.mp4`;
    v.addEventListener("loadeddata", () => { v.currentTime = (i + 0.5) / FPS; }, {once: true}); }
};
// ── events per frame
const evAt = {}; D.events.forEach((e, k) => (evAt[e.f] = evAt[e.f] || []).push(k));
// ── filter (multi-select dropdown): no group ticked = all rows; else rows with an input of a ticked group
const groups = {}; D.events.forEach(e => groups[e.g] = (groups[e.g] || 0) + 1);
const ORDER = ["click top", "click wrist", "click base", "goal", "plan", "ENTER", "move", "gripper", "HOLD", "surface", "clear", "episode"];
const rank = g => { const k = ORDER.findIndex(o => g.startsWith(o)); return String(k < 0 ? 99 : k).padStart(2, "0") + g; };
const GROUPS = Object.keys(groups).sort((a, b) => rank(a) < rank(b) ? -1 : 1);
const sel = new Set();
const evMatch = e => sel.size === 0 || sel.has(e.g);
function buildFilter() {
  const sec = g => g.startsWith("click") ? "clicks" : /^(goal|plan|ENTER|move)/.test(g) ? "goal, plan, moves" : "other";
  let h = `<div class="quick"><button type="button" data-q="none">all rows</button><button type="button" data-q="clicks">all clicks</button><button type="button" data-q="every">every input</button></div>`, last = "";
  for (const g of GROUPS) {
    if (sec(g) !== last) { last = sec(g); h += `<div class="grp">${last}</div>`; }
    h += `<label class="opt"><input type="checkbox" value="${g}" ${sel.has(g) ? "checked" : ""}><span>${g}</span><span class="n">${groups[g]}</span></label>`;
  }
  $("mspan").innerHTML = h;
  updateLabel();
}
function updateLabel() {
  $("mslbl").textContent = sel.size === 0 ? `all rows (${N})` : sel.size === GROUPS.length ? "every input" :
    [...sel].slice(0, 2).join(", ") + (sel.size > 2 ? ` +${sel.size - 2}` : "");
}
function applyFilter(rebuild) { rebuild ? buildFilter() : updateLabel(); refreshEvFrames(); buildEvList(); buildTable(); show(cur, false); }
$("msbtn").onclick = e => { e.stopPropagation(); $("ms").classList.toggle("open"); };
$("mspan").onclick = e => e.stopPropagation();
document.addEventListener("click", () => $("ms").classList.remove("open"));
$("mspan").addEventListener("change", e => { const g = e.target.value; e.target.checked ? sel.add(g) : sel.delete(g); applyFilter(false); });
$("mspan").addEventListener("click", e => { const q = e.target.dataset && e.target.dataset.q; if (!q) return;
  sel.clear(); if (q === "clicks") GROUPS.filter(g => g.startsWith("click")).forEach(g => sel.add(g));
  if (q === "every") GROUPS.forEach(g => sel.add(g)); applyFilter(true); });
let evFrames = [];
function refreshEvFrames() {
  evFrames = [...new Set(D.events.filter(e => evMatch(e) && (sel.size > 0 || e.k !== "move_start")).map(e => e.f))].sort((a, b) => a - b);
}
buildFilter(); refreshEvFrames();
$("evlegend").innerHTML = Object.entries({click:"click",goal:"goal",plan:"plan",execute:"ENTER / move",gripper_close:"gripper close",gripper_open:"gripper open",hold:"HOLD"})
  .map(([k, n]) => `<span style="margin-right:10px"><span class="dot" style="background:${EVC[k]};margin-right:4px"></span>${n}</span>`).join("");
// ── state
let cur = 0, playing = false;
// scroll el into view inside its own scroll box only (scrollIntoView would also scroll the page)
function within(box, el) {
  const top = el.offsetTop - box.firstElementChild.offsetTop, h = el.offsetHeight;
  const head = box.querySelector("thead") ? box.querySelector("thead").offsetHeight : 0;
  if (top - head < box.scrollTop) box.scrollTop = top - head;
  else if (top + h > box.scrollTop + box.clientHeight) box.scrollTop = top + h - box.clientHeight;
}
const fmtJ = arr => arr.map((v, k) => `${SHORT[M[k]] || M[k]} ${v.toFixed(1)}`).join(", ");
function show(i, seek = true) {
  i = Math.max(0, Math.min(N - 1, i | 0)); const prev = cur; cur = i;
  const f = D.frames[i];
  $("slider").value = i;
  $("now").innerHTML = `frame <b>${i}</b> / ${N - 1} &nbsp; <b>${f.t.toFixed(2)} s</b> &nbsp; ${f.c}` + (f.m ? " &nbsp; <b style='color:var(--state)'>moving</b>" : "");
  for (const c of D.cams) $("fl_" + c).textContent = `${c}.mp4 #${i}`;
  if (seek && !playing) for (const v of Object.values(vids)) v.currentTime = (i + 0.5) / FPS;
  let h = "<tr><th>joint</th><th>state</th><th>action</th><th>diff</th></tr>";
  M.forEach((m, k) => { const s = f.s[k], a = f.a[k];
    h += `<tr><td>${m}</td><td>${s.toFixed(1)}</td><td>${a.toFixed(1)}</td><td>${(s - a).toFixed(1)}</td></tr>`; });
  $("jt").innerHTML = h;
  document.querySelectorAll(".ev.here").forEach(e => e.classList.remove("here"));
  // last input at or before this frame
  let last = -1; for (let k = 0; k < D.events.length && D.events[k].f <= i; k++) if (evMatch(D.events[k])) last = k;
  if (last >= 0 && $("ev" + last)) { const el = $("ev" + last); el.classList.add("here"); if (prev !== i) within($("evs"), el); }
  const old = document.querySelector("tr.cur"); if (old) old.classList.remove("cur");
  const row = $("r" + i); if (row) { row.classList.add("cur"); if ($("follow").checked) within($("tw"), row); }
  drawChart();
}
// ── controls
$("slider").max = N - 1;
$("slider").oninput = e => { pause(); show(+e.target.value); };
$("b1").onclick = () => { pause(); show(cur - 1); }; $("f1").onclick = () => { pause(); show(cur + 1); };
$("b10").onclick = () => { pause(); show(cur - 10); }; $("f10").onclick = () => { pause(); show(cur + 10); };
const nextEv = d => { const fs = d > 0 ? evFrames.find(f => f > cur) : [...evFrames].reverse().find(f => f < cur); if (fs !== undefined) { pause(); show(fs); } };
$("ne").onclick = () => nextEv(1); $("pe").onclick = () => nextEv(-1);
$("rate").onchange = e => { for (const v of Object.values(vids)) v.playbackRate = +e.target.value; };
$("play").onclick = () => playing ? pause() : play();
function play() {
  if (!master) return; if (cur >= N - 1) show(0);
  playing = true; $("play").textContent = "❚❚ Pause"; $("play").classList.add("on");
  const t = (cur + 0.5) / FPS;
  for (const v of Object.values(vids)) { v.currentTime = t; v.playbackRate = +$("rate").value; v.play(); }
  requestAnimationFrame(tick);
}
function pause() {
  if (!playing) return; playing = false; $("play").textContent = "▶ Play"; $("play").classList.remove("on");
  for (const v of Object.values(vids)) v.pause();
  show(cur);
}
function tick() {
  if (!playing) return;
  const i = Math.min(N - 1, Math.floor(master.currentTime * FPS));
  for (const v of Object.values(vids)) if (v !== master && Math.abs(v.currentTime - master.currentTime) > 0.15) v.currentTime = master.currentTime;
  if (i !== cur) show(i, false);
  if (master.ended || i >= N - 1) { pause(); return; }
  requestAnimationFrame(tick);
}
document.addEventListener("keydown", e => {
  if (e.target.closest && e.target.closest("#ms")) return;
  if (e.target.tagName === "SELECT" || (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox")) return;
  if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); pause(); show(cur + (e.shiftKey ? 10 : 1)); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); pause(); show(cur - (e.shiftKey ? 10 : 1)); }
  else if (e.key === "]") nextEv(1); else if (e.key === "[") nextEv(-1);
});
// ── inputs list
function buildEvList() {
  $("evs").innerHTML = D.events.map((e, k) => evMatch(e) ?
    `<div class="ev" id="ev${k}" data-f="${e.f}"><span class="t">${e.t.toFixed(2)} s</span><span class="dot" style="background:${EVC[e.k] || "#9aa1ab"}"></span><span>${e.x}</span></div>` : "").join("");
}
buildEvList();
$("evs").onclick = e => { const el = e.target.closest(".ev"); if (el) { pause(); show(+el.dataset.f); } };
// ── table
function buildTable() {
  const parts = [];
  for (let i = 0; i < N; i++) {
    const ev = evAt[i];
    if (sel.size && !(ev && ev.some(k => sel.has(D.events[k].g)))) continue;
    const f = D.frames[i];
    const cells = ["top", "wrist", "base"].map(c => D.cams.includes(c) ? `${c}.mp4 #${i}` : "-");
    const act = fmtJ(f.a) + (f.m ? " (moving)" : "") + (ev ? ` <span class="in">· ${ev.map(k => sel.has(D.events[k].g) ? `<b>${D.events[k].x}</b>` : D.events[k].x).join("; ")}</span>` : "");
    parts.push(`<tr id="r${i}" data-i="${i}" class="${ev ? "hasev" : ""}"><td>${f.t.toFixed(3)} s (${f.c})</td><td>${cells[0]}</td><td>${cells[1]}</td><td>${cells[2]}</td><td class="a">${act}</td></tr>`);
  }
  $("tb").innerHTML = parts.join("") || `<tr><td colspan="5">no rows</td></tr>`;
  $("fcount").textContent = `${parts.length} row${parts.length === 1 ? "" : "s"}`;
}
$("tb").onclick = e => { const tr = e.target.closest("tr"); if (tr) { pause(); show(+tr.dataset.i); } };
// ── chart: one strip per joint (state solid, action dashed), moving shaded, inputs as ticks, cursor
const cv = $("chart"), cx = cv.getContext("2d");
const ranges = M.map((_, k) => { let lo = 1e9, hi = -1e9;
  for (const f of D.frames) for (const v of [f.s[k], f.a[k]]) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = Math.max(2, (hi - lo) * 0.08); return [lo - pad, hi + pad]; });
const EVH = 16, LEFT = 74;
function layout() { const dpr = window.devicePixelRatio || 1; const w = cv.clientWidth;
  const h = EVH + M.length * 52 + 18; cv.style.height = h + "px"; cv.width = w * dpr; cv.height = h * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0); return [w, h]; }
let W = 0, H = 0;
function drawChart() {
  if (!W) [W, H] = layout();
  cx.clearRect(0, 0, W, H);
  const pw = W - LEFT - 10, X = i => LEFT + (N > 1 ? i / (N - 1) : 0) * pw, sh = 52;
  // moving spans
  cx.fillStyle = "rgba(90,169,255,.10)";
  for (let i = 0; i < N; i++) if (D.frames[i].m) { let j = i; while (j < N && D.frames[j].m) j++; cx.fillRect(X(i), EVH, Math.max(1, X(j - 1) - X(i)), M.length * sh); i = j; }
  // input ticks
  for (const e of D.events) { if (e.k === "move_start") continue; cx.fillStyle = EVC[e.k] || "#9aa1ab"; cx.fillRect(X(e.f) - 1, 2, 3, EVH - 5); }
  M.forEach((m, k) => {
    const y0 = EVH + k * sh, [lo, hi] = ranges[k], Y = v => y0 + sh - 6 - (v - lo) / (hi - lo) * (sh - 12);
    cx.strokeStyle = "#2c3038"; cx.lineWidth = 1; cx.beginPath(); cx.moveTo(LEFT, y0 + sh); cx.lineTo(W - 10, y0 + sh); cx.stroke();
    cx.fillStyle = "#9aa1ab"; cx.font = "12px system-ui"; cx.fillText(SHORT[m] || m, 8, y0 + sh / 2 + 4);
    cx.font = "10px system-ui"; cx.fillText(hi.toFixed(0), 44, y0 + 12); cx.fillText(lo.toFixed(0), 44, y0 + sh - 4);
    for (const [key, col, dash] of [["a", "#f0a35a", [5, 4]], ["s", "#5aa9ff", []]]) {
      cx.strokeStyle = col; cx.setLineDash(dash); cx.lineWidth = 1.6; cx.beginPath();
      D.frames.forEach((f, i) => { const v = f[key][k]; if (!isFinite(v)) return; i ? cx.lineTo(X(i), Y(v)) : cx.moveTo(X(i), Y(v)); });
      cx.stroke(); cx.setLineDash([]);
    }
  });
  cx.fillStyle = "#9aa1ab"; cx.font = "11px system-ui";
  for (let s = 0; s <= D.frames[N - 1].t; s += Math.max(5, Math.ceil(D.frames[N - 1].t / 12 / 5) * 5)) {
    const i = Math.round(s * FPS); if (i < N) cx.fillText(s + " s", X(i) - 8, H - 4); }
  cx.strokeStyle = "#ffffff"; cx.lineWidth = 1.5; cx.beginPath(); cx.moveTo(X(cur), 0); cx.lineTo(X(cur), EVH + M.length * sh); cx.stroke();
}
function chartSeek(e) { const r = cv.getBoundingClientRect(); const pw = W - LEFT - 10;
  pause(); show(Math.round((e.clientX - r.left - LEFT) / pw * (N - 1))); }
let drag = false;
cv.onmousedown = e => { drag = true; chartSeek(e); }; window.onmouseup = () => (drag = false);
cv.onmousemove = e => { if (drag) chartSeek(e); };
window.onresize = () => { W = 0; drawChart(); };
buildTable(); show(0);
</script></body></html>
"""


def main():
    args = [a for a in sys.argv[1:] if a != "--open"]
    if len(args) != 1:
        raise SystemExit(__doc__.split("\n\nPer episode")[0])
    target = os.path.abspath(args[0])
    if os.path.exists(os.path.join(target, "frames.jsonl")):
        eps = [target]
    else:
        eps = []
        for root, dirs, files in os.walk(target):
            if "frames.jsonl" in files and re.fullmatch(r"episode_\d+", os.path.basename(root)):
                eps.append(root)
        eps.sort()
    if not eps:
        raise SystemExit(f"no episodes in {target}")
    for d in eps:
        build(d)
    idx = build_index(os.path.dirname(os.path.dirname(eps[0])))
    print(f"index: {idx}")
    if "--open" in sys.argv:
        subprocess.Popen(["xdg-open", idx if len(eps) > 1 else os.path.join(eps[0], "viewer.html")])


if __name__ == "__main__":
    main()
