"""Training rows from recorded episodes (click_goal_real.py --record): one row per DECISION you made - where you
clicked (top / wrist / base camera) and when you closed / opened the gripper - with the three clean camera pictures
of the moment you decided. A model trained on them answers "what next?" each time the arm has stopped; the click
tool's own pipeline (rays, IK, arm_server) still does the moving.

    python so101_real/build_decisions.py recordings/11_11_2026_pick_white_cube_2
    python so101_real/build_decisions.py recordings/DAY1_pick recordings/DAY2_pick --out decisions/pick
    python so101_real/build_decisions.py recordings/... --include-overlay     # also old episodes (overlays burned in)

What one decision is
    click_<camera>   the LAST click of that camera before an ENTER that ran the plan (a burst of re-clicks = one
                     decision; clicks that never got an ENTER, or were cleared with C, are dropped). The picture is
                     the one just before the FIRST click of the burst: what you looked at when you decided. A wrist /
                     base click is combined with the top click still stored, as in the tool - "kept" says which.
    gripper_close    a gripper close / open from X / O or the motors window that moved the gripper (not the open at
    gripper_open     the episode start); the picture is the one just before the gripper started moving.
                     "moving": true = you decided while the arm was still moving (a click during a move is
                     carried out after it): the picture is from the move, the click went into the stopped pose.
    done             the last frame of an episode that closed and then opened the gripper (a finished pick + place).
Episodes without a close followed by an open are skipped as unfinished (--keep-unfinished keeps them, without
"done"). Episodes recorded before the clean pictures (meta.json without "clean_videos") have the overlays burned in
and are skipped unless --include-overlay.

Output (--out, default <dataset>/decisions):
    decisions.jsonl   one line per decision:
                        id, dataset, episode, frame, t, label, point {camera, u, v (camera pixels as clicked),
                        x, y (0..1 of the picture)} or null, burst (raw clicks merged), kept (other cameras' clicks
                        still in use), goal (3D, m) / plan (hover / down) of the move it started, holding (the gripper
                        was closed before this decision), state (joints), images {top, wrist, base} (jpg paths
                        relative to the output folder),
                        history = every earlier decision of the episode as a step {label, x, y} (clicks) or
                        {label, value, stopped_at} (gripper: where it stopped - early = something in the jaws),
                        recent = the last --history (3) of them: what the model gets with the pictures, so it can
                        learn your steering ("I clicked there, the arm ended here, so now click a bit further")
    images/           <dataset>__epNNN__dKK_<camera>.jpg
    preview.html      every decision with its three pictures and the clicked point - check the labels here
"""
import os
import re
import sys
import json
import html
import bisect
import argparse
import collections

import cv2

CAMS = ("top", "wrist", "base")


def _load(ep):
    rows = [json.loads(line) for line in open(os.path.join(ep, "frames.jsonl")) if line.strip()]
    events = sorted((json.loads(line) for line in open(os.path.join(ep, "events.jsonl")) if line.strip()),
                    key=lambda e: e["t"])
    return rows, events, json.load(open(os.path.join(ep, "meta.json")))


def _frame_before(ts, t):
    """Index of the last frame recorded before time t (the picture on screen when you acted)."""
    return max(0, bisect.bisect_left(ts, t) - 1)


def _video_sizes(ep):
    """{camera: (width, height)} of the episode's videos (the pictures a point's x, y are fractions of)."""
    out = {}
    for c in CAMS:
        p = os.path.join(ep, f"{c}.mp4")
        if os.path.exists(p):
            cap = cv2.VideoCapture(p)
            out[c] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
    return out


def _step(d):
    """A decision as one history step: label + point (x, y) or the gripper value + where it stopped."""
    s = {"label": d["label"]}
    p = d.get("point")
    if p and "x" in p:
        s.update(x=p["x"], y=p["y"])
    if d["label"].startswith("gripper"):
        s.update(value=d.get("value"), stopped_at=d.get("stopped_at"))
    return s


def step_text(s):
    """'click_wrist (0.313, 0.321)' / 'gripper_close (stopped at 45.5)' / 'gripper_open'."""
    if "x" in s:
        return f"{s['label']} ({s['x']:.3f}, {s['y']:.3f})"
    if s["label"] == "gripper_close" and s.get("stopped_at") is not None:
        return f"gripper_close (stopped at {s['stopped_at']:.1f})"
    return s["label"]


def episode_decisions(ep, n_recent=3):
    """[(decision dict without images), ...] and a note (why the episode is unfinished) or None."""
    rows, events, meta = _load(ep)
    ts = [r["t"] for r in rows]
    sizes = _video_sizes(ep)
    scale = {c: ((meta.get("cameras") or {}).get(c) or {}).get("view_scale", 1.0) for c in CAMS}
    out, pending, last_goal, last_plan = [], collections.OrderedDict(), None, None
    for k, e in enumerate(events):
        kind = e["kind"]
        if kind == "click":
            pending.setdefault(e["camera"], []).append(e)
        elif kind == "clear":
            pending.clear()
        elif kind == "goal":
            last_goal = e
        elif kind == "plan":
            last_plan = e
        elif kind == "execute":
            used = (last_goal or {}).get("clicks") or {}
            for cam, clicks in sorted(pending.items(), key=lambda kv: kv[1][0]["t"]):
                if cam not in used:
                    continue
                first, last = clicks[0], clicks[-1]
                point = {"camera": cam, "u": last["u"], "v": last["v"]}
                if cam in sizes:                                     # x, y = fractions of the clean picture
                    w, h = sizes[cam]
                    point.update(x=round(last["u"] * scale[cam] / w, 4), y=round(last["v"] * scale[cam] / h, 4))
                out.append({"label": f"click_{cam}", "t_decide": first["t"], "point": point, "burst": len(clicks),
                            "kept": sorted(c for c in used if c != cam),
                            "goal": (last_goal or {}).get("goal"), "plan": (last_plan or {}).get("plan")})
            pending.clear()
        elif kind in ("gripper_open", "gripper_close"):
            if e.get("source") == "episode start" or e.get("moved") is False:
                continue
            starts = [s for s in events[:k] if s["kind"] == "move_start" and set(s.get("target") or {}) == {"gripper"}]
            t = starts[-1]["t"] if starts else e["t"]
            out.append({"label": kind, "t_decide": t, "point": None, "value": e.get("value"),
                        "stopped_at": e.get("now"), "blocked": e.get("blocked")})
        elif kind == "gripper" and e.get("value") is not None:              # recordings before 11-11-2026
            out.append({"label": "gripper_close" if e.get("key") == "x" else "gripper_open", "t_decide": e["t"],
                        "point": None, "value": e.get("value")})
    out.sort(key=lambda d: d["t_decide"])
    labels = [d["label"] for d in out]
    finished = "gripper_close" in labels and "gripper_open" in labels[labels.index("gripper_close"):]
    if finished:
        out.append({"label": "done", "t_decide": rows[-1]["t"] + 1.0, "point": None})
    holding, hist = False, []
    for d in out:
        i = min(_frame_before(ts, d["t_decide"]), len(rows) - 1)
        d.update(frame=i, t=rows[i]["t"], moving=d["label"] != "done" and bool(rows[i].get("moving")),
                 state=rows[i]["state"],
                 holding=holding, history=list(hist), recent=hist[-n_recent:] if n_recent > 0 else [])
        if d["label"] == "gripper_close":
            holding = True
        elif d["label"] == "gripper_open":
            holding = False
        hist.append(_step(d))
    note = None if finished else "no gripper close followed by an open (unfinished pick + place?)"
    return out, meta, note


def read_frames(path, wanted):
    """{index: BGR image} of the wanted frame indexes of one video."""
    cap, got, i, want = cv2.VideoCapture(path), {}, 0, set(wanted)
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = f
            want.discard(i)
        i += 1
    cap.release()
    return got


def find_episodes(targets):
    eps = []
    for t in targets:
        if os.path.exists(os.path.join(t, "frames.jsonl")):
            eps.append(t)
        elif os.path.isdir(t):
            eps += sorted(os.path.join(t, d) for d in os.listdir(t) if re.fullmatch(r"episode_\d+", d))
    return eps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+", help="dataset folders (recordings/<dd_mm_yyyy>_NAME) or episode folders")
    ap.add_argument("--out", help="output folder (default: <dataset>/decisions; needed for several datasets)")
    ap.add_argument("--include-overlay", action="store_true",
                    help="also episodes recorded before the clean pictures (overlays in the images!)")
    ap.add_argument("--keep-unfinished", action="store_true", help="also episodes without a close then an open")
    ap.add_argument("--quality", type=int, default=92, help="jpg quality")
    ap.add_argument("--history", type=int, default=3,
                    help="how many earlier decisions (with their points) go into 'recent' - what the model sees")
    args = ap.parse_args()

    eps = find_episodes(args.targets)
    if not eps:
        raise SystemExit("no episodes found in " + " ".join(args.targets))
    if args.out is None:
        roots = {os.path.dirname(os.path.abspath(e)) for e in eps}
        if len(roots) > 1:
            raise SystemExit("episodes from several datasets: say where the rows go with --out")
        args.out = os.path.join(roots.pop(), "decisions")
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    rows_out, counts, skipped = [], collections.Counter(), []
    for ep in eps:
        ds, epn = os.path.basename(os.path.dirname(os.path.abspath(ep))), os.path.basename(ep)
        decisions, meta, note = episode_decisions(ep, args.history)
        if not meta.get("clean_videos") and not args.include_overlay:
            skipped.append(f"{ds}/{epn}: overlays burned into the pictures (recorded before the clean videos)")
            continue
        if note and not args.keep_unfinished:
            skipped.append(f"{ds}/{epn}: {note}")
            continue
        cams = [c for c in CAMS if os.path.exists(os.path.join(ep, f"{c}.mp4"))]
        pics = {c: read_frames(os.path.join(ep, f"{c}.mp4"), [d["frame"] for d in decisions]) for c in cams}
        for n, d in enumerate(decisions):
            rid = f"{ds}__{epn.replace('episode_', 'ep')}__d{n:02d}"
            images = {}
            for c in cams:
                img = pics[c].get(d["frame"])
                if img is None:
                    continue
                rel = f"images/{rid}_{c}.jpg"
                cv2.imwrite(os.path.join(args.out, rel), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                images[c] = rel
            row = {"id": rid, "dataset": ds, "episode": epn, "clean": bool(meta.get("clean_videos")),
                   **{k: v for k, v in d.items() if k != "t_decide"}, "images": images}
            rows_out.append(row)
            counts[d["label"]] += 1

    with open(os.path.join(args.out, "decisions.jsonl"), "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    page = os.path.join(args.out, "preview.html")
    with open(page, "w") as f:
        f.write(preview_html(rows_out))
    for s in skipped:
        print("skipped", s)
    print(f"{len(rows_out)} decisions from {len({(r['dataset'], r['episode']) for r in rows_out})} episodes -> "
          f"{args.out}/decisions.jsonl")
    for k in ("click_top", "click_wrist", "click_base", "gripper_close", "gripper_open", "done"):
        print(f"  {k:14s} {counts[k]}")
    print(f"check them: {page}")


def preview_html(rows):
    items = []
    for r in rows:
        tiles = []
        for c in CAMS:
            if c not in r["images"]:
                continue
            mark = ""
            p = r["point"]
            if p and p["camera"] == c and "x" in p:
                mark = f'<i class="pt" style="left:{p["x"] * 100:.2f}%;top:{p["y"] * 100:.2f}%"></i>'
            tiles.append(f'<div class="pic"><img loading="lazy" src="{html.escape(r["images"][c])}">{mark}'
                         f'<span>{c}</span></div>')
        p = r["point"]
        what = f'{p["camera"]} ({p["u"]:.0f}, {p["v"]:.0f})' + (f' · {r["burst"]} clicks → last' if r.get("burst", 1) > 1 else "") \
            if p else (f'gripper → {r["value"]:.0f}' if r.get("value") is not None else "")
        extra = []
        if r.get("kept"):
            extra.append("with stored " + "+".join(r["kept"]) + " click")
        if r.get("plan"):
            extra.append(f'plan {r["plan"]}')
        if r.get("moving"):
            extra.append('<b class="warn">decided while the arm was moving</b>')
        if not r.get("clean"):
            extra.append('<b class="warn">overlays in pictures</b>')
        recent = " → ".join(step_text(s) for s in r.get("recent", [])) or "(first decision)"
        items.append(
            f'<section data-l="{r["label"]}"><div class="rc">model also sees: {html.escape(recent)}</div>'
            f'<div class="hd"><b class="lab {r["label"]}">{r["label"]}</b>'
            f'<span>{html.escape(what)}</span><span class="m">{r["dataset"]} / {r["episode"]} · frame {r["frame"]} · '
            f'{r["t"]:.1f} s · {"holding" if r["holding"] else "empty gripper"} · {" · ".join(extra)}</span></div>'
            f'<div class="row">{"".join(tiles)}</div></section>')
    counts = collections.Counter(r["label"] for r in rows)
    buttons = "".join(f'<label><input type="checkbox" value="{k}" checked> {k} ({counts[k]})</label>'
                      for k in ("click_top", "click_wrist", "click_base", "gripper_close", "gripper_open", "done")
                      if counts[k])
    return PREVIEW.replace("__N__", str(len(rows))).replace("__BTN__", buttons).replace("__BODY__", "".join(items))


PREVIEW = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision Rows</title><style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2c3038;--text:#e6e8eb;--muted:#9aa1ab}
body{margin:0;padding:20px 16px;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto}h1{font-size:20px;margin:0 0 6px}p{color:var(--muted);margin:0 0 10px}
.f{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 16px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2}
section{background:var(--panel);border-radius:8px;padding:10px;margin:0 0 12px}
.rc{color:var(--muted);font-size:12px;margin:0 0 6px;font-family:ui-monospace,monospace}
.hd{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;margin:0 0 8px}.m{color:var(--muted);font-size:12px}
.lab{padding:2px 8px;border-radius:4px;color:#111}.click_top{background:#ffd84d}.click_wrist{background:#7fd4ff}
.click_base{background:#c79bff}.gripper_close{background:#ff6b6b}.gripper_open{background:#ff9f43}.done{background:#5ad16a}
.warn{color:#ff6b6b}.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.pic{position:relative}.pic img{width:100%;display:block;border-radius:4px}
.pic span{position:absolute;left:6px;top:4px;background:#000a;padding:0 6px;border-radius:3px;font-size:12px}
.pt{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border:3px solid #ff2d2d;border-radius:50%;box-shadow:0 0 0 2px #000}
@media(max-width:700px){.row{grid-template-columns:1fr}}
</style></head><body><main><h1>Decision rows (__N__)</h1>
<p>One row per decision: the clean pictures at the moment you decided; the red ring = where you clicked (drawn here
only, not in the images).</p><div class="f">__BTN__</div>__BODY__</main><script>
const boxes=[...document.querySelectorAll(".f input")];
boxes.forEach(b=>b.onchange=()=>{const on=new Set(boxes.filter(x=>x.checked).map(x=>x.value));
document.querySelectorAll("section").forEach(s=>s.style.display=on.has(s.dataset.l)?"":"none");});
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
