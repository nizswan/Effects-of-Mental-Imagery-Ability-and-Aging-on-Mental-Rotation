#!/usr/bin/env python3
"""
Mental Rotations Test - local webapp + CSV logger

Folder layout:
  mental_rotation_app.py
  imgs/   (contains 64 png stimuli)
Output:
  data.csv   (created/appended)

NEW IN THIS VERSION
1) Port reuse + cleaner restart behavior:
   - allow_reuse_address = True
   - daemon thread auto-opens browser
   - explicit shutdown on window/tab close (best-effort) via /api/unload
   Note: closing the tab cannot *guarantee* the Python process exits (browser may not send unload reliably),
   but allow_reuse_address + optional auto-port-fallback makes repeated runs painless.

2) Attention checks:
   - 5 attention check trials inserted at random positions across the 192 real trials.
   - Each attention check shows a random image (any of the 64) and prompt: "Click Option Same"
   - Buttons shown: "Same" and "Different" (always)
   - Correct response for attention trials is ALWAYS "same"
   - These do NOT count toward the 192 trials; they are extra.
   - Final CSV includes one extra column: attention (0..5), count correct on attention checks.

CSV columns (772 total):
  1) fname
  2) lname
  3) age
  4..(3+192): identifiable times (192) in ms
  next 192: identifiable performance (192) binary
  next 192: order times (192) in ms
  next 192: order performance (192) in binary
  last: attention (0..5)

Identifiable columns are for each cycle (1..3) and each image title (64):
  c{cycle}_{imgtitle}_time_ms
  c{cycle}_{imgtitle}_correct

Order columns:
  order_{i}_time_ms
  order_{i}_correct

Attention trials:
  - not written into the 192 trial columns at all
  - only summarized in final "attention" column
"""

from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

# -------------------------
# Config
# -------------------------
HOST = "127.0.0.1"
PORT = 8787
IMGS_DIR = Path("imgs")
CSV_PATH = Path("data.csv")

ATTN_N = 5  # number of attention checks inserted (extra trials)
AUTO_PORT_FALLBACK = True  # if PORT is busy, try next ports

# -------------------------
# Helpers
# -------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def list_stimuli() -> list[str]:
    if not IMGS_DIR.exists() or not IMGS_DIR.is_dir():
        raise FileNotFoundError(f"Missing folder: {IMGS_DIR.resolve()}")
    pngs = sorted([p.name for p in IMGS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if len(pngs) == 0:
        raise FileNotFoundError(f"No .png files found in: {IMGS_DIR.resolve()}")
    return pngs

def is_reference(imgname: str) -> bool:
    return "reference" in imgname

def is_mirrored(imgname: str) -> bool:
    return "mirrored" in imgname

def correct_answer(imgname: str) -> str:
    """
    For reference images: ask "Same" vs "Different"
      - normal => Same
      - mirrored => Different
    For centered images: ask "Normal" vs "Reversed"
      - normal => Normal
      - mirrored => Reversed
    """
    if is_reference(imgname):
        return "different" if is_mirrored(imgname) else "same"
    else:
        return "reversed" if is_mirrored(imgname) else "normal"

def build_real_trial_sequence(stimuli64: list[str]) -> list[dict]:
    """
    3 cycles. Each cycle is a random permutation of the 64 stimuli.
    Returns list of 192 REAL trial dicts:
      {
        "kind": "real",
        "order_index": 1..192,
        "cycle": 1..3,
        "img": filename
      }
    """
    seq: list[dict] = []
    order = 1
    for cycle in (1, 2, 3):
        perm = stimuli64[:]
        random.shuffle(perm)
        for img in perm:
            seq.append({"kind": "real", "order_index": order, "cycle": cycle, "img": img})
            order += 1
    return seq

def insert_attention_trials(real_trials: list[dict], stimuli64: list[str], n: int) -> list[dict]:
    """
    Insert n attention checks randomly throughout the full flow.
    Attention checks are extra and do NOT change order_index / cycle for real trials.

    Attention trial dict:
      {
        "kind": "attn",
        "img": <random image from 64>,
        "attn_id": 1..n
      }
    """
    if n <= 0:
        return real_trials[:]

    flow = real_trials[:]  # length 192 initially
    # choose distinct insertion positions in range [0..len(flow)] then insert in descending order
    # so earlier inserts don't shift later indices.
    positions = sorted(random.sample(range(len(flow) + 1), k=n), reverse=True)

    for k, pos in enumerate(positions, start=1):
        img = random.choice(stimuli64)
        flow.insert(pos, {"kind": "attn", "img": img, "attn_id": k})

    return flow

def build_csv_header(stimuli64_sorted: list[str]) -> list[str]:
    # 1) identity
    header = ["fname", "lname", "age"]

    # 2) identifiable times (cycle x image)
    for cycle in (1, 2, 3):
        for img in stimuli64_sorted:
            header.append(f"c{cycle}_{img}_time_ms")

    # 3) identifiable performance
    for cycle in (1, 2, 3):
        for img in stimuli64_sorted:
            header.append(f"c{cycle}_{img}_correct")

    # 4) order times
    for i in range(1, 193):
        header.append(f"order_{i}_time_ms")

    # 5) order performance
    for i in range(1, 193):
        header.append(f"order_{i}_correct")

    # 6) attention score (how many attention checks answered correctly)
    header.append("attention")

    assert len(header) == 772, f"Header has {len(header)} columns, expected 772"
    return header

def ensure_csv_exists(header: list[str]) -> None:
    if CSV_PATH.exists():
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

# -------------------------
# In-memory session store
# -------------------------
SESSIONS_LOCK = threading.Lock()
SESSIONS: dict[str, dict] = {}
# session = {
#   "fname": str, "lname": str, "age": str,
#   "stimuli_sorted": [64],
#   "real_trials": [192 dicts],
#   "flow": [192 + ATTN_N dicts],
#   "pos": int (0..len(flow)),
#   "ident_times": {(cycle,img): ms},
#   "ident_corr": {(cycle,img): 0/1},
#   "order_times": [len 192],
#   "order_corr": [len 192],
#   "attn_correct": int,
#   "finished": bool
# }

# -------------------------
# HTML App (served at "/")
# -------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mental Rotations Test</title>
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { margin: 0; background: #f6f6f7; color: #111; }
    .wrap { max-width: 900px; margin: 0 auto; padding: 28px 18px; }
    .card {
      background: #fff; border: 1px solid #e6e6ea; border-radius: 14px;
      padding: 22px; box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    }
    h1,h2 { margin: 0 0 12px; }
    p { margin: 10px 0; line-height: 1.35; color: #333; }
    .center { text-align: center; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .field { flex: 1 1 180px; display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 14px; color: #333; }
    input {
      padding: 10px 12px; border-radius: 10px; border: 1px solid #d7d7de;
      font-size: 16px; outline: none;
    }
    input:focus { border-color: #7b7bf0; box-shadow: 0 0 0 3px rgba(123,123,240,0.18); }
    .btnrow { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-top: 14px; }
    button {
      padding: 12px 16px; border-radius: 12px; border: 1px solid #d7d7de;
      background: #111; color: #fff; font-weight: 650; font-size: 16px; cursor: pointer;
      min-width: 150px;
    }
    button.secondary { background: #fff; color: #111; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .imgbox { display: flex; justify-content: center; margin: 12px 0 6px; }
    img.stim { max-width: 600px; width: 100%; height: auto; border-radius: 12px; border: 1px solid #e6e6ea; }
    .meta { display: flex; justify-content: space-between; gap: 10px; margin-top: 10px; color: #444; font-size: 14px; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; background: #f1f1f3; padding: 2px 6px; border-radius: 6px; border: 1px solid #e3e3e8;}
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card" id="screen-title">
      <h1 class="center">Mental Rotations Test</h1>
      <p class="center">Click start to begin.</p>
      <div class="btnrow">
        <button id="btn-start">Start</button>
      </div>
    </div>

    <div class="card hidden" id="screen-identity">
      <h2 class="center">Participant Information</h2>
      <div class="row">
        <div class="field">
          <label for="fname">First Name</label>
          <input id="fname" autocomplete="given-name" />
        </div>
        <div class="field">
          <label for="lname">Last Name</label>
          <input id="lname" autocomplete="family-name" />
        </div>
        <div class="field">
          <label for="age">Age</label>
          <input id="age" inputmode="numeric" />
        </div>
      </div>
      <div class="btnrow">
        <button class="secondary" id="btn-back">Back</button>
        <button id="btn-begin-test">Start Test</button>
      </div>
      <p id="identity-error" class="center" style="color:#b00020; margin-top:10px;"></p>
    </div>

    <div class="card hidden" id="screen-test">
      <h2 class="center" id="prompt">Prompt</h2>
      <div class="imgbox">
        <img class="stim" id="stim-img" alt="stimulus" />
      </div>

      <div class="btnrow" id="choices"></div>

      <div class="meta">
        <div>Trial: <span class="kbd" id="trial-idx">1</span> / <span class="kbd">192</span></div>
        <div>Cycle: <span class="kbd" id="cycle-idx">1</span> / <span class="kbd">3</span></div>
      </div>
    </div>

    <div class="card hidden" id="screen-done">
      <h2 class="center">Thank you for participating in this research endeavor!</h2>
      <p class="center">You may now close this window.</p>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);

  const screens = {
    title: $("screen-title"),
    identity: $("screen-identity"),
    test: $("screen-test"),
    done: $("screen-done")
  };

  function show(which) {
    for (const k of Object.keys(screens)) screens[k].classList.add("hidden");
    screens[which].classList.remove("hidden");
  }

  let sessionId = null;
  let flow = [];
  let pos = 0;
  let t0 = 0;

  async function api(path, method="GET", body=null) {
    const opts = { method, headers: {} };
    if (body !== null) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    return await res.json();
  }

  function renderTrial(item) {
    // item.kind is "real" or "attn"
    // For display meta, attention trials show the current real-trial count (last seen) but do not increment the 192 counter.
    if (item.kind === "real") {
      $("trial-idx").textContent = String(item.order_index);
      $("cycle-idx").textContent = String(item.cycle);
    } else {
      // attention check: keep trial/cycle display stable (or show n/a)
      $("trial-idx").textContent = $("trial-idx").textContent || "1";
      $("cycle-idx").textContent = $("cycle-idx").textContent || "1";
    }

    $("stim-img").src = `/imgs/${encodeURIComponent(item.img)}`;

    const choices = $("choices");
    choices.innerHTML = "";

    if (item.kind === "attn") {
      $("prompt").textContent = "Click Option Same";
      addChoiceButton("Same", "same");
      addChoiceButton("Different", "different");
    } else {
      const isReference = item.img.includes("reference");
      if (isReference) {
        $("prompt").textContent = "Are these the same letter?";
        addChoiceButton("Same", "same");
        addChoiceButton("Different", "different");
      } else {
        $("prompt").textContent = "Is this letter normal or reversed?";
        addChoiceButton("Normal", "normal");
        addChoiceButton("Reversed", "reversed");
      }
    }

    t0 = performance.now();
  }

  function setButtonsEnabled(enabled) {
    const btns = $("choices").querySelectorAll("button");
    btns.forEach(b => b.disabled = !enabled);
  }

  function addChoiceButton(label, value) {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", async () => {
      if (!sessionId) return;
      setButtonsEnabled(false);
      const dt = Math.max(0, Math.round(performance.now() - t0)); // ms

      const item = flow[pos];

      try {
        await api("/api/record", "POST", {
          session_id: sessionId,
          kind: item.kind,
          order_index: item.kind === "real" ? item.order_index : null,
          cycle: item.kind === "real" ? item.cycle : null,
          img: item.img,
          answer: value,
          time_ms: dt
        });
      } catch (e) {
        console.error(e);
        setButtonsEnabled(true);
        return;
      }

      pos += 1;
      if (pos >= flow.length) {
        try {
          await api("/api/finish", "POST", { session_id: sessionId });
        } catch (e) {
          console.error(e);
        }
        show("done");
        return;
      }

      renderTrial(flow[pos]);
      setButtonsEnabled(true);
    });
    $("choices").appendChild(b);
  }

  $("btn-start").addEventListener("click", () => {
    show("identity");
  });

  $("btn-back").addEventListener("click", () => {
    show("title");
  });

  $("btn-begin-test").addEventListener("click", async () => {
    $("identity-error").textContent = "";
    const fname = $("fname").value.trim();
    const lname = $("lname").value.trim();
    const age = $("age").value.trim();

    if (!fname || !lname || !age) {
      $("identity-error").textContent = "Please fill out First Name, Last Name, and Age.";
      return;
    }

    try {
      const resp = await api("/api/start", "POST", { fname, lname, age });
      sessionId = resp.session_id;
      flow = resp.flow;
      pos = 0;

      show("test");
      renderTrial(flow[pos]);
      setButtonsEnabled(true);
    } catch (e) {
      console.error(e);
      $("identity-error").textContent = "Could not start the test. Please try again.";
    }
  });

  // Best-effort: tell server the tab is closing (so it can shut down if desired)
  window.addEventListener("beforeunload", () => {
    try {
      navigator.sendBeacon("/api/unload", JSON.stringify({ session_id: sessionId || "" }));
    } catch (e) {}
  });

  show("title");
</script>
</body>
</html>
"""

# -------------------------
# HTTP Handler
# -------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "MentalRotationsHTTP/1.1"

    def _send(self, code: int, data: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, code: int, obj):
        data = json.dumps(obj).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path.startswith("/imgs/"):
            name = unquote(path[len("/imgs/"):])
            if ".." in name or name.startswith("/") or name.startswith("\\"):
                self._send(HTTPStatus.BAD_REQUEST, b"Bad path", "text/plain; charset=utf-8")
                return
            img_path = IMGS_DIR / name
            if not img_path.exists() or not img_path.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")
                return
            data = img_path.read_bytes()
            self._send(HTTPStatus.OK, data, "image/png")
            return

        self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Special-case: sendBeacon may send text/plain; parse best-effort
        try:
            payload = self._read_json()
        except Exception:
            payload = {}

        if path == "/api/start":
            fname = str(payload.get("fname", "")).strip()
            lname = str(payload.get("lname", "")).strip()
            age = str(payload.get("age", "")).strip()
            if not fname or not lname or not age:
                self._send(HTTPStatus.BAD_REQUEST, b"Missing identity fields", "text/plain; charset=utf-8")
                return

            stimuli64 = list_stimuli()
            stimuli_sorted = sorted(stimuli64)

            real_trials = build_real_trial_sequence(stimuli64)  # 192
            flow = insert_attention_trials(real_trials, stimuli64, ATTN_N)  # 192 + 5

            sid = uuid.uuid4().hex
            with SESSIONS_LOCK:
                SESSIONS[sid] = {
                    "fname": fname,
                    "lname": lname,
                    "age": age,
                    "stimuli_sorted": stimuli_sorted,
                    "real_trials": real_trials,
                    "flow": flow,
                    "pos": 0,
                    "ident_times": {},
                    "ident_corr": {},
                    "order_times": [None] * 192,
                    "order_corr": [None] * 192,
                    "attn_correct": 0,
                    "finished": False,
                }

            # send full flow to client
            self._send_json(HTTPStatus.OK, {"session_id": sid, "flow": flow})
            return

        if path == "/api/record":
            sid = str(payload.get("session_id", "")).strip()
            kind = str(payload.get("kind", "")).strip().lower()
            img = str(payload.get("img", "")).strip()
            ans = str(payload.get("answer", "")).strip().lower()
            time_ms = int(payload.get("time_ms", -1))

            if not sid or not img or time_ms < 0:
                self._send(HTTPStatus.BAD_REQUEST, b"Bad record payload", "text/plain; charset=utf-8")
                return
            if kind not in ("real", "attn"):
                self._send(HTTPStatus.BAD_REQUEST, b"Bad kind", "text/plain; charset=utf-8")
                return

            with SESSIONS_LOCK:
                sess = SESSIONS.get(sid)
                if not sess or sess.get("finished"):
                    self._send(HTTPStatus.BAD_REQUEST, b"Invalid session", "text/plain; charset=utf-8")
                    return

                if kind == "attn":
                    # Attention check correctness: ALWAYS "same"
                    if ans == "same":
                        sess["attn_correct"] += 1
                    self._send_json(HTTPStatus.OK, {"ok": True})
                    return

                # real trial
                order_index = payload.get("order_index", None)
                cycle = payload.get("cycle", None)
                if order_index is None or cycle is None:
                    self._send(HTTPStatus.BAD_REQUEST, b"Missing order_index/cycle for real trial", "text/plain; charset=utf-8")
                    return

                order_index = int(order_index)
                cycle = int(cycle)

                if order_index < 1 or order_index > 192 or cycle not in (1, 2, 3):
                    self._send(HTTPStatus.BAD_REQUEST, b"Bad order_index/cycle", "text/plain; charset=utf-8")
                    return

                corr = 1 if ans == correct_answer(img) else 0

                sess["ident_times"][(cycle, img)] = time_ms
                sess["ident_corr"][(cycle, img)] = corr

                sess["order_times"][order_index - 1] = time_ms
                sess["order_corr"][order_index - 1] = corr
                sess["pos"] = max(sess["pos"], order_index)

            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        if path == "/api/finish":
            sid = str(payload.get("session_id", "")).strip()
            if not sid:
                self._send(HTTPStatus.BAD_REQUEST, b"Missing session_id", "text/plain; charset=utf-8")
                return

            with SESSIONS_LOCK:
                sess = SESSIONS.get(sid)
                if not sess or sess.get("finished"):
                    self._send_json(HTTPStatus.OK, {"ok": True, "already": True})
                    return

                # Require all 192 real trials recorded
                if any(v is None for v in sess["order_times"]) or any(v is None for v in sess["order_corr"]):
                    self._send(HTTPStatus.BAD_REQUEST, b"Not all 192 trials recorded", "text/plain; charset=utf-8")
                    return

                stimuli_sorted = sess["stimuli_sorted"]
                header = build_csv_header(stimuli_sorted)
                ensure_csv_exists(header)

                row = {}
                row["fname"] = sess["fname"]
                row["lname"] = sess["lname"]
                row["age"] = sess["age"]

                for cycle in (1, 2, 3):
                    for img in stimuli_sorted:
                        row[f"c{cycle}_{img}_time_ms"] = sess["ident_times"].get((cycle, img), "")
                for cycle in (1, 2, 3):
                    for img in stimuli_sorted:
                        row[f"c{cycle}_{img}_correct"] = sess["ident_corr"].get((cycle, img), "")

                for i in range(1, 193):
                    row[f"order_{i}_time_ms"] = sess["order_times"][i - 1]
                for i in range(1, 193):
                    row[f"order_{i}_correct"] = sess["order_corr"][i - 1]

                row["attention"] = int(sess.get("attn_correct", 0))

                with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([row.get(col, "") for col in header])

                sess["finished"] = True

            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        if path == "/api/unload":
            # Best-effort: if the browser tab closes, optionally shutdown server when no active sessions.
            # We won't force-shutdown here because multiple participants or tabs may exist.
            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")

    def log_message(self, format, *args):
        return

# -------------------------
# Server creation
# -------------------------
def make_server(host: str, port: int) -> ThreadingHTTPServer:
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), Handler)

def find_free_port(host: str, start_port: int, tries: int = 25) -> int:
    """
    Try start_port, start_port+1, ... for a bindable port.
    """
    import socket
    for p in range(start_port, start_port + tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, p))
            return p
        except OSError:
            continue
        finally:
            try:
                s.close()
            except Exception:
                pass
    raise OSError(f"No free port found in range {start_port}..{start_port+tries-1}")

# -------------------------
# Main
# -------------------------
def main():
    # Validate stimuli early
    stimuli = list_stimuli()
    if len(stimuli) != 64:
        print(f"[warn] Found {len(stimuli)} .png files in imgs/ (expected 64). The app will still run.")

    # Port selection
    port = PORT
    if AUTO_PORT_FALLBACK:
        try:
            port = find_free_port(HOST, PORT, tries=50)
        except Exception:
            port = PORT

    print(f"Serving on http://{HOST}:{port}")
    print(f"Images directory: {IMGS_DIR.resolve()}")
    print(f"CSV output: {CSV_PATH.resolve()}")

    httpd = make_server(HOST, port)

    url = f"http://{HOST}:{port}/"
    try:
        webbrowser.open(url, new=1, autoraise=True)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()