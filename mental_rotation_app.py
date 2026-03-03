#!/usr/bin/env python3
"""
Mental Rotation Task - local webapp + CSV logger

Folder layout:
  mental_rotation_app.py
  imgs/       (contains stimuli .png)
  prac_imgs/  (contains practice .png)

Output:
  data.csv   (created/appended)

CHANGES REQUESTED
1) Identity:
   - Collect ONLY a free-form participant "id" (no restrictions).
   - CSV stores id instead of fname/lname/age.

2) Buttons / scoring labels:
   - For *reference* (two-character) stimuli: buttons are "Same" vs "Mirrored".
   - For *centered* (one-character) stimuli: buttons are "Normal" vs "Mirrored".
   - Attention checks remain 5 trials and require "Same" (buttons shown: Same/Mirrored).

3) Practice round (not timed/scored, not recorded):
   - From prac_imgs/, show in order:
       (1) centered_normal
       (2) centered_mirrored
       (3) reference_normal
       (4) reference_mirrored
     Angles can be random (picked from filenames).
   - Practice title text:
       "This text is Normal" (centered normal)
       "This text is Same"   (reference normal)
       "This text is Mirrored" (any mirrored case)
   - After the 4 practice trials, ask the participant to start the actual test.
   - Flow: begin -> id collection -> practice -> start real test (timing + accuracy) with 5 attention checks.

4) Extra CSV summary columns (Block A):
   - For letters {R,G} and each angle present:
       letter_angle_mean_time
       letter_angle_score
       reference_letter_angle_mean_time
       reference_letter_angle_score
       centered_letter_angle_mean_time
       centered_letter_angle_score
     *These are computed across the 3 cycles using ONE canonical centered image and ONE canonical reference image per cycle
      for that (letter, angle). Canonical choice prefers *_normal_* if available (per type), else any match.*
     Scores sum correctness across cycles (so letter_angle_score is in [0,6]).

   - Also aggregate over all angles for each letter:
       letter_mean_time, letter_score
       reference_letter_mean_time, reference_letter_score
       centered_letter_mean_time, centered_letter_score

   - For every mean_time column above, also write a scored-only mean:
       <name>_mean_time_scored
     This averages time ONLY across correct trials (binary mask). If none correct, it is left blank.

Notes:
- The original 192 real trials structure is preserved (3 cycles of 64 stimuli).
- Attention checks are extra trials (not written into the 192 trial columns), summarized in "attention".
"""

from __future__ import annotations

import csv
import json
import random
import re
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
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
PRAC_DIR = Path("prac_imgs")
CSV_PATH = Path("data.csv")

ATTN_N = 5  # number of attention checks inserted (extra trials)
AUTO_PORT_FALLBACK = True  # if PORT is busy, try next ports

# -------------------------
# Helpers
# -------------------------
def list_pngs(folder: Path) -> list[str]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder.resolve()}")
    pngs = sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    if len(pngs) == 0:
        raise FileNotFoundError(f"No .png files found in: {folder.resolve()}")
    return pngs

def list_stimuli() -> list[str]:
    return list_pngs(IMGS_DIR)

def list_practice() -> list[str]:
    return list_pngs(PRAC_DIR)

def is_reference(imgname: str) -> bool:
    return "reference" in imgname.lower()

def is_centered(imgname: str) -> bool:
    return "centered" in imgname.lower()

def is_mirrored(imgname: str) -> bool:
    return "mirrored" in imgname.lower()

def is_normal(imgname: str) -> bool:
    return "normal" in imgname.lower()

def correct_answer(imgname: str) -> str:
    """
    Answer vocabulary:
      - reference images: "same" or "mirrored"
      - centered images:  "normal" or "mirrored"
    """
    if is_reference(imgname):
        return "mirrored" if is_mirrored(imgname) else "same"
    else:
        return "mirrored" if is_mirrored(imgname) else "normal"

# Expected-ish naming pattern examples:
#   R_centered_mirrored_270.png
#   G_reference_normal_090.png
# We'll be permissive.
_STIM_RE = re.compile(
    r"^(?P<letter>[A-Za-z]+)_(?P<kind>centered|reference)_(?P<state>normal|mirrored)_(?P<angle>\d{1,3})\.png$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class StimInfo:
    letter: str
    kind: str      # "centered" or "reference"
    state: str     # "normal" or "mirrored"
    angle: str     # keep as zero-padded-ish string if present

def parse_stim(img: str) -> StimInfo | None:
    m = _STIM_RE.match(img)
    if not m:
        return None
    letter = m.group("letter").upper()
    kind = m.group("kind").lower()
    state = m.group("state").lower()
    angle = m.group("angle")
    # normalize angle to 3 digits if numeric
    try:
        angle_i = int(angle)
        angle = f"{angle_i:03d}"
    except Exception:
        pass
    return StimInfo(letter=letter, kind=kind, state=state, angle=angle)

def build_real_trial_sequence(stimuli64: list[str]) -> list[dict]:
    """
    3 cycles. Each cycle is a random permutation of the 64 stimuli.
    Returns list of 192 REAL trial dicts:
      {
        "kind": "real",
        "dir": "imgs",
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
            seq.append({"kind": "real", "dir": "imgs", "order_index": order, "cycle": cycle, "img": img})
            order += 1
    return seq

def insert_attention_trials(real_trials: list[dict], stimuli64: list[str], n: int) -> list[dict]:
    """
    Insert n attention checks randomly throughout the full flow.
    Attention checks are extra and do NOT change order_index / cycle for real trials.

    Attention trial dict:
      {
        "kind": "attn",
        "dir": "imgs",
        "img": <random image from 64>,
        "attn_id": 1..n
      }
    """
    if n <= 0:
        return real_trials[:]

    flow = real_trials[:]  # length 192 initially
    positions = sorted(random.sample(range(len(flow) + 1), k=n), reverse=True)

    for k, pos in enumerate(positions, start=1):
        img = random.choice(stimuli64)
        flow.insert(pos, {"kind": "attn", "dir": "imgs", "img": img, "attn_id": k})

    return flow

def pick_practice_trials(prac_pngs: list[str]) -> list[dict]:
    """
    Choose exactly 4 practice images from prac_imgs in this order:
      1) centered_normal
      2) centered_mirrored
      3) reference_normal
      4) reference_mirrored
    Angle can be random (picked via filename match).
    """
    def pick(where_kind: str, where_state: str) -> str:
        candidates = []
        for p in prac_pngs:
            low = p.lower()
            if where_kind in low and where_state in low:
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(
                f"Could not find practice image matching '{where_kind}_{where_state}' in {PRAC_DIR.resolve()}"
            )
        return random.choice(candidates)

    img1 = pick("centered", "normal")
    img2 = pick("centered", "mirrored")
    img3 = pick("reference", "normal")
    img4 = pick("reference", "mirrored")

    return [
        {"kind": "prac", "dir": "prac_imgs", "img": img1},
        {"kind": "prac", "dir": "prac_imgs", "img": img2},
        {"kind": "prac", "dir": "prac_imgs", "img": img3},
        {"kind": "prac", "dir": "prac_imgs", "img": img4},
    ]

def build_csv_header(stimuli64_sorted: list[str]) -> list[str]:
    """
    Base columns (keeps the original layout except identity fields):
      1) id
      2..(1+192): identifiable times (192) in ms
      next 192: identifiable performance (192) binary
      next 192: order times (192) in ms
      next 192: order performance (192) in binary
      + attention
      + Block A summary columns (dynamic based on angles found for R/G)
    """
    header: list[str] = ["id"]

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

    # 6) attention score
    header.append("attention")

    # 7) Block A summary columns
    header.extend(build_block_a_headers(stimuli64_sorted))

    return header

def ensure_csv_exists(header: list[str]) -> None:
    if CSV_PATH.exists():
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

# -------------------------
# Block A aggregation
# -------------------------
def build_canonical_map(stimuli_sorted: list[str]) -> dict[tuple[str, str, str], str]:
    """
    Map (letter, angle, kind) -> canonical filename.
    Canonical preference: normal > anything else.
    kind in {"centered","reference"}.
    """
    bucket: dict[tuple[str, str, str], list[str]] = {}
    for img in stimuli_sorted:
        info = parse_stim(img)
        if not info:
            continue
        if info.letter not in ("R", "G"):
            continue
        if info.kind not in ("centered", "reference"):
            continue
        key = (info.letter, info.angle, info.kind)
        bucket.setdefault(key, []).append(img)

    canon: dict[tuple[str, str, str], str] = {}
    for key, imgs in bucket.items():
        # prefer *_normal_*
        normals = [x for x in imgs if is_normal(x)]
        chosen = sorted(normals)[0] if normals else sorted(imgs)[0]
        canon[key] = chosen
    return canon

def angles_for_letter(canon: dict[tuple[str, str, str], str], letter: str) -> list[str]:
    angs = sorted({angle for (L, angle, kind) in canon.keys() if L == letter and kind in ("centered", "reference")})
    return angs

def build_block_a_headers(stimuli_sorted: list[str]) -> list[str]:
    canon = build_canonical_map(stimuli_sorted)
    cols: list[str] = []

    # letter+angle columns
    for letter in ("R", "G"):
        for angle in angles_for_letter(canon, letter):
            base = f"{letter}_{angle}"
            cols.append(f"{base}_mean_time")
            cols.append(f"{base}_mean_time_scored")
            cols.append(f"{base}_score")

            cols.append(f"reference_{base}_mean_time")
            cols.append(f"reference_{base}_mean_time_scored")
            cols.append(f"reference_{base}_score")

            cols.append(f"centered_{base}_mean_time")
            cols.append(f"centered_{base}_mean_time_scored")
            cols.append(f"centered_{base}_score")

    # letter (no angle) columns
    for letter in ("R", "G"):
        cols.append(f"{letter}_mean_time")
        cols.append(f"{letter}_mean_time_scored")
        cols.append(f"{letter}_score")

        cols.append(f"reference_{letter}_mean_time")
        cols.append(f"reference_{letter}_mean_time_scored")
        cols.append(f"reference_{letter}_score")

        cols.append(f"centered_{letter}_mean_time")
        cols.append(f"centered_{letter}_mean_time_scored")
        cols.append(f"centered_{letter}_score")

    return cols

def mean_or_blank(values: list[int]) -> str:
    if not values:
        return ""
    return f"{(sum(values) / len(values)):.6g}"

def compute_block_a(sess: dict) -> dict[str, str | int]:
    """
    Computes all Block A columns into a dict.
    Uses canonical images per (letter, angle, kind) and cycles 1..3.

    Important: This is designed so that per (letter, angle):
      - centered contributes 3 trials (one per cycle)
      - reference contributes 3 trials (one per cycle)
      - combined contributes 6 trials -> score in [0,6]
    """
    stimuli_sorted: list[str] = sess["stimuli_sorted"]
    canon = build_canonical_map(stimuli_sorted)

    ident_times: dict[tuple[int, str], int] = sess["ident_times"]
    ident_corr: dict[tuple[int, str], int] = sess["ident_corr"]

    out: dict[str, str | int] = {}

    def collect(imgs: list[str]) -> tuple[list[int], list[int]]:
        times: list[int] = []
        corrs: list[int] = []
        for cycle in (1, 2, 3):
            for img in imgs:
                t = ident_times.get((cycle, img), None)
                c = ident_corr.get((cycle, img), None)
                if t is None or c is None:
                    continue
                times.append(int(t))
                corrs.append(int(c))
        return times, corrs

    def write_mean_score(prefix: str, times: list[int], corrs: list[int]) -> None:
        out[f"{prefix}_mean_time"] = mean_or_blank(times)
        # scored mean time: only correct trials
        scored_times = [t for (t, c) in zip(times, corrs) if c == 1]
        out[f"{prefix}_mean_time_scored"] = mean_or_blank(scored_times)
        out[f"{prefix}_score"] = int(sum(corrs)) if corrs else 0

    # letter+angle
    for letter in ("R", "G"):
        angs = angles_for_letter(canon, letter)
        for angle in angs:
            ref_img = canon.get((letter, angle, "reference"))
            cen_img = canon.get((letter, angle, "centered"))

            ref_imgs = [ref_img] if ref_img else []
            cen_imgs = [cen_img] if cen_img else []
            both_imgs = ref_imgs + cen_imgs

            # combined
            times, corrs = collect(both_imgs)
            write_mean_score(f"{letter}_{angle}", times, corrs)

            # reference only
            times, corrs = collect(ref_imgs)
            write_mean_score(f"reference_{letter}_{angle}", times, corrs)

            # centered only
            times, corrs = collect(cen_imgs)
            write_mean_score(f"centered_{letter}_{angle}", times, corrs)

    # letter (no angle): aggregate across all angles for that letter
    for letter in ("R", "G"):
        ref_imgs: list[str] = []
        cen_imgs: list[str] = []
        for angle in angles_for_letter(canon, letter):
            ri = canon.get((letter, angle, "reference"))
            ci = canon.get((letter, angle, "centered"))
            if ri:
                ref_imgs.append(ri)
            if ci:
                cen_imgs.append(ci)

        both_imgs = ref_imgs + cen_imgs

        times, corrs = collect(both_imgs)
        write_mean_score(f"{letter}", times, corrs)

        times, corrs = collect(ref_imgs)
        write_mean_score(f"reference_{letter}", times, corrs)

        times, corrs = collect(cen_imgs)
        write_mean_score(f"centered_{letter}", times, corrs)

    return out

# -------------------------
# In-memory session store
# -------------------------
SESSIONS_LOCK = threading.Lock()
SESSIONS: dict[str, dict] = {}
# session = {
#   "id": str,
#   "stimuli_sorted": [64],
#   "real_trials": [192 dicts],
#   "flow": [192 + ATTN_N dicts],
#   "practice": [4 dicts],
#   "pos": int,
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
  <title>Mental Rotation Task</title>
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
    .row { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
    .field { flex: 1 1 280px; max-width: 420px; display: flex; flex-direction: column; gap: 6px; }
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
      min-width: 160px;
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
      <h1 class="center">Mental Rotation Task</h1>
      <p class="center">Click start to begin.</p>
      <div class="btnrow">
        <button id="btn-start">Start</button>
      </div>
    </div>

    <div class="card hidden" id="screen-identity">
      <h2 class="center">Participant Information</h2>
      <div class="row">
        <div class="field">
          <label for="pid">Participant ID</label>
          <input id="pid" autocomplete="off" />
        </div>
      </div>
      <div class="btnrow">
        <button class="secondary" id="btn-back">Back</button>
        <button id="btn-begin-practice">Continue</button>
      </div>
      <p id="identity-error" class="center" style="color:#b00020; margin-top:10px;"></p>
    </div>

    <div class="card hidden" id="screen-ready">
      <h2 class="center">Practice complete</h2>
      <p class="center">When you're ready, start the real task.</p>
      <div class="btnrow">
        <button id="btn-begin-test">Start Task</button>
      </div>
    </div>

    <div class="card hidden" id="screen-test">
      <h2 class="center" id="prompt">Prompt</h2>
      <div class="imgbox">
        <img class="stim" id="stim-img" alt="stimulus" />
      </div>

      <div class="btnrow" id="choices"></div>

      <div class="meta" id="meta-row">
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
    ready: $("screen-ready"),
    test: $("screen-test"),
    done: $("screen-done")
  };

  function show(which) {
    for (const k of Object.keys(screens)) screens[k].classList.add("hidden");
    screens[which].classList.remove("hidden");
  }

  let sessionId = null;

  let practice = [];
  let practicePos = 0;

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

  function imgSrc(item) {
    const dir = item.dir || "imgs";
    if (dir === "prac_imgs") return `/prac_imgs/${encodeURIComponent(item.img)}`;
    return `/imgs/${encodeURIComponent(item.img)}`;
  }

  function clearChoices() {
    const choices = $("choices");
    choices.innerHTML = "";
  }

  function setButtonsEnabled(enabled) {
    const btns = $("choices").querySelectorAll("button");
    btns.forEach(b => b.disabled = !enabled);
  }

  function addChoiceButton(label, value, onClick) {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", onClick);
    $("choices").appendChild(b);
  }

  function practiceTitleText(item) {
    const name = (item.img || "").toLowerCase();
    const mirrored = name.includes("mirrored");
    const reference = name.includes("reference");
    if (mirrored) return "This text is Mirrored";
    if (reference) return "This text is Same";
    return "This text is Normal";
  }

  function renderPractice(item) {
    $("stim-img").src = imgSrc(item);
    $("prompt").textContent = practiceTitleText(item);

    // Hide trial/cycle row during practice
    $("meta-row").style.display = "none";

    clearChoices();

    const isReference = (item.img || "").toLowerCase().includes("reference");
    if (isReference) {
      addChoiceButton("Same", "same", () => advancePractice());
      addChoiceButton("Mirrored", "mirrored", () => advancePractice());
    } else {
      addChoiceButton("Normal", "normal", () => advancePractice());
      addChoiceButton("Mirrored", "mirrored", () => advancePractice());
    }

    // practice: no timing / no recording
  }

  function advancePractice() {
    practicePos += 1;
    if (practicePos >= practice.length) {
      show("ready");
      return;
    }
    renderPractice(practice[practicePos]);
  }

  function renderTrial(item) {
    // item.kind is "real" or "attn"
    $("meta-row").style.display = ""; // show during real test

    if (item.kind === "real") {
      $("trial-idx").textContent = String(item.order_index);
      $("cycle-idx").textContent = String(item.cycle);
    } else {
      // attention: keep displays as-is
      $("trial-idx").textContent = $("trial-idx").textContent || "1";
      $("cycle-idx").textContent = $("cycle-idx").textContent || "1";
    }

    $("stim-img").src = imgSrc(item);
    clearChoices();

    if (item.kind === "attn") {
      $("prompt").textContent = "Click Option Same";
      addChoiceButton("Same", "same", () => submitAnswer("same"));
      addChoiceButton("Mirrored", "mirrored", () => submitAnswer("mirrored"));
    } else {
      const isReference = (item.img || "").toLowerCase().includes("reference");
      if (isReference) {
        $("prompt").textContent = "Are these the same letter?";
        addChoiceButton("Same", "same", () => submitAnswer("same"));
        addChoiceButton("Mirrored", "mirrored", () => submitAnswer("mirrored"));
      } else {
        $("prompt").textContent = "Is this letter normal or mirrored?";
        addChoiceButton("Normal", "normal", () => submitAnswer("normal"));
        addChoiceButton("Mirrored", "mirrored", () => submitAnswer("mirrored"));
      }
    }

    t0 = performance.now();
    setButtonsEnabled(true);
  }

  async function submitAnswer(value) {
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
  }

  $("btn-start").addEventListener("click", () => {
    show("identity");
  });

  $("btn-back").addEventListener("click", () => {
    show("title");
  });

  $("btn-begin-practice").addEventListener("click", async () => {
    $("identity-error").textContent = "";
    const pid = $("pid").value.trim();
    if (!pid) {
      $("identity-error").textContent = "Please enter your Participant ID.";
      return;
    }

    try {
      const resp = await api("/api/start", "POST", { id: pid });
      sessionId = resp.session_id;

      practice = resp.practice || [];
      practicePos = 0;

      flow = resp.flow || [];
      pos = 0;

      if (!practice.length) {
        // If practice missing, go straight to ready -> test
        show("ready");
      } else {
        show("test");
        renderPractice(practice[practicePos]);
      }
    } catch (e) {
      console.error(e);
      $("identity-error").textContent = "Could not start. Please try again.";
    }
  });

  $("btn-begin-test").addEventListener("click", () => {
    if (!flow.length) {
      show("done");
      return;
    }
    show("test");
    renderTrial(flow[pos]);
  });

  // Best-effort: tell server the tab is closing
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
            self._send(HTTPStatus.OK, img_path.read_bytes(), "image/png")
            return

        if path.startswith("/prac_imgs/"):
            name = unquote(path[len("/prac_imgs/"):])
            if ".." in name or name.startswith("/") or name.startswith("\\"):
                self._send(HTTPStatus.BAD_REQUEST, b"Bad path", "text/plain; charset=utf-8")
                return
            img_path = PRAC_DIR / name
            if not img_path.exists() or not img_path.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")
                return
            self._send(HTTPStatus.OK, img_path.read_bytes(), "image/png")
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
            pid = str(payload.get("id", "")).strip()
            if not pid:
                self._send(HTTPStatus.BAD_REQUEST, b"Missing id", "text/plain; charset=utf-8")
                return

            stimuli64 = list_stimuli()
            stimuli_sorted = sorted(stimuli64)

            # practice selection
            prac_pngs = list_practice()
            practice = pick_practice_trials(prac_pngs)

            # real test flow
            real_trials = build_real_trial_sequence(stimuli64)  # 192
            flow = insert_attention_trials(real_trials, stimuli64, ATTN_N)  # 192 + 5

            sid = uuid.uuid4().hex
            with SESSIONS_LOCK:
                SESSIONS[sid] = {
                    "id": pid,
                    "stimuli_sorted": stimuli_sorted,
                    "real_trials": real_trials,
                    "flow": flow,
                    "practice": practice,
                    "pos": 0,
                    "ident_times": {},
                    "ident_corr": {},
                    "order_times": [None] * 192,
                    "order_corr": [None] * 192,
                    "attn_correct": 0,
                    "finished": False,
                }

            self._send_json(HTTPStatus.OK, {"session_id": sid, "practice": practice, "flow": flow})
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

                row: dict[str, object] = {}
                row["id"] = sess["id"]

                # identifiable times/correct
                for cycle in (1, 2, 3):
                    for img in stimuli_sorted:
                        row[f"c{cycle}_{img}_time_ms"] = sess["ident_times"].get((cycle, img), "")
                for cycle in (1, 2, 3):
                    for img in stimuli_sorted:
                        row[f"c{cycle}_{img}_correct"] = sess["ident_corr"].get((cycle, img), "")

                # order times/correct
                for i in range(1, 193):
                    row[f"order_{i}_time_ms"] = sess["order_times"][i - 1]
                for i in range(1, 193):
                    row[f"order_{i}_correct"] = sess["order_corr"][i - 1]

                # attention summary
                row["attention"] = int(sess.get("attn_correct", 0))

                # Block A summaries
                block_a = compute_block_a(sess)
                row.update(block_a)

                with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([row.get(col, "") for col in header])

                sess["finished"] = True

            self._send_json(HTTPStatus.OK, {"ok": True})
            return

        if path == "/api/unload":
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
    # Validate folders early
    stimuli = list_stimuli()
    if len(stimuli) != 64:
        print(f"[warn] Found {len(stimuli)} .png files in imgs/ (expected 64). The app will still run.")

    _ = list_practice()  # raises if missing/empty

    # Port selection
    port = PORT
    if AUTO_PORT_FALLBACK:
        try:
            port = find_free_port(HOST, PORT, tries=50)
        except Exception:
            port = PORT

    print(f"Serving on http://{HOST}:{port}")
    print(f"Images directory: {IMGS_DIR.resolve()}")
    print(f"Practice directory: {PRAC_DIR.resolve()}")
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