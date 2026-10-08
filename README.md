# Eyemouse

**Control your Windows cursor with your eyes, using nothing but a webcam.**

Eyemouse tracks your pupils with MediaPipe Face Mesh (with Iris refinement), maps
your gaze onto the screen through a 9-point calibration, and smooths it into a
usable pointer. Clicks come from **touching your temple** with a finger, or from a
**wink** — no hardware, no IR illuminator, no glasses.

> **Set expectations first.** A webcam eye tracker is genuinely usable for large
> targets, accessibility work, hands-busy demos, and slow deliberate clicking. It
> is *not* a mouse replacement for fast text editing or gaming. Expect roughly
> 20–40 px of accuracy after a good calibration — about the size of a toolbar
> button. The **dwell click** and **sniper mode** exist precisely because of this,
> and most users end up relying on both.

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [First run](#first-run)
- [Calibration walkthrough](#calibration-walkthrough)
- [Controls](#controls)
- [On-screen overlay](#on-screen-overlay)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting: lighting](#troubleshooting-lighting)
- [Troubleshooting: calibration accuracy](#troubleshooting-calibration-accuracy)
- [Troubleshooting: everything else](#troubleshooting-everything-else)
- [Architecture](#architecture)
- [Testing without a webcam](#testing-without-a-webcam)
- [Known limitations](#known-limitations)

---

## Requirements

| | |
|---|---|
| **OS** | Windows 10 or 11 (x64 or arm64) |
| **Python** | 3.10 – 3.12 |
| **Camera** | Any webcam, 640×480 or better, mounted so your eyes fill a good part of the frame |
| **Libraries** | `mediapipe`, `opencv-contrib-python`, `numpy`, `pyautogui`, `pynput` |
| **Models** | Two `.task` bundles (~7 MB total), downloaded by a script |

Linux and macOS will import and run, but cursor control is limited or absent —
the DPI-awareness and screen-metrics code is Windows-specific. Use `--dry-run`
to experiment elsewhere.

---

## Install

### The easy way

Double-click **`run.bat`**. It creates `.venv`, installs the dependencies,
downloads the model bundles, and starts the app. On later runs it skips straight
to launching.

### Manually

```bat
cd Eyemouse
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools\download_models.py
```

Then verify everything before you trust it:

```bat
python main.py --check
```

That prints your Python/OpenCV/MediaPipe versions, whether the MediaPipe Tasks
API is available, whether both model bundles are present, the detected screen
size, which mouse backend won, and whether the camera opens. It exits non-zero
and tells you exactly what to fix.

### ⚠️ Do not install `opencv-python`

MediaPipe depends on **`opencv-contrib-python`**, which provides the *same* `cv2`
module. Installing both into one environment makes the second overwrite files
from the first and leaves a mixed install that fails at `import cv2` with
confusing errors. `opencv-contrib-python` is a superset — Eyemouse uses nothing
that differs. If you already installed the plain one:

```bat
pip uninstall -y opencv-python opencv-contrib-python
pip install -r requirements.txt
```

---

## First run

```bat
python main.py
```

1. The camera opens and a calibration screen fills your display.
2. Work through the 9 points (see below).
3. A fit report is printed — **RMS** and **max** error in pixels. RMS under
   ~30 px is good; over ~60 px means something was wrong (see
   [calibration troubleshooting](#troubleshooting-calibration-accuracy)).
4. A preview window appears and the cursor starts following your gaze.

The calibration is saved to `calibration.json` next to the code, so the next
launch skips straight to step 4 unless you press **F8**.

Useful flags:

| Flag | Effect |
|---|---|
| `--check` | Diagnose the environment and exit |
| `--calibrate` | Force a fresh calibration even if one is saved |
| `--no-calibrate` | Skip calibration (the cursor will not move until you calibrate) |
| `--dry-run` | Never touch the real cursor — everything is drawn instead |
| `--no-overlay` | No preview window (lower CPU, but no visual feedback) |
| `--camera N` | Pick a camera index |
| `--config PATH` | Use a different config file |
| `--write-config` | Write a default `config.json` and exit |
| `--frames N` | Stop after N frames (for testing) |
| `--quiet` | Suppress the startup notes |

---

## Calibration walkthrough

Calibration is the difference between a toy and something you can actually use.
Two minutes of care here is worth more than any amount of filter tuning.

**Before you start:**

- Sit where you will actually work, at your normal distance (50–70 cm).
- Get your lighting right first — see [lighting](#troubleshooting-lighting).
- Close other apps that grab the webcam (Teams, Zoom, OBS, browser tabs).

**During the run:**

An orange dot appears, with a white ring around it and a green arc that fills as
samples are collected.

1. **Look at the centre of the dot**, not at the screen generally. Pick a single
   pixel and keep your eyes on it.
2. **Hold still.** Each point collects 30 frames; the first 8 are thrown away to
   let the filters settle and the remaining 22 are kept. Wriggling through all
   of them poisons the fit.
3. **Do not move your head.** Blink normally — blinks are detected and skipped —
   but deliberately moving your head between points teaches the model head
   motion instead of eye motion.
4. Keep going until all 9 points are done. The order is left-to-right,
   top-to-bottom.

**Reading the report:**

| Metric | Good | Action if worse |
|---|---|---|
| `RMS` | < 30 px | Re-run, checking lighting and head stability |
| `max` | < 80 px | Usually one bad point — note which one and re-run |
| per-point spread | even | A single outlier point means you blinked or looked away there |

`--calibrate` re-runs it at any time; **F8** does the same mid-session.

---

## Controls

### Looking

Move your eyes. The cursor follows. `gaze.gain` scales how far your eyes have to
travel to cross the screen — raise it if the cursor feels sluggish, lower it if
it overshoots.

### Clicking

Both mechanisms are on by default (`clicks.gesture: "both"`) and either one
fires. Set it to `"temple"` or `"wink"` to use only one, or `"none"` to rely
purely on dwell.

| Action | Left click | Right click |
|---|---|---|
| **Temple** | Touch your **left** temple with any of index / middle / ring fingertip | Touch your **right** temple |
| **Wink** | Close your **left** eye, keep the right open | Close your **right** eye, keep the left open |
| **Dwell** | Hold your gaze still for 1.4 s | Disabled by default (`clicks.dwell.right_click_ms: 0`) |

Every button has a **500 ms debounce** (`clicks.debounce_ms`), so a shaky temple
touch or a double wink produces one click, not several. Temple touches and winks
are **edge-triggered and latched**: they fire once when you make the gesture and
will not fire again until you release it, however long you hold.

> **Temple touches: which hand?** MediaPipe labels handedness assuming a mirrored
> (selfie) image, and that assumption flips depending on your camera. Rather than
> guess, Eyemouse defaults to `handedness_mode: "temple_side"` — **the side of
> your face you touch decides the button**, not which hand it was. This is
> correct for everyone and immune to the ambiguity. If you specifically want
> "must be my left hand", set `"strict"`, and flip `swap_handedness: true` if it
> comes out backwards.

### Scrolling

The default is **edge dwell**, chosen because it cannot collide with clicking:

- Look at the **top 6%** of the screen and hold for **800 ms** → scroll up.
- Look at the **bottom 6%** and hold 800 ms → scroll down.
- Keep holding and it repeats, accelerating up to 3× the base amount.
- While an edge is active, **dwell-click is suppressed** so you cannot
  accidentally click while scrolling.

The alternative is `scroll.mode: "palm"`: hold an **open palm** to the camera for
400 ms to toggle scroll mode, then move your eyes vertically to scroll. Make a
**fist** to leave, or just wait 6 s. Palm mode suppresses temple clicks, winks
and dwell clicks while it is on — again, by design, so the two can never fight.

### Precision

| Trigger | Effect |
|---|---|
| Touch **both** temples at once | Sniper mode: the cursor moves at ¼ speed. **No click is produced.** |
| Hold **Ctrl** | Same, via `precision.sniper_key` |
| Press **F6** | Latch sniper mode on/off without holding anything |

Sniper mode scales the *delta* rather than remapping absolutely, so when you
release it the cursor resumes exactly where it was with no jump. Smoothing is
also boosted 2.2× while it is on.

### Hotkeys

| Key | Action |
|---|---|
| **F8** | Recalibrate |
| **F7** | Pause / resume tracking |
| **F6** | Toggle sniper latch |
| **F9** | Save the current config (including live slider values) to `config.json` |
| **Esc** or **q** | Quit |

All are configurable under `hotkeys`. Hotkeys are captured through `pynput`'s
global hook when available, so they work even when the preview window is not
focused.

---

## On-screen overlay

The preview window shows your camera feed with:

- the **gaze point** and a short trail of where it has been,
- **temple zones** drawn on your face, so you can see exactly where to touch,
- hand landmarks and fingertips when a hand is detected,
- optional iris landmarks,
- a **HUD** with FPS, calibration state, tracking state, sniper/scroll mode, and
  the frame time of each pipeline stage,
- **flash messages** on every click and scroll event, so you get confirmation
  even when the cursor is elsewhere.

Seven **live sliders** let you tune without editing JSON — they write straight
through to the config, and **F9** persists them:

| Slider | Config key | Range |
|---|---|---|
| `Gain_x100` | `gaze.gain` | 0.20 – 4.00 |
| `Smooth_win` | `gaze.moving_average_window` | 1 – 20 |
| `OneEuro_x1000` | `gaze.one_euro.min_cutoff` | 0.005 – 0.500 |
| `Deadzone_px` | `gaze.deadzone_px` | 0 – 40 |
| `Dwell_ms_x50` | `clicks.dwell.ms` | 200 – 5000 |
| `Temple_r_x100` | `clicks.temple.radius_ratio` | 0.05 – 1.20 |
| `Debounce_ms` | `clicks.debounce_ms` | 0 – 5000 |

`overlay.desktop_dot` (off by default) draws a dot on your *actual* desktop at
the gaze point instead of only in the preview. It is opt-in because it spawns a
transparent always-on-top window, which some setups dislike.

---

## Configuration

Everything lives in `config.json`. Create a fresh copy any time with:

```bat
python main.py --write-config
```

The file is validated on load, not rejected:

- **out-of-range numbers are clamped** to their documented bounds,
- **invalid choices fall back** to the default,
- **wrong types are coerced** (`"0.5"` → `0.5`, `"true"` → `True`),
- **contradictory combinations are repaired** (e.g. a calibration inset that
  would put target dots inside the scroll band is raised),
- and every one of those actions is **printed at startup** so nothing changes
  silently. Add `--quiet` to suppress the notes.

Saving is atomic — written to a temp file then renamed — so a crash mid-write
cannot leave you with a truncated config.

### Keys worth knowing

| Key | Default | Notes |
|---|---|---|
| `gaze.filter` | `ma_one_euro` | One of `none`, `moving_average`, `one_euro`, `kalman`, `ma_one_euro`. Moving average first, then One Euro — the average kills high-frequency tremor, One Euro then adapts to speed so fast motion stays responsive. |
| `gaze.gain` | `1.0` | Eye-to-screen scaling. First thing to tune if the cursor under- or overshoots. |
| `gaze.deadzone_px` | `2.5` | Movement below this is not emitted at all. Compares against the last *emitted* position, so slow deliberate drift still accumulates rather than being stuck. |
| `calibration.points` | `9` | `5` (corners + centre) is faster but noticeably less accurate at the edges; `13` adds quarter points and helps if you work near screen borders. |
| `calibration.model` | `quadratic` | `linear` if you have very few points, `quadratic_head` to additionally fit head pose. |
| `clicks.dwell.ms` | `1400` | Lower for speed, higher to stop accidental clicks. |
| `performance.process_scale` | `1.0` | **The biggest FPS lever.** `0.5` quarters the pixel count fed to MediaPipe. |
| `performance.hands_every_n_frames` | `3` | Hand landmarks are carried forward between runs. Raise to 5 if you rarely use temple clicks. |
| `mediapipe.min_detection_confidence` | `0.5` | Raise to 0.7 if the face is lost in bad light; lower to 0.3 if detection is flaky. |
| `mouse.screen_inset_px` | `6` | Keeps the cursor this far from the physical edge, so the last row of pixels and the taskbar remain reachable but the corner cannot be slammed. |
| `mouse.dpi_aware` | `true` | **Leave this on.** At 125%/150% display scaling a DPI-unaware process sees a virtualised screen size while the cursor moves in physical pixels, so the pointer lands in the wrong place and cannot reach the right or bottom edges. |
| `mouse.backend` | `auto` | Tries `pynput`, then `pyautogui`, then a null backend that moves nothing and logs why. |

Run `python main.py --write-config` and open the file for the full annotated list.

---

## Performance

Target: **≥15 fps on CPU**. The HUD shows live FPS and, with
`performance.show_timing: true`, the per-stage breakdown (face inference, hand
inference, gaze math, gesture logic, overlay drawing) so you can see where your
time actually goes.

If you are below 15 fps, in order of impact:

1. **`performance.process_scale: 0.5`** — by far the largest win. Inference cost
   scales with pixel count; gaze accuracy barely notices at 320×240.
2. **`performance.hands_every_n_frames: 5`** — HandLandmarker is much more
   expensive than FaceLandmarker, and a hand approaching your temple is not a
   fast event.
3. **`overlay.show: false`** or turn off `overlay.draw_hands` /
   `overlay.draw_iris` — drawing landmarks every frame is not free.
4. **Lower the camera resolution** — `camera.width: 480, camera.height: 360`.
5. **Close the preview when you don't need it.** Once calibrated, `--no-overlay`
   is the cheapest way to run.
6. **`mediapipe.delegate: "gpu"`** — attempted and silently falls back to CPU.
   The stock Windows MediaPipe wheels are CPU-only, so this only helps if you
   built MediaPipe with CUDA support.

There is exactly **one `cv2.waitKey` per loop iteration**. Two would double input
latency. The app logic itself — gaze extraction, filtering, gesture arbitration
and the cursor call, excluding all MediaPipe inference and drawing — measures
about **0.25 ms per frame**, so inference and rendering own essentially your
whole budget.

---

## Troubleshooting: lighting

Lighting is the single biggest determinant of whether this works. Iris tracking
needs a clear pupil/iris boundary, and that boundary is contrast.

**Get this right:**

- **Even, diffuse light on your face.** A window in front of you, or a lamp
  bounced off a wall/ceiling, is ideal. Soft light means soft shadows, which
  means the landmark model sees your eye sockets as they actually are.
- **The light should be in front of you, roughly at eye level.**
- **Brightness on your screen helps**, since it lights your face from the front.

**Avoid, in rough order of how badly they hurt:**

| Problem | Why it breaks tracking | Fix |
|---|---|---|
| **Backlight** (window/behind you) | Your face becomes a silhouette; pupils vanish into shadow. This is the #1 cause of "face lost". | Close the blinds, or turn your desk around. Never put a bright light behind you. |
| **Hard side light** | One half of your face blows out, the other falls into deep shadow. The iris on the dark side is unusable and gaze becomes asymmetric. | Diffuse it — bounce it off a wall, or add a second light on the dark side. |
| **Overhead-only lighting** | Puts your eye sockets in shadow. | Add a desk lamp at face height in front of you. |
| **Complete darkness, screen only** | The screen lights the front of your face but the eyes sit in shadow, and pupils dilate so much the iris boundary disappears. | Turn on a dim room light. |
| **Flickering light / cheap LEDs** | Rolling shutter banding makes the iris position jitter frame to frame. | Replace the bulb, or move out of its direct beam. |
| **Glasses reflections** | A bright specular blob lands on the lens, exactly where the pupil is. | Tilt the light source off-axis, or tilt your screen; anti-reflective coating helps a lot. |

**Symptom-specific fixes:**

- **"face lost" / cursor frozen** — you are backlit or too dim. Also check
  `mediapipe.min_detection_confidence` (lower it to 0.3) and `gaze.min_face_width`
  (your face may be too small in frame — move closer).
- **Gaze drifts up/down over time** — your eyes are adapting to changing light.
  Stabilise the lighting and recalibrate.
- **Only one eye tracks** — usually hard side light. Check the overlay: if one
  eye's landmarks are jittery, that side is under-lit. Gaze averaging already
  ignores eyes below `EAR 0.145`, so a genuinely bad eye is dropped rather than
  dragging the estimate.
- **Glasses fogging or smudges** — clean the lenses. A smudge in front of the
  pupil is indistinguishable from a pupil.

---

## Troubleshooting: calibration accuracy

**The cursor is consistently offset in one direction**

You are looking *near* the dot rather than *at* it, or you moved your head
during the run. Re-run and fix your gaze on one pixel. If it is still offset
after a careful run, nudge `gaze.offset_x` / `gaze.offset_y`, or recalibrate —
do not fight it with gain.

**The cursor is fine in the middle but wrong at the edges**

That is a nonlinearity the quadratic model did not capture. Try
`calibration.points: 13`, and make sure `calibration.inset_ratio` is not too
large — if the outermost target dots are near the screen border you may not have
been able to look at them without moving your head. Lower it to `0.08`.

**Accuracy is fine but the cursor jitters**

Increase `gaze.moving_average_window` (4 → 8) and raise `gaze.deadzone_px`
(2.5 → 5). Both cost a little responsiveness. If it jitters only when you hold
still, it is sensor noise; if it jitters when you move, your FPS is too low and
the filter is being starved.

**The cursor lags noticeably behind your eyes**

Too much smoothing. Lower `moving_average_window` to 2, or raise
`gaze.one_euro.beta` (0.012 → 0.05) — beta is the speed-responsiveness term, and
raising it lets the filter react to fast motion while still damping tremor.

**The cursor overshoots or undershoots**

That is `gaze.gain`, not calibration. Overshoot → lower it; undershoot → raise
it. Adjust with the `Gain_x100` slider live.

**Accuracy degrades when you lean back / change distance**

Webcam gaze estimation is relative to the head, and head pose is only partially
compensated. Recalibrate at the distance you will actually sit, and try
`calibration.model: "quadratic_head"`.

**A single point in the report has a huge error**

You blinked, looked away, or the face was lost on that point. The report lists
per-point error — find which one, and re-run. `calibration.discard_first`
(default 8) already throws away the settling samples; if your camera is slow to
expose, raise it to 15.

**Calibration will not start / immediately finishes**

Check the startup notes. If your face is not detected the session cannot collect
samples — fix lighting first, then recalibrate with **F8**.

---

## Troubleshooting: everything else

### `python main.py --check` first

For almost every problem below, run `--check`. It reports the actual failure
rather than making you guess.

### The cursor does not move at all

- The startup notes will say
  `falling back to NullMouse: the cursor will NOT move`, with the reason for each
  backend that failed. Read them.
- Did you pass `--dry-run`? That deliberately moves nothing.
- Are you calibrated? Without a calibration model the app reports
  `not calibrated` and refuses to move the cursor. Press **F8**.
- Is tracking paused? **F7** toggles it; the HUD shows the state.
- Is your face being detected? Watch the preview — no landmarks means no gaze.

### The cursor moves but clicks do not happen

- Check `clicks.gesture` is not `"none"`.
- For temple clicks: the overlay draws the temple zones. Your fingertip must
  enter that circle and stay for `hold_frames` (2) frames. Increase
  `clicks.temple.radius_ratio` if you cannot hit it.
- For winks: one eye must close below `EAR 0.20` **and** be under 60% of the
  open eye's EAR, **and** the open eye must be at least `0.22`. If you have
  naturally narrow eyes, lower `clicks.wink.open_eye_min` to 0.15.
- Both are debounced at 500 ms — rapid repeated attempts will be swallowed.

### I get clicks I did not intend

- **Rubbing or touching your eye** fires a wink on its rising edge. This is
  unavoidable with a wink-based interface; use `"temple"` only.
- **Resting your hand on your temple** fires once, then stays latched until you
  move away. That is intentional — it does not auto-repeat every 500 ms.
- **Dwell clicks while reading.** Raise `clicks.dwell.ms` to 2500, or set
  `clicks.dwell.enabled: false`.
- Note that pressing **both** temples is sniper mode and produces *no* click, by
  design.

### `import cv2` fails, or `module 'cv2' has no attribute ...`

You have both `opencv-python` and `opencv-contrib-python`. See
[the warning above](#-do-not-install-opencv-python).

### `TrackerSetupError` about missing models

```
MediaPipe model bundles not found. Run: python tools/download_models.py
```

Run exactly that. It fetches `face_landmarker.task` and `hand_landmarker.task`
into `models/`, verifies each is not truncated, and tells you what to do if your
network blocks the download. If it does, download them in a browser from the
URLs printed and drop them into `models/` yourself.

### MediaPipe reports neither Tasks nor Solutions API

MediaPipe 1.x removed `mp.solutions`. `mp_backend.py` prefers the Tasks API and
falls back to Solutions on older versions; if neither exists the version is too
old. Upgrade: `pip install -U "mediapipe>=0.10.14"`.

### numpy / mediapipe version conflict

MediaPipe **before 0.10.30** requires `numpy<2`. Let pip resolve it:

```bat
pip install -U "mediapipe>=0.10.30"
```

### Camera will not open

- Close every other app using it.
- Try `--camera 1`, `--camera 2`.
- On Windows, `camera.backend_windows: "dshow"` is the default; if your camera
  is a network/virtual device try removing it (set to `""`).
- Windows Settings → Privacy → Camera → allow desktop apps.

### FPS is low

See [Performance](#performance). Start with `process_scale: 0.5`.

### The cursor cannot reach the right or bottom edge

DPI scaling. Make sure `mouse.dpi_aware: true` (the default) and restart.
`--check` prints what awareness level was actually set.

### Everything is mirrored

`camera.mirror_preview` mirrors the *preview* so it behaves like a mirror, which
is what makes "touch your left temple" intuitive. The gaze mapping is computed
from calibration and is unaffected. If the preview feels wrong, flip it — but
recalibrate afterwards.

---

## Architecture

Modules are flat at the repository root so `python main.py` works with no
import-path setup.

| File | Responsibility |
|---|---|
| **`main.py`** | Entry point. Camera capture, the frame loop, calibration orchestration, hotkey handling, CLI, environment checks. |
| **`gaze.py`** | Pupil/iris extraction from landmarks, all the filters (moving average, One Euro, Kalman, deadzone), and `GazePipeline` which maps a gaze vector to screen pixels. **No MediaPipe import** — it takes landmark arrays, which is what makes it testable. |
| **`calibration.py`** | Grid generation, the interactive session, the least-squares fit (`CalibrationModel`), save/load, and the calibration screen rendering. |
| **`gestures.py`** | Turns per-frame observations into discrete `Event`s: temple clicks, winks, dwell, edge scroll, palm scroll, sniper. All arbitration lives here. |
| **`mouse_control.py`** | `MouseController` plus the `pynput` / `pyautogui` / null backends, DPI awareness, screen metrics, clamping. |
| `mp_backend.py` | The only module that imports MediaPipe. Tasks and legacy Solutions backends behind one interface, model discovery, timestamp monotonicity. |
| `overlay.py` | Preview window: HUD, gaze trail, landmark drawing, event flashes, live sliders, optional desktop dot. |
| `hotkeys.py` | Global keyboard hook with graceful degradation, edge-triggered presses, auto-repeat suppression, live hold state. |
| `landmarks.py` | Canonical MediaPipe landmark indices (iris rings, eye corners/lids, temples, fingertips). |
| `config.py` | Defaults, ranges, choices, validation with repair-and-report, atomic load/save. |
| `util.py` | `clamp`, a monotonic clock (with a manual mode for tests), `FPSMeter`, `StageTimer`. |
| `synthetic.py` | Generates plausible Face Mesh / Hands landmark arrays so the whole pipeline can be tested without a webcam. |
| `tools/download_models.py` | Fetches and verifies the two `.task` model bundles. |
| `tools/simulate.py` | A 43-check end-to-end simulation of the whole app. No webcam, no display, no MediaPipe. |
| `tests/` | 443 pytest cases. |

**Design notes that matter:**

- **The gaze extractor runs in two passes.** It first computes eye aspect ratios
  for both eyes, and *only then* tries to resolve a pupil. Computing EAR after an
  early return would make a full blink look like "face lost" instead of "blink",
  which in turn would make blinking fire clicks.
- **Per-eye axis sign.** The lateral→medial direction points opposite ways in
  image coordinates for the left and right eye. Averaging raw per-eye projections
  without a per-eye `axis_sign` cancels them out and pins gaze to the centre.
- **Sniper scales deltas, not absolute positions.** Remapping absolutely would
  teleport the cursor on exit.
- **`GestureEngine.reset()` deliberately does not clear the debouncer.** It runs
  right after calibration, and a hand still resting on your temple must not
  produce a click the instant tracking starts.
- **Scroll amount signs follow `pyautogui.scroll()`:** positive is up, negative
  is down.

---

## Testing without a webcam

The entire pipeline is testable headlessly, because `gaze.py` and `gestures.py`
consume landmark arrays rather than images:

```bat
.venv\Scripts\activate
pytest tests -q              # 443 tests, ~2 s
python tools\simulate.py     # 43 end-to-end checks, ~2 s
```

`tools/simulate.py` drives the real `GazePipeline` and `GestureEngine` with
synthetic landmarks and a `NullMouse`, and asserts on behaviour that matters:

- a 9-point fit reaches **RMS < 8 px**, and interpolates unseen points to
  **< 25 px**;
- the cursor reaches 9 specified screen positions **within 30 px**;
- a horizontal sweep moves **monotonically** (no sign flips);
- smoothing cuts jitter by **> 40%**;
- blinks and repeated blinks produce **zero** clicks;
- both temples together give sniper mode and **zero** clicks;
- dwell fires **once** at ~1400 ms, re-arms after movement, and tolerates jitter;
- edge scrolling honours the 800 ms guard and **suppresses dwell-click**;
- the 500 ms debounce collapses a double wink into one click;
- non-inference overhead stays under **20 ms/frame**, leaving room for 15 fps.

`pytest` covers the filters, calibration maths, gaze extraction, every gesture
and its arbitration, config validation and repair, mouse backends (including the
Windows-only `pynput`/`pyautogui` paths via injected fakes), hotkey edge
triggering, overlay rendering and slider bounds, and the full `EyemouseApp`
frame loop end to end.

---

## Known limitations

Being straight about these will save you time:

- **Accuracy is ~20–40 px** after a good calibration. This is a property of
  estimating gaze from a single unlit webcam, not a bug. Sniper mode and dwell
  clicking exist to work around it.
- **It is head-relative.** Moving your head moves the pointer. There is no
  head-pose-invariant absolute gaze from a mono webcam.
- **Recalibrate when your setup changes** — different seat, different distance,
  different lighting, different monitor. Calibration is a snapshot.
- **Rubbing your eye fires a click.** Any wink-shaped event does. The
  `max_duration_ms` guard rejects a wink *confirmed* long after the eye closed
  (for example if the face was lost mid-wink), so a click cannot fire from stale
  state — but it cannot un-fire a genuine rising edge.
- **Glasses and sunglasses.** Reflections on lenses sit exactly where the pupil
  is. Sunglasses will not work at all.
- **One face.** `num_faces=1`, both for speed and because two faces would fight
  over the cursor.
- **The webcam must be roughly centred under or over the monitor.** Extreme
  angles make the iris geometry degenerate.

---

## Licence

Provided as-is for personal and accessibility use. MediaPipe model bundles are
distributed by Google under their own terms; `tools/download_models.py` fetches
them from the official `storage.googleapis.com/mediapipe-models` URLs and does
not redistribute them.
