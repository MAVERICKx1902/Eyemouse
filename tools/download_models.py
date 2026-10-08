#!/usr/bin/env python3
"""Download the MediaPipe model bundles Eyemouse needs.

The Tasks API (mediapipe >= 0.10.30, including all of 1.x) does **not** ship
the models inside the wheel -- ``FaceLandmarker`` and ``HandLandmarker`` need
``face_landmarker.task`` and ``hand_landmarker.task`` on disk. This script
fetches them into ``models/`` next to the project.

Usage::

    python tools/download_models.py            # fetch anything missing
    python tools/download_models.py --force    # re-download everything
    python tools/download_models.py --check    # report only, download nothing

If your machine cannot reach storage.googleapis.com (corporate proxy, offline
build), download the two URLs printed by ``--check`` on any machine and copy
them into ``models/`` -- nothing else is required.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request

# Make the project root importable when run as `python tools/download_models.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mp_backend as mpb  # noqa: E402

# Rough sizes for the float16 bundles; used only to sanity-check the download.
MIN_BYTES = {mpb.FACE_MODEL_NAME: 1_500_000, mpb.HAND_MODEL_NAME: 500_000}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def download(url: str, dest: str, timeout: int = 60) -> bool:
    """Stream ``url`` to ``dest`` via a temp file, so a failure never leaves a
    truncated file that later looks like a valid model."""
    print(f"  GET {url}")
    d = os.path.dirname(dest)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=d or ".", suffix=".part")
    os.close(tmp_fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Eyemouse-setup/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            with open(tmp_path, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100.0 * got / total
                        bar = "#" * int(pct // 4)
                        print(f"\r  [{bar:<25}] {pct:5.1f}%  {human(got)}", end="", flush=True)
                    else:
                        print(f"\r  {human(got)}", end="", flush=True)
        print()
        name = os.path.basename(dest)
        expected = MIN_BYTES.get(name, 100_000)
        size = os.path.getsize(tmp_path)
        if size < expected:
            print(f"  ! downloaded file is only {human(size)}; expected at least "
                  f"{human(expected)}. The response was probably an error page.")
            return False
        os.replace(tmp_path, dest)
        print(f"  OK  {dest}  ({human(size)})")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        print(f"\n  ! failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch Eyemouse's MediaPipe model bundles.")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--check", action="store_true", help="report status and exit")
    p.add_argument("--dir", default=None, help="target directory (default: <project>/models)")
    args = p.parse_args(argv)

    info = mpb.probe()
    target_dir = args.dir or mpb.model_dir(ROOT)

    print(f"mediapipe : {info.get('version')} (api: {info.get('api')})")
    if info.get("api") == "solutions":
        print("Your mediapipe exposes the legacy solutions API, which bundles its")
        print("own models. Downloading these files is harmless but unnecessary.")
    elif info.get("api") is None:
        print("! No usable MediaPipe landmark API detected:")
        for e in info.get("errors", []):
            print(f"    {e}")
    print(f"models dir: {target_dir}")

    found = mpb.find_models(ROOT)
    todo = []
    for name, url in mpb.MODEL_URLS.items():
        dest = os.path.join(target_dir, name)
        have = found.get(name)
        if have and not args.force:
            print(f"  = {name}: present ({human(os.path.getsize(have))}) at {have}")
            continue
        print(f"  - {name}: {'re-downloading' if have else 'missing'}")
        if args.check:
            print(f"      {url}")
            todo.append((name, url, dest))
            continue
        todo.append((name, url, dest))

    if args.check:
        missing = [n for n, _, _ in todo]
        print("RESULT: all models present" if not missing
              else f"RESULT: missing {', '.join(missing)}")
        return 0 if not missing else 1

    if not todo:
        print("Nothing to do.")
        return 0

    failures = []
    for name, url, dest in todo:
        print(f"\nDownloading {name} ...")
        if not download(url, dest):
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        print("\nTroubleshooting:")
        print("  * Behind a proxy?  set HTTPS_PROXY and retry.")
        print("  * No internet?     copy the files from another machine into")
        print(f"                     {target_dir}")
        print("  * Or pin the legacy API, which bundles its own models:")
        print("                     pip install \"mediapipe==0.10.21\" \"numpy<2\"")
        return 1
    print("All models ready. Verify with:  python main.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
