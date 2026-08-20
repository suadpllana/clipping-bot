"""Command line interface."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from . import beats, plan, render, shots
from .ffmpeg import probe

# The Windows console defaults to cp1252, which cannot encode the box-drawing
# and check characters used below. Reconfigure rather than dropping to ASCII.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def edit_cmd(argv: list[str] | None = None) -> int:
    """The original single-edit command. Behaviour unchanged."""
    ap = argparse.ArgumentParser(
        prog="clipbot",
        description="Turn movies or scenepacks into beat-synced vertical edits.",
    )
    ap.add_argument("video", type=Path,
                    help="Source video file, or a folder of clips.")
    ap.add_argument("audio", type=Path,
                    help="Music track to cut to.")
    ap.add_argument("-o", "--out", type=Path, default=Path("out.mp4"))
    ap.add_argument("-d", "--duration", type=float, default=30.0,
                    help="Target length in seconds (30 / 45 / 60).")
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000,
                    help="Reroll the edit with a different seed.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel segment renders. Each one can use ~1-2 GB of "
                         "RAM at 1080x1920, so raise this only if you have "
                         "headroom to spare.")
    ap.add_argument("--audio-start", type=float, default=None,
                    help="Force the song start time; default picks the drop.")
    ap.add_argument("--skip-intro", type=float, default=0.0,
                    help="Ignore the first N seconds of the source.")
    ap.add_argument("--skip-outro", type=float, default=0.0,
                    help="Ignore the last N seconds (credits).")
    ap.add_argument("--max-stride", type=int, default=None,
                    help="Cap beats-per-cut. 1 = cut on every beat.")
    ap.add_argument("--bias", type=float, default=0.0, metavar="B",
                    help="Shift the vertical crop window: -1 hard left, "
                         "0 centre, 1 hard right. Useful on 2.39:1 sources, "
                         "where only ~23%% of the frame width survives.")
    ap.add_argument("--no-score", action="store_true",
                    help="Skip visual shot scoring (much faster, lower quality).")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args(argv)

    if not args.video.exists():
        print(f"error: no such video: {args.video}", file=sys.stderr)
        return 2
    if not args.audio.exists():
        print(f"error: no such audio: {args.audio}", file=sys.stderr)
        return 2

    work = args.out.resolve().parent / f".clipbot_{args.out.stem}"
    work.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Beat map -------------------------------------------------------
        print("[1/5] analysing audio…", flush=True)
        bm = beats.analyze(args.audio, work)
        if args.audio_start is not None:
            win = (args.audio_start, args.audio_start + args.duration)
        else:
            win = beats.pick_window(bm, args.duration)
        drop = f", drop at {_fmt(bm.drop)}" if bm.drop else ""
        print(f"      {bm.tempo:.1f} BPM, {len(bm.beats)} beats{drop}")
        print(f"      using {_fmt(win[0])}–{_fmt(win[1])}")

        # 2. Shots ----------------------------------------------------------
        print("[2/5] detecting shots…", flush=True)
        sources = shots.find_videos(args.video)
        if not sources:
            print("error: no video files found", file=sys.stderr)
            return 2
        pool: list[shots.Shot] = []
        for i, src in enumerate(sources, 1):
            if len(sources) > 1:
                print(f"      ({i}/{len(sources)}) {src.name}", flush=True)
            pool += shots.split_shots(
                src, skip_intro=args.skip_intro, skip_outro=args.skip_outro,
            )
        print(f"      {len(pool)} raw shots")

        if not pool:
            print("error: no shots detected — try --skip-intro 0 --skip-outro 0",
                  file=sys.stderr)
            return 1

        # 3. Score / rank ---------------------------------------------------
        print("[3/5] ranking shots…", flush=True)
        pool = shots.score_shots(pool, sample=not args.no_score)
        need = int(args.duration * 2.2) + 8      # generous headroom for fast cuts
        pool = shots.diversify(pool, need)
        print(f"      {len(pool)} usable")

        # 4. Plan + render --------------------------------------------------
        segs = plan.build_plan(bm, pool, window=win, seed=args.seed,
                               max_stride=args.max_stride)
        total = sum(s.duration for s in segs)
        jobs = render.safe_jobs(args.jobs)
        note = f", {jobs} at a time" if jobs > 1 else ""
        print(f"[4/5] rendering {len(segs)} cuts "
              f"(avg {total / len(segs):.2f}s{note})…", flush=True)

        tty = sys.stdout.isatty()

        def _prog(done, n):
            pct = int(done / n * 100)
            if tty:
                print(f"\r      {done}/{n}  {pct:3d}%", end="", flush=True)
            elif done == n or done % 10 == 0:
                print(f"      {done}/{n}  {pct:3d}%", flush=True)

        parts = render.render_all(segs, work / "segments",
                                  jobs=args.jobs, progress=_prog,
                                  bias=args.bias)
        if tty:
            print()

        # 5. Concat + mux ---------------------------------------------------
        print("[5/5] encoding final…", flush=True)
        render.concat_and_mux(parts, args.audio, win[0], total,
                              args.out, work)

        info = probe(args.out)
        print(f"\n✓ {args.out}  "
              f"{info.width}x{info.height} @{info.fps:.0f}fps  "
              f"{info.duration:.1f}s  "
              f"{args.out.stat().st_size / 1e6:.1f} MB")
        print(f"  seed {args.seed} — rerun with --seed {args.seed} to reproduce")
        return 0

    finally:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def ui_cmd(argv: list[str]) -> int:
    """Launch the web UI."""
    ap = argparse.ArgumentParser(prog="clipbot ui",
                                 description="Serve the clipbot web UI.")
    ap.add_argument("-p", "--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. Leave as-is unless you mean to expose "
                         "the UI on your network — it has no authentication.")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent jobs. Renders are memory-hungry; 1 is "
                         "deliberate.")
    args = ap.parse_args(argv)

    import os
    os.environ["CLIPBOT_WORKERS"] = str(max(args.workers, 1))
    try:
        from .server import serve
    except ImportError as e:
        print(f"error: the web UI needs fastapi and uvicorn:\n"
              f"  pip install -r requirements.txt\n  ({e})", file=sys.stderr)
        return 2
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def clips_cmd(argv: list[str]) -> int:
    """Cut a long source into N ready-to-post clips."""
    from .pipeline import Settings, run_pipeline

    ap = argparse.ArgumentParser(
        prog="clipbot clips",
        description="Cut a movie or episode into N ready-to-post clips.")
    ap.add_argument("video", type=Path)
    ap.add_argument("-n", "--count", type=int, default=5)
    ap.add_argument("-s", "--secs", type=float, default=30.0,
                    help="Length of each clip in seconds.")
    ap.add_argument("-o", "--out", type=Path, default=Path("clips"),
                    help="Output directory.")
    ap.add_argument("--music", type=Path, default=None,
                    help="Supply a track to make beat-synced montages instead "
                         "of straight highlight cuts.")
    ap.add_argument("--aspect", default="9:16",
                    choices=["9:16", "4:5", "1:1", "16:9"])
    ap.add_argument("--frame", default="auto",
                    choices=["auto", "fit", "fill", "pad"],
                    help="How a widescreen source becomes vertical. fit keeps "
                         "the whole frame over a blurred copy of itself; fill "
                         "crops to the action and loses the sides; pad uses "
                         "flat black. auto picks fit whenever a crop would "
                         "throw away real picture.")
    ap.add_argument("--quality", default="high",
                    choices=["max", "high", "balanced", "small"])
    ap.add_argument("--no-auto-skip", action="store_true",
                    help="Do not detect and avoid the recap, opening, ending "
                         "and next-episode preview.")
    ap.add_argument("--skip-intro", type=float, default=0.0)
    ap.add_argument("--skip-outro", type=float, default=0.0)
    ap.add_argument("--no-sharpen", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    args = ap.parse_args(argv)

    settings = Settings(
        video=args.video, out_dir=args.out, count=args.count, length=args.secs,
        mode="music" if args.music else "highlight", audio=args.music,
        aspect=args.aspect, frame=args.frame, quality=args.quality,
        auto_skip=not args.no_auto_skip,
        skip_intro=args.skip_intro, skip_outro=args.skip_outro,
        sharpen=not args.no_sharpen, normalize_audio=not args.no_normalize,
        seed=args.seed,
    )

    tty = sys.stdout.isatty()
    last = [""]

    def report(stage: str, frac: float, message: str) -> None:
        line = f"  [{int(frac * 100):3d}%] {message}"
        if tty:
            print(f"\r{line:<74}", end="", flush=True)
        elif message != last[0]:
            print(line, flush=True)
        last[0] = message

    try:
        results = run_pipeline(settings, report)
    except ValueError as e:
        if tty:
            print()
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if tty:
            print()
        print("interrupted", file=sys.stderr)
        return 130

    if tty:
        print()
    print()
    for c in results:
        print(f"  {c.index:2d}. {_fmt(c.start)} → {_fmt(c.end)}  "
              f"{c.duration:5.1f}s  {c.size / 1e6:6.1f} MB  "
              f"{','.join(c.tags)}")
    print(f"\n✓ {len(results)} clips in {args.out.resolve()}")
    return 0


_USAGE = """clipbot — clips out of movies and episodes

  clipbot ui                            open the web UI (upload, tweak, download)
  clipbot clips VIDEO -n 8 -s 30        cut 8 clips of 30s each
  clipbot clips VIDEO --music SONG.mp3  beat-synced montages instead
  clipbot VIDEO SONG.mp3 -d 30          the original single beat-synced edit

Add -h after any subcommand for its options."""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print(f"clipbot {__version__}")
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd in ("ui", "serve", "web"):
        return ui_cmd(rest)
    if cmd == "clips":
        return clips_cmd(rest)
    # No subcommand: the original `clipbot VIDEO AUDIO` form.
    return edit_cmd(argv)


if __name__ == "__main__":
    raise SystemExit(main())
