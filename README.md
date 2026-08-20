# clipbot

Turns a full movie or episode into short, vertical, ready-to-post clips.

```
./run.sh
```

That installs what's missing, starts the UI, and opens
[http://127.0.0.1:8000](http://127.0.0.1:8000). Drop in a video, say how many
clips you want and how long each should be, and download them when they're
done.

## Install

ffmpeg is the only thing you need beforehand:

```
brew install ffmpeg              # macOS
sudo apt install ffmpeg          # Linux
winget install Gyan.FFmpeg       # Windows, then restart your shell
```

`run.sh` handles the rest (virtualenv + `pip install -r requirements.txt`).
Python 3.10 or newer is required — librosa dropped 3.9, and macOS still ships
it.

## The two modes

**Highlights** — the default, and the one you want for an episode or a film.
It finds the strongest moments and cuts them straight, keeping the source's own
audio, so a clip is a real excerpt with the dialogue intact.

**Beat-synced edit** — the original montage: shots pulled from across the
source and cut to a track you supply. Needs a song.

## How highlights work

1. **One coarse pass.** The whole file is decoded once at 64x36 greyscale, a
   few samples per second. That yields per-moment brightness, contrast, motion,
   shot boundaries, and where in the frame the action sits — cheap enough to
   run over a two-hour film.
2. **Audio envelope.** Loudness *and* local dynamics, from the same file.
   Loudness says something is playing; dynamics say something is *happening* —
   dialogue and impacts modulate, room tone and a score drone don't. Without
   that term the credits scroll ranks as highly as a fight.
3. **Window scoring.** A rolling mean of the combined interest signal, plus a
   bonus for densely-cut passages, minus everything that's a black frame, a
   fade, or a flat title card.
4. **Spread.** Picks are greedy on score but with an enforced minimum gap, so
   you get clips from across the runtime instead of five angles on the same
   two-minute set piece.
5. **Snap.** Each pick is pulled to a real shot boundary, found by re-scanning
   just that window at full frame rate. A clip that opens three frames into a
   shot reads as a mistake.
6. **Framing.** Cropping 16:9 to 9:16 throws away ~60% of the width, so the
   crop position is resolved *per shot* from where the activity actually is,
   and held constant within each shot. The step happens on a cut, where it's
   invisible; a continuously moving crop reads as a drifting camera.

## Command line

The UI and the CLI share one pipeline, so they can't drift apart.

```
clipbot ui                            open the UI
clipbot clips VIDEO -n 8 -s 30        8 clips, 30s each
clipbot clips VIDEO --music SONG.mp3  beat-synced montages instead
clipbot VIDEO SONG.mp3 -d 30          the original single edit
```

`clipbot clips` options:

| Flag | Meaning |
|---|---|
| `-n, --count` | How many clips |
| `-s, --secs` | Length of each, in seconds |
| `-o, --out` | Output directory |
| `--music` | Supply a track to get beat-synced montages |
| `--aspect` | `9:16`, `4:5`, `1:1`, `16:9` |
| `--quality` | `max`, `high`, `balanced`, `small` |
| `--skip-intro` / `--skip-outro` | Ignore logos / recaps / credits |
| `--no-sharpen` | Skip the post-crop sharpen |
| `--no-normalize` | Leave loudness alone |
| `--seed N` | Reroll a beat-synced edit. Same seed = same edit |

The original single-edit command and all of its flags are unchanged — see
`clipbot VIDEO AUDIO -h`.

## Output

Every clip is 1080p in the chosen shape, h264 high profile, `+faststart`, with
bt709 tagged on the frames. Audio is AAC 192k stereo at 48 kHz, normalised to
−14 LUFS.

Colour is tagged with the `setparams` filter rather than `-colorspace` and
friends: on current ffmpeg builds the encoder options only land the matrix, and
untagged primaries/transfer is why a re-uploaded clip can come back looking
washed out.

Timestamps are rebased to zero on both streams (`setpts` / `asetpts`). A fast
seek lands video on the next whole frame but audio on the next packet, which
otherwise leaves the two starting far enough apart to read as a lip-sync error.
Measured on a source with a synchronised flash and click, cutting adds about
16 ms — inside one AAC frame.

## Memory

**Jobs run one at a time, deliberately.** Each render is a large-frame filter
graph; several at once can exhaust system memory and force a hard power-off.
Raise it with `clipbot ui --workers N` only if you have headroom to spare.

The beat-synced path's `--jobs` is additionally clamped at runtime against free
RAM, and its effects use `crop`+`scale` rather than `zoompan` for the same
reason — `zoompan` renders its intermediate at the input resolution and holds a
large internal canvas, which at 1080x1920 is on the order of a gigabyte per
process. The current chain peaks at roughly 100-160 MB per segment.

## Notes on quality

- Beat-synced segments are written to a near-lossless intermediate (CRF 16,
  ultrafast) so the **final** pass is the only encode that costs real quality.
  It's more disk I/O than a one-shot filtergraph, but it's what makes per-clip
  effects tractable, and the picture holds up.
- Widescreen source is scaled to fill and cropped, so the sides of a 2.39:1
  frame are lost. That is the right trade for this format — black bars read as
  lazy. Highlight mode picks the crop position from the action; the beat-synced
  path takes a fixed `--bias`.
- There is no face tracking. The framing heuristic follows motion and detail,
  which is usually the subject but isn't guaranteed to be.
- Beat tracking assumes a steady 4/4 pulse. Rubato, live drumming, and heavy
  tempo changes will drift; `--max-stride 1` and a manual `--audio-start`
  usually recover it.
- A beat-synced clip's length is set by the beat grid, so it lands near the
  requested figure rather than exactly on it. Highlight clips are exact.

## Where things live

| Path | |
|---|---|
| `clipbot/analyze.py` | the coarse whole-file pass |
| `clipbot/highlight.py` | moment selection, framing, clip render |
| `clipbot/pipeline.py` | orchestration shared by UI and CLI |
| `clipbot/jobs.py` | job queue, progress, cancellation |
| `clipbot/server.py` | HTTP API |
| `clipbot/web/` | the UI (no build step) |
| `clipbot/beats.py` `plan.py` `render.py` `shots.py` | the beat-synced path |

Uploads and rendered clips go to `~/.clipbot` by default; set `CLIPBOT_DATA` to
move that. Nothing is sent anywhere — the server binds to localhost and has no
authentication, so don't put it on a public address.

## Rights

Movie footage and commercial music are somebody else's copyright. Whether a
given edit is fair use, and whether it survives Content ID, is on you.
