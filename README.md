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
It finds the fights and the conversations worth hearing, skips the recap, the
opening and the credits, and cuts straight, keeping the source's own audio — so
a clip is a real excerpt with the dialogue intact.

**Beat-synced edit** — the original montage: shots pulled from across the
source and cut to a track you supply. Needs a song.

## How highlights work

1. **One coarse pass.** The whole file is decoded once at 64x36 greyscale, a
   few samples per second. That yields per-moment brightness, contrast, motion,
   shot boundaries, and where in the frame the action sits — cheap enough to
   run over a two-hour film.
2. **Audio, in more than one dimension.** From the same file: loudness, local
   dynamics, transient impacts, how much of the energy sits in the speech band,
   and how hard the envelope modulates at the 2–8 Hz syllable rate. Loudness
   says something is playing; the rest say *what*.
3. **Structure.** An episode isn't one continuous thing, and four of its parts
   are the same every week. See below — this is the step that stopped clips
   coming back as the theme song.
4. **Two scores, not one.** A fight and a conversation look nothing alike, and
   a single blended "interest" curve finds neither: it rewards whatever is
   merely busy, which is exactly what an opening sequence is. So each is scored
   on its own terms — motion, impacts and cut rate against speech, delivery and
   dynamics — and a window is judged by whichever it is better at. Anything
   above two clips is held to a mix rather than five of the same kind.
5. **Spread.** Picks are greedy on score but with an enforced minimum gap, so
   you get clips from across the runtime instead of five angles on the same
   two-minute set piece.
6. **Snap.** Each pick slides onto real shot boundaries, found by re-scanning
   just that window at full frame rate. Both ends are graded and the window
   moves rather than stretching, so the length stays exact. A clip that opens
   three frames into a shot reads as a mistake, and one that stops three frames
   short of the next cut reads as a dropped connection.
7. **Framing.** See *Shape and framing*.

## Skipping the parts that aren't the episode

An episode goes recap → opening → part A → eyecatch → part B → ending →
next-episode preview, and none of those bookends is a clip. They are also a
highlight detector's favourite thing in the file: loud, fast-cut, high-motion,
wall-to-wall music. Two signals find them, and neither needs a model.

**A music bed.** Themes are mastered flat — the level never falls away between
phrases the way it does between lines of dialogue. A window whose *quiet* tenth
is still loud, over a minute or more, is a theme.

**Nobody speaks.** Where the file carries a text subtitle track, ninety seconds
without a single cue in a show that talks constantly is the opening. It is the
tighter of the two boundaries, because the music bed bleeds into whatever scene
the theme follows. A track that turns out to be signs-only is ignored rather
than believed.

Boundaries then snap to the beat of near-silence that broadcast animation puts
either side of every structural join, which lands them within about a second.
Everything before the opening is recap and everything after the ending is the
preview, so both go with them. Music in the *middle* of an episode is a
montage, not a theme, and is only discouraged.

If it finds nothing it says so and uses the whole file; if it somehow writes
off most of the runtime it downgrades itself to a preference rather than
handing back nothing. `--no-auto-skip` turns it off. `--skip-intro` and
`--skip-outro` still apply on top.

## Shape and framing

Cropping 16:9 to 9:16 keeps 32% of the width. Two thirds of every shot goes,
including whoever was standing at the edge of it, and choosing *where* to crop
does not get that back — which is why `--frame` defaults to not doing it.

| `--frame` | |
|---|---|
| `fit` | The picture over a blurred, desaturated copy of itself. How much picture is `--zoom`, below. |
| `fill` | Scaled up and cropped. The crop position is resolved *per shot* from where the activity is and held constant within the shot, so it steps on a cut where the step is invisible — a continuously moving crop reads as a drifting, unmotivated camera. |
| `pad` | The same as `fit` but on flat black. |
| `auto` | The default. `fill` when the source is already near the output shape and a crop costs nothing, `fit` otherwise. |

The backdrop is blurred at a sixth of the output resolution and scaled back up.
At that radius it is indistinguishable from blurring at full size, and it costs
about as much as one extra scale.

### `--zoom`

The whole 16:9 frame in a 9:16 canvas is 32% of its height — everything is
visible and everything is small. `--zoom` is the dial between that and a full
crop, rather than a third mode:

| `--zoom` | 16:9 into 9:16 |
|---|---|
| `1.0` | whole frame, picture 32% of the height |
| `1.4` | 71% of the width, picture 44% of the height — **the default** |
| `1.8` | 56% of the width, 57% of the height |
| `3.2`+ | the picture covers the canvas: identical to `fill` |

Past the point where the picture already covers the canvas, more zoom only
throws frame away, so it's clamped there. A source that is *already* the output
shape therefore ignores `--zoom` entirely instead of being cropped for nothing.
When zoom does trim the sides, it trims them at the same per-shot position
`fill` uses, so what survives is where the action is.

If the picture still reads as too small, the other lever is the output shape
itself: `--aspect 4:5` is a shorter canvas, so the same picture fills 45% of it
at `--zoom 1.0`.

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
| `--frame` | `auto`, `fit`, `fill`, `pad` — see *Shape and framing* |
| `--zoom` | How big the picture is inside the frame. 1.0 shows all of it |
| `--quality` | `max`, `high`, `balanced`, `small` |
| `--no-auto-skip` | Don't detect the recap / OP / ED / preview |
| `--skip-intro` / `--skip-outro` | Ignore a fixed number of seconds as well |
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
- `--frame` applies to highlight mode only. The beat-synced path always scales
  to fill and crops, with a fixed `--bias`.
- There is no face tracking. `fill`'s framing heuristic follows motion and
  detail, which is usually the subject but isn't guaranteed to be — one more
  reason `auto` prefers `fit` on a widescreen source.
- Structure detection is tuned on episodic television, where the theme is a
  minute and a half of unbroken music. A film has no opening to find, and it
  says so rather than inventing one.
- Beat tracking assumes a steady 4/4 pulse. Rubato, live drumming, and heavy
  tempo changes will drift; `--max-stride 1` and a manual `--audio-start`
  usually recover it.
- A beat-synced clip's length is set by the beat grid, so it lands near the
  requested figure rather than exactly on it. Highlight clips are exact.

## Where things live

| Path | |
|---|---|
| `clipbot/analyze.py` | the coarse whole-file pass |
| `clipbot/segments.py` | recap / OP / ED / preview detection |
| `clipbot/subs.py` | dialogue timing from an embedded subtitle track |
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
