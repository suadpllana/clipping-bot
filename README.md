# clipbot

Turns a full movie or a 10-minute scenepack into a vertical, beat-synced edit
ready to post — 30s, 45s, or 60s.

```
python -m clipbot MOVIE.mp4 SONG.mp3 -o edit.mp4 -d 30
```

## Install

```
winget install Gyan.FFmpeg          # then restart your shell
pip install -r requirements.txt
```

## How it works

1. **Beat map** — `librosa` tracks beats on the percussive component of the
   track, plus per-beat energy. It also locates the **drop** (the largest
   sustained jump in energy) and starts the edit there by default, because
   that is the part people actually clip.
2. **Shot detection** — PySceneDetect cuts the source into individual shots.
   Long takes are subdivided rather than thrown away; a movie is mostly long
   takes, and they still contain usable 2-second pieces.
3. **Shot ranking** — each shot is sampled at low resolution and scored on
   contrast, motion, and brightness. Black frames, fades, and static title
   cards are dropped. Picks are spaced out so one action scene doesn't supply
   every shot.
4. **Cut plan** — beats are mapped to cuts. Cut density follows the music:
   every 4th beat in quiet sections, every beat once the energy is up. Pure
   data, nothing rendered yet.
5. **Render + mux** — each segment is rendered to 1080x1920/60fps, then
   concatenated and muxed with the music, normalised to -14 LUFS.

## Options

| Flag | Meaning |
|---|---|
| `-d, --duration` | Target length: `30`, `45`, `60` |
| `--seed N` | Reroll the edit. Same seed = same edit |
| `--audio-start S` | Force the song start instead of using the drop |
| `--skip-intro S` / `--skip-outro S` | Ignore studio logos / credits |
| `--max-stride N` | Cap beats-per-cut. `1` = cut on every single beat |
| `--no-score` | Skip visual ranking. Much faster, noticeably worse |
| `--jobs N` | Parallel renders. **See the memory note below** |
| `--keep-temp` | Keep intermediates for debugging |

Reroll until you like one — the seed is printed at the end of every run:

```
python -m clipbot movie.mkv song.mp3 -d 30 --seed 1
python -m clipbot movie.mkv song.mp3 -d 30 --seed 2
```

## Memory

**`--jobs` defaults to 1, deliberately.** Each render is a large-frame filter
graph; running several at once can exhaust system memory and force a hard
power-off. `--jobs` is additionally clamped at runtime against free RAM.
Raise it only if you have headroom to spare, and watch Task Manager the first
time you do.

Effects use `crop`+`scale` rather than `zoompan` for the same reason —
`zoompan` renders its intermediate at the input resolution and holds a large
internal canvas, which at 1080x1920 is on the order of a gigabyte per process.
The current chain peaks at roughly 100-160 MB per segment.

## Notes on quality

- Segments are written to a near-lossless intermediate (CRF 16, ultrafast) so
  the **final** pass is the only encode that costs real quality. It's more disk
  I/O than a one-shot filtergraph, but it's what makes per-clip effects
  tractable, and the picture holds up.
- Widescreen source is scaled to fill and centre-cropped, so the sides of a
  2.39:1 frame are lost. That is the right trade for this format — black bars
  read as lazy. If a specific shot needs its full width, cut it out of the
  source folder.
- Vertical framing is centre-weighted. There is no face tracking, so a subject
  standing far off-centre can end up cropped out of frame.
- Beat tracking assumes a steady 4/4 pulse. Rubato, live drumming, and heavy
  tempo changes will drift; `--max-stride 1` and a manual `--audio-start`
  usually recover it.

## Rights

Movie footage and commercial music are somebody else's copyright. Whether a
given edit is fair use, and whether it survives Content ID, is on you.
