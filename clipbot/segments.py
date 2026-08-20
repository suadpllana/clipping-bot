"""Find the parts of an episode that are not the episode.

An anime episode is not one continuous piece of story. It is, typically:

    recap → opening theme → part A → eyecatch → part B → ending theme → preview

Four of those seven are the same every week, and none of them is a clip. They
are also exactly what a naive "most is happening here" score picks first: an
opening sequence is a loud, fast-cut, high-motion montage — a highlight
detector's dream and a viewer's skip button.

Two signals separate them from the show, and both are cheap:

* **A music bed.** Themes are mastered flat. The level never falls away
  between phrases the way it does between lines of dialogue.
* **Nobody speaks.** Where the file carries a subtitle track this is decisive:
  a minute and a half without a single cue, in a show that talks constantly,
  is the opening.

Boundaries then snap to the beat of near-silence that broadcast animation puts
either side of every structural join, which lands them within a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analyze import Timeline

# What a theme song plausibly is. Shorter is a bumper, longer is a scene.
MIN_THEME = 35.0
MAX_THEME = 210.0


def _edge_zone(duration: float) -> float:
    """How near an end of the file a theme has to sit to be *the* opening or
    *the* ending rather than a montage in the middle of the show."""
    return max(300.0, duration * 0.25)


LABELS = {
    "recap": "recap",
    "opening": "opening theme",
    "leadin": "recap and opening",
    "ending": "ending theme",
    "preview": "next-episode preview",
    "music": "music montage",
}


@dataclass
class Segment:
    start: float
    end: float
    kind: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def label(self) -> str:
        return LABELS.get(self.kind, self.kind)


@dataclass
class Structure:
    """What was found, and the per-sample multiplier it implies."""
    segments: list[Segment] = field(default_factory=list)
    weight: np.ndarray | None = None
    used_subs: bool = False

    def excluded(self) -> list[Segment]:
        """The segments a clip may not come from at all.

        A music montage is not one of them: it is worse than a scene but it is
        still the show, and on a source with no dialogue it may be all there is.
        """
        return [s for s in self.segments if s.kind != "music"]

    def summary(self) -> str:
        out = self.excluded()
        if not out:
            return ""
        return ", ".join(f"{s.label} {_hms(s.start)}–{_hms(s.end)}" for s in out)


def _hms(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------

def _runs(mask: np.ndarray, fps: float, *, bridge: float = 0.0,
          least: float = 0.0) -> list[list[float]]:
    """Contiguous True stretches of `mask`, in seconds."""
    if not mask.any():
        return []
    edges = np.flatnonzero(
        np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)))
    out = [[float(a) / fps, float(b) / fps] for a, b in zip(edges[::2], edges[1::2])]

    if bridge > 0:
        merged = [out[0]]
        for r in out[1:]:
            if r[0] - merged[-1][1] <= bridge:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        out = merged
    return [r for r in out if r[1] - r[0] >= least]


def _snap_out(gaps: list[tuple[float, float]], t: float, tol: float,
              *, later: bool) -> float:
    """Move a theme boundary out to the furthest near-silence within `tol`.

    Outward and furthest, not nearest, because the two errors are not
    symmetric. A theme runs into a title card or a beat of dead air before the
    show resumes, and swallowing that costs nothing; stopping short of it
    leaves the last seconds of the theme sitting in a clip.
    """
    near = [g for g in gaps if g[1] >= t - tol and g[0] <= t + tol]
    if not near:
        return t
    return max(t, max(g[1] for g in near)) if later else min(t, min(g[0] for g in near))


def _split_at_silence(run: list[float], gaps: list[tuple[float, float]],
                      bed: np.ndarray, fps: float) -> list[float]:
    """Trim a run to the sub-span that a pair of silences actually brackets.

    A climax scored under continuous music runs straight into the ending theme
    with no dialogue to separate them, so the raw run covers both. The silence
    the broadcast puts before the theme is the real boundary; whichever side of
    it holds the stronger music bed is the theme, and the other side is show.
    """
    inner = [g for g in gaps if run[0] + 20.0 < g[0] and g[1] < run[1] - 20.0]
    if not inner:
        return run

    parts = []
    edge = run[0]
    for g in inner:
        parts.append((edge, g[0]))
        edge = g[1]
    parts.append((edge, run[1]))

    def strength(p: tuple[float, float]) -> float:
        a, b = int(p[0] * fps), int(p[1] * fps)
        seg = bed[a:max(b, a + 1)]
        return float(seg.mean()) if len(seg) else 0.0

    ok = [p for p in parts if p[1] - p[0] >= MIN_THEME]
    if not ok:
        return run
    best = max(ok, key=strength)
    return [best[0], best[1]]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def analyse(tl: Timeline, *, enabled: bool = True) -> Structure:
    """Locate recap / opening / ending / preview and weight the timeline."""
    n = len(tl.brightness)
    weight = np.ones(n, dtype=np.float32)
    st = Structure(weight=weight)
    if not enabled or not tl.has_audio or tl.duration < 120.0:
        return st

    fps = tl.fps
    dur = tl.duration
    beds = _runs(tl.bed > 0.45, fps, bridge=15.0, least=30.0)

    # Where a subtitle track exists, a theme is the stretch nobody speaks over,
    # and that stretch is a far tighter boundary than the music bed alone —
    # the bed bleeds into the scored scene the theme follows.
    wordless: list[list[float]] = []
    if tl.dialogue is not None:
        st.used_subs = True
        wordless = _runs(tl.dialogue < 0.5, fps, bridge=4.0, least=40.0)

    cands: list[list[float]] = []
    for w in wordless:
        overlap = sum(max(0.0, min(b[1], w[1]) - max(b[0], w[0])) for b in beds)
        if overlap >= 30.0:
            cands.append(list(w))

    if not cands:
        # No wordless stretch to go on — either there is no subtitle track, or
        # it captions the theme's lyrics, which makes the opening look like
        # ninety seconds of talking. Fall back to the music bed alone, but
        # never over something the captions say is a conversation.
        for b in beds:
            if tl.dialogue is not None:
                a, z = int(b[0] * fps), int(b[1] * fps)
                if float(tl.dialogue[a:max(z, a + 1)].mean()) > 0.35:
                    continue
            cands.append(list(b))

    edge = _edge_zone(dur)
    found: list[Segment] = []
    for run in cands:
        run = _split_at_silence(run, tl.quiet, tl.bed, fps)

        # Snap outwards onto the silence that brackets the join, so the beat of
        # dead air belongs to the theme rather than to the clip after it.
        run[0] = _snap_out(tl.quiet, run[0], 12.0, later=False)
        run[1] = _snap_out(tl.quiet, run[1], 12.0, later=True)

        if not MIN_THEME <= run[1] - run[0] <= MAX_THEME:
            continue

        if run[0] <= edge:
            found.append(Segment(run[0], run[1], "opening"))
        elif run[1] >= dur - edge:
            found.append(Segment(run[0], run[1], "ending"))
        else:
            found.append(Segment(run[0], run[1], "music"))

    # Only one of each: the strongest candidate wins, and any other run near an
    # end of the file is a montage, not a second opening.
    found = _dedupe(found, tl, fps)

    # Everything before the opening is recap, logos and last week's cliffhanger
    # — the thing that gets clipped by mistake more than anything else. Same
    # story after the ending, where the next-episode preview lives.
    op = next((s for s in found if s.kind == "opening"), None)
    ed = next((s for s in found if s.kind == "ending"), None)
    if op and op.start > 8.0:
        found.append(Segment(0.0, op.start, "recap"))
    elif op:
        # The run reached the top of the file, so it is the recap and the theme
        # together. Say so rather than naming a boundary that was not found.
        op.kind = "leadin"
    if ed and dur - ed.end > 8.0:
        found.append(Segment(ed.end, dur, "preview"))

    found.sort(key=lambda s: s.start)
    for s in found:
        a, b = int(s.start * fps), int(np.ceil(s.end * fps))
        # A montage keeps a reduced weight rather than none: it is worse than a
        # scene but it is still the show, and on a source with no dialogue at
        # all it may be all there is.
        weight[max(a, 0):min(b, n)] *= 0.35 if s.kind == "music" else 0.0

    # Safety valve. If the heuristics have written off most of the runtime,
    # they are wrong about something — fall back to discouraging those regions
    # instead of forbidding them, so the job still returns clips.
    if float((weight < 0.5).mean()) > 0.55:
        weight[:] = np.maximum(weight, 0.2)

    st.segments = found
    return st


def _dedupe(found: list[Segment], tl: Timeline, fps: float) -> list[Segment]:
    """Keep the best opening and the best ending; demote the rest."""
    def strength(s: Segment) -> float:
        a, b = int(s.start * fps), int(s.end * fps)
        seg = tl.bed[a:max(b, a + 1)]
        return float(seg.mean()) if len(seg) else 0.0

    out: list[Segment] = []
    for kind in ("opening", "ending"):
        same = [s for s in found if s.kind == kind]
        if same:
            keep = max(same, key=strength)
            out.append(keep)
            out += [Segment(s.start, s.end, "music") for s in same if s is not keep]
    out += [s for s in found if s.kind == "music"]
    return out
