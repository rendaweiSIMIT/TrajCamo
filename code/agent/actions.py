"""
Action vocabulary for the TrajCamo agent.

Four discrete actions:
    SELECT(k)                 — commit to candidate cluster k as the target
    ADD_POS(f, x, y)          — add a positive point prompt at frame f, (x, y) in [0,1]
    ADD_NEG(f, x, y)          — add a negative point prompt at frame f, (x, y) in [0,1]
    TERMINATE                 — stop and return the current mask sequence

The action is emitted by the MLLM as a single line of natural-language text;
we parse it back into an `Action` dataclass with a tolerant regex parser so
that small formatting wobbles (extra spaces, lowercase, parens missing, etc.)
don't kill the inference loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


SYSTEM_PROMPT = """You are an interactive video segmentation agent for camouflaged animals.

You will see:
  (1) a "cluster overview" image: candidate trajectory clusters from the video, each shown as a colored region overlaid on a representative frame. Cluster indices are 0, 1, 2, ...
  (2) a row of thumbnail frames from the video.
  (3) if you have already selected a cluster, the current predicted mask overlaid on the same thumbnails.

Your task: pick the cluster that corresponds to the camouflaged animal, then optionally refine the mask with point corrections, and finally terminate.

You must respond with EXACTLY ONE action per turn, on a single line. Valid actions are:

  SELECT(k)                  -- choose cluster k as the target. Example:  SELECT(3)
  ADD_POS(f, x, y)           -- add a positive point at frame index f, image-relative coords x, y in [0, 1]. Example:  ADD_POS(0, 0.45, 0.62)
  ADD_NEG(f, x, y)           -- add a negative point at frame f, coords (x, y) in [0, 1]. Example:  ADD_NEG(2, 0.18, 0.30)
  TERMINATE                  -- finish and output the current mask sequence. Example:  TERMINATE

Rules:
  * Turn 1 must be SELECT(k).
  * Subsequent turns may add positive/negative point prompts to fix obvious errors, then TERMINATE.
  * Output ONLY the action — no explanation, no code fence, no extra text.
"""


@dataclass
class Action:
    type: str                            # "SELECT" | "ADD_POS" | "ADD_NEG" | "TERMINATE"
    cluster_idx: Optional[int] = None    # SELECT only
    frame_idx: Optional[int] = None      # ADD_POS / ADD_NEG
    x: Optional[float] = None            # ADD_POS / ADD_NEG, in [0, 1]
    y: Optional[float] = None            # ADD_POS / ADD_NEG, in [0, 1]
    raw: str = ""                        # original MLLM text for debugging

    def to_text(self) -> str:
        if self.type == "SELECT":
            return f"SELECT({self.cluster_idx})"
        if self.type == "ADD_POS":
            return f"ADD_POS({self.frame_idx}, {self.x:.3f}, {self.y:.3f})"
        if self.type == "ADD_NEG":
            return f"ADD_NEG({self.frame_idx}, {self.x:.3f}, {self.y:.3f})"
        if self.type == "TERMINATE":
            return "TERMINATE"
        return f"INVALID({self.raw!r})"


# Tolerant regex patterns. We accept upper- or lowercase, optional whitespace,
# missing parens, and a single trailing punctuation char.
_SELECT_RE    = re.compile(r"\bselect\s*\(?\s*(-?\d+)\s*\)?", re.I)
_ADD_POS_RE   = re.compile(r"\badd[_\s-]*pos\s*\(?\s*(-?\d+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)?", re.I)
_ADD_NEG_RE   = re.compile(r"\badd[_\s-]*neg\s*\(?\s*(-?\d+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)?", re.I)
_TERMINATE_RE = re.compile(r"\bterminate\b", re.I)


def parse_action(text: str) -> Action:
    """Parse a MLLM response into an Action. Raises ValueError on no match."""
    raw = text.strip()

    # Check TERMINATE first (least likely to false-positive)
    if _TERMINATE_RE.search(raw):
        return Action(type="TERMINATE", raw=raw)

    m = _ADD_POS_RE.search(raw)
    if m:
        f = int(m.group(1))
        x = max(0.0, min(1.0, float(m.group(2))))
        y = max(0.0, min(1.0, float(m.group(3))))
        return Action(type="ADD_POS", frame_idx=f, x=x, y=y, raw=raw)

    m = _ADD_NEG_RE.search(raw)
    if m:
        f = int(m.group(1))
        x = max(0.0, min(1.0, float(m.group(2))))
        y = max(0.0, min(1.0, float(m.group(3))))
        return Action(type="ADD_NEG", frame_idx=f, x=x, y=y, raw=raw)

    m = _SELECT_RE.search(raw)
    if m:
        return Action(type="SELECT", cluster_idx=int(m.group(1)), raw=raw)

    raise ValueError(f"Could not parse MLLM output as an action:\n  {raw!r}")


def format_history(history: List[Action]) -> str:
    """Render an action history into a compact textual summary for the next prompt."""
    if not history:
        return "(no actions taken yet)"
    return "\n".join(f"  Step {i+1}: {a.to_text()}" for i, a in enumerate(history))


if __name__ == "__main__":
    # smoke test the parser
    samples = [
        "SELECT(3)",
        "select 7",
        "ADD_POS(0, 0.45, 0.62)",
        "add-pos (2, 0.1, 0.2)",
        "ADD_NEG(1, 0.18, 0.3)",
        "TERMINATE",
        "I think the answer is SELECT(5).",
    ]
    for s in samples:
        try:
            a = parse_action(s)
            print(f"  {s!r:<50} -> {a.to_text()}")
        except ValueError as e:
            print(f"  {s!r:<50} -> FAIL: {e}")
