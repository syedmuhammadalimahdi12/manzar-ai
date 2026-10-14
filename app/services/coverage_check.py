"""
MANZAR — Coverage Check Service  (SRS Section 9.1)
====================================================
Evaluates whether captured images give sufficient angular coverage
of the accident scene before reconstruction begins.

Required angles: front, rear, left, right
Each photo carries an angle_hint from the capture UI.

Logic:
  1. Group photos by angle_hint.
  2. Check which of the 4 required angles are represented.
  3. If all 4 present with >= min photos each → sufficient.
  4. Otherwise → return the specific missing angles.
"""

from __future__ import annotations

from typing import Any


# The four cardinal angles we need for a complete scene
REQUIRED_ANGLES = {"front", "rear", "left", "right"}

# Minimum photos per angle to consider it "covered"
MIN_PHOTOS_PER_ANGLE = 1

# Total minimum photos (across all angles) for reconstruction to be meaningful
MIN_TOTAL_PHOTOS = 4


def check_coverage(photos: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyse photo coverage based on angle hints.

    Parameters
    ----------
    photos : list of photo records, each with at least {"angle_hint": str|None}

    Returns
    -------
    dict with:
        sufficient       : bool
        missing_angles   : list[str]  — specific angle names not yet covered
        covered_angles   : list[str]  — angles that are covered
        photo_count      : int
        angle_breakdown  : dict[str, int] — angle → photo count
        recommendation   : str  — human-readable suggestion
    """
    # --- Build angle → photo count map ---
    angle_counts: dict[str, int] = {}
    untagged = 0

    for photo in photos:
        hint = (photo.get("angle_hint") or "").strip().lower()
        if hint and hint in REQUIRED_ANGLES:
            angle_counts[hint] = angle_counts.get(hint, 0) + 1
        else:
            untagged += 1

    # --- Determine covered vs missing ---
    covered_angles = [
        angle for angle in REQUIRED_ANGLES
        if angle_counts.get(angle, 0) >= MIN_PHOTOS_PER_ANGLE
    ]
    missing_angles = sorted(REQUIRED_ANGLES - set(covered_angles))

    total = len(photos)
    sufficient = (
        len(missing_angles) == 0
        and total >= MIN_TOTAL_PHOTOS
    )

    # --- Build recommendation ---
    if sufficient:
        recommendation = "All angles covered. Ready for reconstruction."
    elif missing_angles:
        angle_list = ", ".join(missing_angles)
        recommendation = (
            f"Missing angle(s): {angle_list}. "
            f"Please capture {MIN_PHOTOS_PER_ANGLE}+ photo(s) from each missing angle."
        )
    else:
        recommendation = (
            f"Need at least {MIN_TOTAL_PHOTOS} photos total. "
            f"Currently have {total}."
        )

    return {
        "sufficient": sufficient,
        "missing_angles": list(missing_angles),
        "covered_angles": sorted(covered_angles),
        "photo_count": total,
        "untagged_photos": untagged,
        "angle_breakdown": angle_counts,
        "recommendation": recommendation,
    }
