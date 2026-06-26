"""FIFA World Cup 2026 Annex C third-place bracket assignments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
COMBINATION_FILE = BASE_DIR / "third_place_combination.csv"

# Group winners that face a third-placed team in the round of 32.
WINNER_GROUPS = ("A", "B", "D", "E", "G", "I", "K", "L")
SLOT_COLUMNS = tuple(f"slot_1{group}" for group in WINNER_GROUPS)


def _parse_third_group(value: str) -> str:
    """Convert a cell like '3D' to group letter 'D'."""
    if not value.startswith("3") or len(value) != 2:
        raise ValueError(f"Invalid third-place slot value: {value!r}")
    return value[1]


def load_annex_c_table(
    combination_file: Path = COMBINATION_FILE,
) -> dict[str, dict[str, str]]:
    """Load Annex C rows keyed by sorted qualifying group letters."""
    table: dict[str, dict[str, str]] = {}
    with combination_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            combination_key = row["groups_advancing"]
            table[combination_key] = {
                winner_group: _parse_third_group(row[column])
                for winner_group, column in zip(WINNER_GROUPS, SLOT_COLUMNS)
            }
    return table


_ANNEX_C_TABLE = load_annex_c_table()


def lookup_third_place_assignments(
    qualifying_groups: set[str] | frozenset[str],
) -> dict[str, str]:
    """Return winner-group to third-group mapping for one Annex C combination."""
    combination_key = "".join(sorted(qualifying_groups))
    try:
        return _ANNEX_C_TABLE[combination_key]
    except KeyError as exc:
        raise KeyError(
            f"No Annex C row for qualifying third-place groups: "
            f"{combination_key!r}"
        ) from exc


def winner_third_slot_frozensets(
    knockout_raw: list[tuple[str, Any, Any]],
) -> dict[str, frozenset[str]]:
    """Map each R32 group winner to its third-place slot frozenset."""
    mapping: dict[str, frozenset[str]] = {}
    for match_id, slot1, slot2 in knockout_raw:
        if not match_id.startswith("R_32"):
            continue
        for winner_slot, third_slot in ((slot1, slot2), (slot2, slot1)):
            if (
                winner_slot[0] == "group_pos"
                and len(winner_slot[1]) == 1
                and winner_slot[2] == 1
                and third_slot[0] == "group_pos"
                and len(third_slot[1]) > 1
                and third_slot[2] == 3
            ):
                mapping[winner_slot[1][0]] = frozenset(third_slot[1])
    return mapping
