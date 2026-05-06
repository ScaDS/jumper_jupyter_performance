"""Utility functions for magic command detection."""

from typing import FrozenSet
from functools import lru_cache

from IPython import get_ipython


@lru_cache(maxsize=1)
def get_line_magics_cached() -> FrozenSet[str]:
    """Return cached set of all registered line magic names.

    Returns:
        Frozen set of magic command names (without % prefix).
    """
    ip = get_ipython()
    return frozenset(ip.magics_manager.lsmagic().get("line", []))


def is_known_line_magic(line: str, line_magics: frozenset) -> bool:
    """Check if a line starts with a known magic command.

    Args:
        line: Single line of code to check.
        line_magics: Set of known magic names (without % prefix).

    Returns:
        True if line starts with %<known_magic>, False otherwise.
    """
    s = line.lstrip()
    if not s.startswith("%"):
        return False
    name = s[1:].split(None, 1)[0]
    return name in line_magics


def is_pure_line_magic_cell(raw_cell: str) -> bool:
    """
    A pure line-magic cell = each non-empty line is either:
      - starts with %<known_magic> (optionally with arguments),
      - or is a comment (#...).
    """
    line_magics = get_line_magics_cached()
    lines = raw_cell.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if is_known_line_magic(line, line_magics):
            continue
        return False
    return True


def is_local_magic_cell(raw_cell: str, local_magics: set) -> bool:
    """
    Check if cell contains only local magic commands (and comments/empty lines).
    Returns True if all non-empty, non-comment lines are local magics.
    """
    lines = raw_cell.splitlines()
    has_magic = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("%"):
            name = stripped[1:].split(None, 1)[0]
            if name in local_magics:
                has_magic = True
                continue
        return False
    return has_magic
