from __future__ import annotations

import re


_ARTICLE_RE = re.compile(r"^\d{4}([._-]?)(\d{4})$")


def is_valid_article_code(text: str) -> bool:
    """Checks if text matches allowed article patterns:
    6042.0206 | 6042-0206 | 6042_0206 | 60420206
    """
    if not text:
        return False
    return bool(_ARTICLE_RE.match(text.strip()))

