from talon import quotations


def _strip_signature(text: str) -> str:
    """Cut at the standard '-- ' signature delimiter line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == "--" or line.rstrip() == "-- ":
            return "\n".join(lines[:i]).rstrip()
    return text


def clean_body(text: str) -> str:
    """Remove quoted replies (talon) and a trailing signature block."""
    if not text:
        return ""
    without_quotes = quotations.extract_from_plain(text)
    return _strip_signature(without_quotes).strip()
