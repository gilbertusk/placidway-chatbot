"""Code-level checks that run on every answer, after the LLM has written it.

The prompt asks the model not to invent prices, but a prompt is a request,
not a guarantee. These checks are the guarantee:

* `unsupported_amounts`: every money amount in the answer must appear in the
  passages the answer cites (or be the user's own figure being corrected).
  Amounts are compared as digits, so "$18,995", "18995 USD" and the Spanish
  "18.995 USD" all match.
* Citation ids must belong to passages that were actually retrieved
  (enforced in bot.py).
"""

from __future__ import annotations

import re

CURRENCY = r"(?:US\$|USD|EUR|GBP|MXN|TRY|THB|INR|AED|\$|€|£|₹|฿|dollars?|euros?|pesos?)"
NUMBER = r"\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
MONEY_RE = re.compile(
    rf"(?:{CURRENCY}\s?(?P<a>{NUMBER}))|(?:(?P<b>{NUMBER})\s?(?:{CURRENCY}))",
    re.IGNORECASE,
)
ANY_NUMBER_RE = re.compile(NUMBER)


def normalise_amount(raw: str) -> str:
    """'18,995' -> '18995', '18.995' -> '18995', '1,500.00' -> '1500', '64.50' -> '6450'."""
    raw = raw.strip()
    raw = re.sub(r"[.,]00$", "", raw)  # drop a zero cents part
    return re.sub(r"\D", "", raw).lstrip("0") or "0"


def money_amounts(text: str) -> set[str]:
    amounts = set()
    for match in MONEY_RE.finditer(text or ""):
        raw = match.group("a") or match.group("b")
        if raw:
            amounts.add(normalise_amount(raw))
    return amounts


SIMPLE_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")


def all_numbers(text: str) -> set[str]:
    """Every number in `text`, normalised. Read two ways so that a space used as a
    thousands separator ("18 995") and two separate numbers ("3 950") both count."""
    text = text or ""
    found = {normalise_amount(m.group(0)) for m in ANY_NUMBER_RE.finditer(text)}
    found |= {normalise_amount(m.group(0)) for m in SIMPLE_NUMBER_RE.finditer(text)}
    return found


def unsupported_amounts(answer: str, source_text: str, user_text: str = "") -> list[str]:
    """Money amounts in `answer` that do not appear anywhere in the cited sources.

    Amounts the user typed are allowed so the bot can say "the page does not
    list $50; it lists $18,995". Whether the bot *agrees* with the user's
    figure is handled by the prompt; this check stops invented numbers.
    """
    allowed = all_numbers(source_text) | all_numbers(user_text)
    return sorted(a for a in money_amounts(answer) if a not in allowed)
