"""The fixed list of pages the bot is allowed to know about.

The crawler fetches exactly these URLs and never follows links, so the
knowledge base cannot grow beyond what PlacidWay shared for the assessment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str
    page_type: str
    url: str


SOURCES: tuple[Source, ...] = (
    Source(
        id="itc-profile",
        page_type="Clinic profile",
        url="https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    ),
    Source(
        id="adenocarcinoma-package",
        page_type="Treatment package",
        url="https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    ),
    Source(
        id="adrenal-package",
        page_type="Treatment package",
        url="https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    ),
    Source(
        id="anal-package",
        page_type="Treatment package",
        url="https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC",
    ),
    Source(
        id="itc-videos",
        page_type="Clinic videos",
        url="https://www.placidway.com/profile-videos/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    ),
    Source(
        id="itc-price-list",
        page_type="Price list",
        url="https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center",
    ),
    Source(
        id="alt-medicine-prices",
        page_type="Price comparison",
        url="https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1",
    ),
)

SOURCES_BY_ID = {s.id: s for s in SOURCES}
