"""Collect disclosure metadata (EN + AR headlines and attachment paths).

DFM files each disclosure twice: an English record and a separate Arabic record
published a few seconds later, each with its own id and its own PDF. We pair
them by symbol and nearest publication time.

Output: data/raw/disclosures.csv (one row per English disclosure, with matched Arabic fields)
        data/raw/attachments.csv (one row per English attached document)
        data/raw/attachments_ar.csv (one row per Arabic attached document)
"""
import pandas as pd

from config import RAW, STOCKS
from collection.dfm_api import list_disclosures

# Maximum gap between an English filing and its Arabic twin.
PAIR_TOLERANCE = pd.Timedelta(minutes=10)


def to_frames(records: list[dict], symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten API records into a disclosures frame and an attachments frame."""
    disc, att = [], []
    for r in records:
        resources = r.get("resources") or []
        disc.append({"id": r["id"], "symbol": symbol, "published": r["publication_date"],
                     "headline": r["headline"], "n_attachments": len(resources)})
        for res in resources:
            att.append({"disclosure_id": r["id"], "attachment_id": res["id"], "symbol": symbol,
                        "published": r["publication_date"], "description": res.get("description", ""),
                        "language": res.get("language", ""), "r_path": res["r_path"]})
    disc, att = pd.DataFrame(disc), pd.DataFrame(att)
    disc["published"] = pd.to_datetime(disc["published"])
    if len(att):
        att["published"] = pd.to_datetime(att["published"])
    return disc, att


def pair_languages(en: pd.DataFrame, ar: pd.DataFrame) -> pd.DataFrame:
    """Attach each English disclosure's Arabic twin (nearest time, same symbol)."""
    ar = ar.rename(columns={"id": "id_ar", "headline": "headline_ar", "published": "published_ar"})
    ar["published"] = ar["published_ar"]
    merged = pd.merge_asof(
        en.sort_values("published"), ar[["symbol", "published", "id_ar", "headline_ar", "published_ar"]]
        .sort_values("published"),
        on="published", by="symbol", direction="nearest", tolerance=PAIR_TOLERANCE)
    # One Arabic record must not be matched to two English ones: keep the closest.
    merged["gap"] = (merged["published_ar"] - merged["published"]).abs()
    dup = merged["id_ar"].notna() & merged.sort_values("gap").duplicated("id_ar", keep="first").sort_index()
    merged.loc[dup, ["id_ar", "headline_ar", "published_ar"]] = None
    return merged.drop(columns=["gap", "published_ar"]).rename(columns={"headline": "headline_en"})


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    disclosures, attachments, attachments_ar = [], [], []

    for symbol in STOCKS:
        en_d, en_a = to_frames(list_disclosures(symbol, lang="en"), symbol)
        ar_d, ar_a = to_frames(list_disclosures(symbol, lang="ar"), symbol)
        paired = pair_languages(en_d, ar_d)
        print(f"{symbol:12} EN={len(en_d):4}  AR={len(ar_d):4}  paired={paired['id_ar'].notna().sum():4}")
        disclosures.append(paired)
        attachments.append(en_a)
        attachments_ar.append(ar_a)

    d = pd.concat(disclosures).sort_values(["symbol", "published"])
    d.to_csv(RAW / "disclosures.csv", index=False, encoding="utf-8-sig")
    pd.concat(attachments).sort_values(["symbol", "published"]).to_csv(
        RAW / "attachments.csv", index=False, encoding="utf-8-sig")
    pd.concat(attachments_ar).sort_values(["symbol", "published"]).to_csv(
        RAW / "attachments_ar.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(d)} disclosures; {d['id_ar'].notna().mean():.0%} have an Arabic twin")


if __name__ == "__main__":
    main()
