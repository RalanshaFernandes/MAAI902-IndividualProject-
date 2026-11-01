"""Suggested labels for the gold set, produced by an AI annotator (Claude).

Labelled from the item text only (item_id, source, company, text) - the
annotator did not see FinBERT's predictions. Criterion: would this sentence,
as news today, be good / bad / neither for a shareholder of the company?
Garbled OCR text is "unclear". These are SUGGESTIONS: the author reviews and
corrects them in src/app/review_gold_labels.py (review mode); the report must describe the gold
set as "AI-suggested labels (Claude), verified and corrected by the author".

Output: labels/ai_labels.csv
"""
import pandas as pd

from config import ROOT

P, N, Z, U = "positive", "negative", "neutral", "unclear"
LABELS = {
    "G001": P, "G002": P, "G003": P, "G004": Z, "G005": U, "G006": Z, "G007": Z, "G008": P, "G009": N, "G010": P,
    "G011": P, "G012": P, "G013": Z, "G014": P, "G015": Z, "G016": P, "G017": Z, "G018": Z, "G019": P, "G020": Z,
    "G021": P, "G022": Z, "G023": Z, "G024": P, "G025": P, "G026": P, "G027": P, "G028": Z, "G029": Z, "G030": P,
    "G031": N, "G032": P, "G033": N, "G034": P, "G035": P, "G036": N, "G037": P, "G038": P, "G039": N, "G040": U,
    "G041": P, "G042": P, "G043": Z, "G044": Z, "G045": Z, "G046": Z, "G047": Z, "G048": P, "G049": Z, "G050": P,
    "G051": P, "G052": Z, "G053": Z, "G054": Z, "G055": N, "G056": P, "G057": P, "G058": N, "G059": Z, "G060": Z,
    "G061": P, "G062": Z, "G063": Z, "G064": P, "G065": P, "G066": P, "G067": Z, "G068": Z, "G069": P, "G070": Z,
    "G071": Z, "G072": P, "G073": Z, "G074": P, "G075": Z, "G076": Z, "G077": Z, "G078": Z, "G079": P, "G080": P,
    "G081": Z, "G082": P, "G083": N, "G084": P, "G085": Z, "G086": Z, "G087": Z, "G088": P, "G089": Z, "G090": Z,
    "G091": P, "G092": P, "G093": Z, "G094": P, "G095": P, "G096": P, "G097": Z, "G098": P, "G099": Z, "G100": Z,
    "G101": P, "G102": Z, "G103": P, "G104": Z, "G105": N, "G106": N, "G107": N, "G108": P, "G109": Z, "G110": Z,
    "G111": Z, "G112": Z, "G113": Z, "G114": Z, "G115": Z, "G116": P, "G117": Z, "G118": Z, "G119": P, "G120": Z,
    "G121": Z, "G122": P, "G123": N, "G124": N, "G125": Z, "G126": Z, "G127": P, "G128": Z, "G129": Z, "G130": P,
    "G131": N, "G132": Z, "G133": P, "G134": P, "G135": Z, "G136": Z, "G137": P, "G138": P, "G139": Z, "G140": P,
    "G141": Z, "G142": Z, "G143": P, "G144": Z, "G145": N, "G146": P, "G147": P, "G148": N, "G149": P, "G150": N,
    "G151": P, "G152": P, "G153": P, "G154": N, "G155": P, "G156": Z, "G157": Z, "G158": P, "G159": Z, "G160": P,
    "G161": Z, "G162": Z, "G163": N, "G164": N, "G165": P, "G166": Z, "G167": Z, "G168": N, "G169": P, "G170": Z,
    "G171": Z, "G172": Z, "G173": Z, "G174": Z, "G175": Z, "G176": Z, "G177": Z, "G178": Z, "G179": Z, "G180": Z,
    "G181": Z, "G182": Z, "G183": P, "G184": Z, "G185": Z, "G186": Z, "G187": Z, "G188": Z, "G189": P, "G190": Z,
    "G191": Z, "G192": Z, "G193": Z, "G194": Z, "G195": Z, "G196": Z, "G197": Z, "G198": Z, "G199": P, "G200": Z,
    "G201": Z, "G202": Z, "G203": Z, "G204": N, "G205": P, "G206": Z, "G207": P, "G208": Z, "G209": N, "G210": P,
    "G211": Z, "G212": P, "G213": N, "G214": P, "G215": P, "G216": N, "G217": N, "G218": Z, "G219": Z, "G220": Z,
    "G221": U, "G222": Z, "G223": P, "G224": Z, "G225": Z, "G226": P, "G227": Z, "G228": Z, "G229": N, "G230": P,
    "G231": Z, "G232": P, "G233": P, "G234": Z, "G235": P, "G236": P, "G237": P, "G238": Z, "G239": N, "G240": Z,
    "G241": P, "G242": Z, "G243": P, "G244": N, "G245": P, "G246": Z, "G247": Z, "G248": Z, "G249": Z, "G250": P,
    "G251": Z, "G252": Z,
}

if __name__ == "__main__":
    out = pd.DataFrame({"item_id": list(LABELS), "ai_label": list(LABELS.values())})
    sample = pd.read_csv(ROOT / "labels" / "gold_sample.csv")
    assert set(out["item_id"]) == set(sample["item_id"]), "label ids must match the sample"
    out.to_csv(ROOT / "labels" / "ai_labels.csv", index=False)
    print(out["ai_label"].value_counts().to_string())
