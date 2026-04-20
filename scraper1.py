import io
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

ZIP_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.zip"
OUT_DIR = Path("house_fd_data")
PDF_DIR = OUT_DIR / "pdfs"
RAW_DIR = OUT_DIR / "raw"
OUT_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

LOOKBACK_DAYS = 14
TIMEOUT = 60

# Asset codes you may want to treat as "market/security-ish"
SECURITY_CODES = {
    "ST",   # stock
    "OT",   # other / often ETFs, notes, etc.
    "MF",   # mutual fund
    "EF",   # exchange-traded fund if encountered
    "GS",   # government securities
    "CS",   # corporate securities if encountered
}

# Tickers or vehicle names that are still interesting even if not [ST]
COMMON_MARKET_TERMS = [
    "ETF", "SPDR", "Invesco", "Vanguard", "iShares", "QQQ", "SPY", "DIA"
]


def download_zip_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def extract_first_tabular_file(zip_bytes: bytes) -> tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        candidates = [
            n for n in names
            if n.lower().endswith(".txt") or n.lower().endswith(".csv")
        ]
        if not candidates:
            raise RuntimeError(f"No .txt or .csv file found in ZIP. Files: {names}")
        # Usually there is just one; otherwise prefer .txt first
        candidates.sort(key=lambda x: (not x.lower().endswith(".txt"), x))
        name = candidates[0]
        return name, zf.read(name)


def parse_fd_index(tabular_bytes: bytes) -> pd.DataFrame:
    # The Clerk export is typically tab-delimited text.
    text = tabular_bytes.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FilingDate_dt"] = pd.to_datetime(df["FilingDate"], errors="coerce")
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


def build_pdf_url(row: pd.Series) -> str:
    filing_type = str(row.get("FilingType", "")).strip().upper()
    year = str(row.get("Year", "")).strip()
    docid = str(row.get("DocID", "")).strip()

    if filing_type == "P":
        return f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{docid}.pdf"

    # Annual / broader disclosure bucket
    return f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}/{docid}.pdf"


def is_recent(row: pd.Series, lookback_days: int) -> bool:
    filing_date = row.get("FilingDate_dt")
    if pd.isna(filing_date):
        return False
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=lookback_days))
    return filing_date >= cutoff


def download_pdf(url: str, target_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200 and "application/pdf" in r.headers.get("content-type", "").lower():
            target_path.write_bytes(r.content)
            return True
        return False
    except Exception:
        return False


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def likely_purchase_snippets(text: str) -> list[str]:
    """
    Very heuristic:
    - looks for lines/blocks that contain transaction type P
    - prefers assets with [ST] or common market terms
    """
    text = clean_text(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = []

    for i, line in enumerate(lines):
        upper = line.upper()

        has_purchase = re.search(r"\bP\b", upper) is not None or " PURCHASE " in f" {upper} "
        has_security_code = any(f"[{code}]" in upper for code in SECURITY_CODES)
        has_market_term = any(term.upper() in upper for term in COMMON_MARKET_TERMS)

        # Common PTR lines often contain ticker in parentheses and [ST]
        if has_purchase and (has_security_code or has_market_term):
            snippet = line

            # Attach up to 2 following lines because descriptions often spill over
            for j in range(i + 1, min(i + 3, len(lines))):
                if len(lines[j]) < 250:
                    snippet += " | " + lines[j]

            hits.append(snippet)

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            deduped.append(h)

    return deduped


def main():
    print("Downloading ZIP...")
    zip_bytes = download_zip_bytes(ZIP_URL)
    (RAW_DIR / "2026FD.zip").write_bytes(zip_bytes)

    print("Extracting tabular file...")
    tab_name, tab_bytes = extract_first_tabular_file(zip_bytes)
    (RAW_DIR / tab_name).write_bytes(tab_bytes)

    print(f"Parsing {tab_name} ...")
    df = parse_fd_index(tab_bytes)
    df = normalize_dates(df)

    # Keep recent PTRs
    recent = df[df.apply(lambda r: is_recent(r, LOOKBACK_DAYS), axis=1)].copy()
    recent_ptr = recent[recent["FilingType"].str.upper() == "P"].copy()

    if recent_ptr.empty:
        print("No recent PTR filings found.")
        recent.to_csv(OUT_DIR / "recent_filings.csv", index=False)
        return

    recent_ptr["pdf_url"] = recent_ptr.apply(build_pdf_url, axis=1)

    review_rows = []

    for _, row in recent_ptr.sort_values("FilingDate_dt", ascending=False).iterrows():
        docid = str(row["DocID"]).strip()
        year = str(row["Year"]).strip()
        pdf_url = row["pdf_url"]

        pdf_path = PDF_DIR / f"{year}_{docid}.pdf"
        print(f"Checking {docid}: {pdf_url}")

        ok = pdf_path.exists() or download_pdf(pdf_url, pdf_path)
        if not ok:
            review_rows.append({
                "DocID": docid,
                "Year": year,
                "Last": row.get("Last", ""),
                "First": row.get("First", ""),
                "FilingDate": row.get("FilingDate", ""),
                "FilingType": row.get("FilingType", ""),
                "pdf_url": pdf_url,
                "pdf_found": False,
                "purchase_hit_count": 0,
                "purchase_snippets": "",
            })
            continue

        text = extract_pdf_text(pdf_path)
        snippets = likely_purchase_snippets(text)

        review_rows.append({
            "DocID": docid,
            "Year": year,
            "Last": row.get("Last", ""),
            "First": row.get("First", ""),
            "FilingDate": row.get("FilingDate", ""),
            "FilingType": row.get("FilingType", ""),
            "pdf_url": pdf_url,
            "pdf_found": True,
            "purchase_hit_count": len(snippets),
            "purchase_snippets": " || ".join(snippets[:10]),
        })

    out = pd.DataFrame(review_rows)

    # Only rows with likely purchases
    likely = out[(out["pdf_found"] == True) & (out["purchase_hit_count"] > 0)].copy()

    out.to_csv(OUT_DIR / "recent_ptr_review.csv", index=False)
    likely.to_csv(OUT_DIR / "recent_stock_purchase_hits.csv", index=False)

    print("\nDone.")
    print(f"Review file: {OUT_DIR / 'recent_ptr_review.csv'}")
    print(f"Likely hits: {OUT_DIR / 'recent_stock_purchase_hits.csv'}")


if __name__ == "__main__":
    main()