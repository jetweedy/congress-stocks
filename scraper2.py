import io
import re
import csv
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from pypdf import PdfReader

ZIP_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.zip"
BASE_PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{docid}.pdf"

OUT_DIR = Path("house_ptr_monitor")
RAW_DIR = OUT_DIR / "raw"
PDF_DIR = OUT_DIR / "pdfs"
PARSED_DIR = OUT_DIR / "parsed"

for d in [OUT_DIR, RAW_DIR, PDF_DIR, PARSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOOKBACK_DAYS = 30
TIMEOUT = 60

# Heuristic asset classes to keep when looking for "recent stock purchases"
LIKELY_MARKET_ASSET_CODES = {"ST", "OT", "MF", "EF"}
STRICT_STOCK_CODES = {"ST"}

COMMON_ETF_HINTS = [
    "ETF", "SPDR", "ISHARES", "VANGUARD", "INVESCO", "TRUST", "FUND"
]

HEADER_NOISE_PATTERNS = [
    r"^P T R$",
    r"^Clerk of the House of Representatives",
    r"^Name:",
    r"^Status:",
    r"^State/District:",
    r"^ID Owner Asset Transaction$",
    r"^Type$",
    r"^Date Notification$",
    r"^Date$",
    r"^Amount Cap\.$",
    r"^Gains >$",
    r"^\$200\?$",
    r"^Filing ID #",
    r"^\* For the complete list of asset type abbreviations",
    r"^I CERTIFY that",
    r"^Digitally Signed:",
    r"^I V D$",
    r"^I P O$",
    r"^C  S$",
    r"^Yes No$",
    r"^F I$",
    r"^T$",
]

AMOUNT_RE = re.compile(r"\$\d[\d,]*(?:\s*-\s*\$\d[\d,]*)?")
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
ASSET_CODE_RE = re.compile(r"\[([A-Z]{2})\]")
TRANSACTION_START_RE = re.compile(
    r"\b(?P<tx_type>P|S|E|X|C|G|R|I|T)(?:\s+\(partial\))?\s+"
    r"(?P<tx_date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<notif_date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<amount>\$\d[\d,]*(?:\s*-\s*\$\d[\d,]*)?)"
)

@dataclass
class ParsedTransaction:
    docid: str
    report_year: str
    member_name: str
    state_district: str
    filing_date_index: str
    pdf_url: str
    asset_name: str
    ticker: str
    asset_code: str
    tx_type: str
    tx_partial: bool
    tx_date: str
    notification_date: str
    amount_range: str
    owner_code: str
    raw_detail: str
    source_pdf: str

def download_zip_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content

def extract_first_txt_or_csv(zip_bytes: bytes) -> tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.lower().endswith((".txt", ".csv"))]
        if not candidates:
            raise RuntimeError(f"No .txt/.csv file found in ZIP: {names}")
        candidates.sort(key=lambda x: (not x.lower().endswith(".txt"), x.lower()))
        name = candidates[0]
        return name, zf.read(name)

def parse_index_bytes(tabular_bytes: bytes) -> pd.DataFrame:
    text = tabular_bytes.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["FilingDate_dt"] = pd.to_datetime(df["FilingDate"], errors="coerce")
    return df

def build_ptr_url(year: str, docid: str) -> str:
    return BASE_PTR_URL.format(year=year, docid=docid)

def download_pdf(url: str, target_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
            target_path.write_bytes(r.content)
            return True
        return False
    except Exception:
        return False

def extract_pdf_lines(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def clean_lines(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = line.replace("\x00", " ").strip()
        if not line:
            continue
        is_noise = any(re.search(pat, line, flags=re.I) for pat in HEADER_NOISE_PATTERNS)
        if is_noise:
            continue
        cleaned.append(line)
    return cleaned

def extract_member_metadata(lines: List[str]) -> tuple[str, str]:
    member_name = ""
    state_district = ""
    for line in lines:
        if line.startswith("Name:"):
            member_name = line.replace("Name:", "").strip()
        elif line.startswith("State/District:"):
            state_district = line.replace("State/District:", "").strip()
    return member_name, state_district

def parse_ticker(asset_text: str) -> str:
    # Ticker often appears in parentheses before [ST], e.g. "(SPGI) [ST]"
    m = re.search(r"\(([A-Z0-9.\-/]+)\)\s*\[[A-Z]{2}\]", asset_text)
    if m:
        return m.group(1)
    return ""

def looks_like_owner_line(line: str) -> bool:
    return line.startswith("S O:")

def parse_owner_code(line: str) -> str:
    return line.replace("S O:", "").strip()

def parse_transactions_from_lines(
    raw_lines: List[str],
    docid: str,
    year: str,
    filing_date_index: str,
    pdf_url: str,
    pdf_filename: str,
) -> List[ParsedTransaction]:
    member_name, state_district = extract_member_metadata(raw_lines)
    lines = clean_lines(raw_lines)

    transactions = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Find the line that starts the transaction row
        tx_match = TRANSACTION_START_RE.search(line)

        if not tx_match:
            i += 1
            continue

        # Asset block is the lines immediately before the transaction-start line,
        # walking backward until another owner/detail marker or obvious boundary.
        asset_parts = []
        j = i - 1
        while j >= 0:
            prev_line = lines[j]

            if (
                prev_line.startswith(("F S:", "S O:", "D:", "L:"))
                or TRANSACTION_START_RE.search(prev_line)
            ):
                break

            # prepend because we're walking backward
            asset_parts.insert(0, prev_line)

            # Stop once we already have an asset code line
            joined = " ".join(asset_parts)
            if ASSET_CODE_RE.search(joined):
                break

            j -= 1

        asset_text = " ".join(asset_parts).strip()
        asset_code_match = ASSET_CODE_RE.search(asset_text)
        asset_code = asset_code_match.group(1) if asset_code_match else ""
        ticker = parse_ticker(asset_text)

        tx_type = tx_match.group("tx_type")
        tx_partial = "(partial)" in line.lower()
        tx_date = tx_match.group("tx_date")
        notif_date = tx_match.group("notif_date")
        amount_range = tx_match.group("amount")

        owner_code = ""
        raw_detail_parts = []

        k = i + 1
        while k < len(lines):
            next_line = lines[k]

            if TRANSACTION_START_RE.search(next_line):
                break

            # probable start of next asset block:
            # line with [XX] but not a metadata line
            if (
                ASSET_CODE_RE.search(next_line)
                and not next_line.startswith(("F S:", "S O:", "D:", "L:"))
                and not TRANSACTION_START_RE.search(next_line)
            ):
                # likely next asset block
                break

            if looks_like_owner_line(next_line):
                owner_code = parse_owner_code(next_line)
            elif next_line.startswith(("D:", "L:", "F S:")):
                raw_detail_parts.append(next_line)

            k += 1

        transactions.append(
            ParsedTransaction(
                docid=docid,
                report_year=year,
                member_name=member_name,
                state_district=state_district,
                filing_date_index=filing_date_index,
                pdf_url=pdf_url,
                asset_name=asset_text,
                ticker=ticker,
                asset_code=asset_code,
                tx_type=tx_type,
                tx_partial=tx_partial,
                tx_date=tx_date,
                notification_date=notif_date,
                amount_range=amount_range,
                owner_code=owner_code,
                raw_detail=" | ".join(raw_detail_parts),
                source_pdf=pdf_filename,
            )
        )

        i = k
    return transactions

def is_recent_ptr(row: pd.Series, lookback_days: int) -> bool:
    if str(row.get("FilingType", "")).upper().strip() != "P":
        return False
    dt = row.get("FilingDate_dt")
    if pd.isna(dt):
        return False
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=lookback_days))
    return dt >= cutoff

def is_likely_stock_purchase(tx: ParsedTransaction, strict_stock_only: bool = False) -> bool:
    if tx.tx_type != "P":
        return False

    asset_upper = tx.asset_name.upper()

    if strict_stock_only:
        return tx.asset_code in STRICT_STOCK_CODES

    if tx.asset_code in LIKELY_MARKET_ASSET_CODES:
        return True

    return any(term in asset_upper for term in COMMON_ETF_HINTS)

def main():
    print("Downloading annual index ZIP...")
    zip_bytes = download_zip_bytes(ZIP_URL)
    zip_path = RAW_DIR / "2026FD.zip"
    zip_path.write_bytes(zip_bytes)

    print("Extracting text index...")
    txt_name, txt_bytes = extract_first_txt_or_csv(zip_bytes)
    txt_path = RAW_DIR / txt_name
    txt_path.write_bytes(txt_bytes)

    print(f"Parsing index file: {txt_name}")
    df = parse_index_bytes(txt_bytes)

    recent_ptr = df[df.apply(lambda r: is_recent_ptr(r, LOOKBACK_DAYS), axis=1)].copy()
    if recent_ptr.empty:
        print("No recent PTR rows found in index.")
        return

    recent_ptr["pdf_url"] = recent_ptr.apply(
        lambda r: build_ptr_url(str(r["Year"]).strip(), str(r["DocID"]).strip()),
        axis=1,
    )

    all_transactions: List[ParsedTransaction] = []
    pdf_status_rows = []

    for _, row in recent_ptr.sort_values("FilingDate_dt", ascending=False).iterrows():
        docid = str(row["DocID"]).strip()
        year = str(row["Year"]).strip()
        pdf_url = str(row["pdf_url"]).strip()
        filing_date = str(row["FilingDate"]).strip()

        pdf_filename = f"{year}_{docid}.pdf"
        pdf_path = PDF_DIR / pdf_filename

        print(f"Resolving DocID {docid} -> {pdf_url}")
        ok = pdf_path.exists() or download_pdf(pdf_url, pdf_path)

        pdf_status_rows.append({
            "DocID": docid,
            "Year": year,
            "Last": row.get("Last", ""),
            "First": row.get("First", ""),
            "FilingDate": filing_date,
            "pdf_url": pdf_url,
            "pdf_found": ok,
        })

        if not ok:
            continue

        try:
            raw_lines = extract_pdf_lines(pdf_path)
            parsed = parse_transactions_from_lines(
                raw_lines=raw_lines,
                docid=docid,
                year=year,
                filing_date_index=filing_date,
                pdf_url=pdf_url,
                pdf_filename=pdf_filename,
            )
            all_transactions.extend(parsed)
        except Exception as e:
            print(f"Failed parsing {pdf_filename}: {e}")

    pd.DataFrame(pdf_status_rows).to_csv(PARSED_DIR / "pdf_status.csv", index=False)

    if not all_transactions:
        print("No transactions parsed.")
        return

    tx_df = pd.DataFrame([asdict(tx) for tx in all_transactions])
    tx_df.to_csv(PARSED_DIR / "all_recent_ptr_transactions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    purchases_df = tx_df[tx_df["tx_type"] == "P"].copy()
    purchases_df.to_csv(PARSED_DIR / "all_recent_purchases.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    likely_market_df = tx_df[
        tx_df.apply(
            lambda r: is_likely_stock_purchase(
                ParsedTransaction(**r.to_dict()),
                strict_stock_only=False
            ),
            axis=1
        )
    ].copy()
    likely_market_df.to_csv(PARSED_DIR / "recent_likely_stock_purchases.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    strict_stock_df = tx_df[
        (tx_df["tx_type"] == "P") & (tx_df["asset_code"] == "ST")
    ].copy()
    strict_stock_df.to_csv(PARSED_DIR / "recent_strict_stock_purchases.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print("\nDone.")
    print(f"Index file: {txt_path}")
    print(f"PDF status: {PARSED_DIR / 'pdf_status.csv'}")
    print(f"All transactions: {PARSED_DIR / 'all_recent_ptr_transactions.csv'}")
    print(f"All purchases: {PARSED_DIR / 'all_recent_purchases.csv'}")
    print(f"Likely stock purchases: {PARSED_DIR / 'recent_likely_stock_purchases.csv'}")
    print(f"Strict stock purchases: {PARSED_DIR / 'recent_strict_stock_purchases.csv'}")

if __name__ == "__main__":
    main()