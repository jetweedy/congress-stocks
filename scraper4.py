import io
import re
import csv
import json
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
from pypdf import PdfReader

# Optional enrichment
try:
    import yfinance as yf
except ImportError:
    yf = None


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "zip_url": "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.zip",
    "ptr_pdf_url_template": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{docid}.pdf",
    "lookback_days": 30,
    "cluster_window_days": 14,
    "request_timeout": 60,
    "sleep_between_downloads": 0.25,
    "sleep_between_price_queries": 0.25,
    "sleep_between_sector_queries": 0.10,
    "output_dir": "house_ptr_monitor",
    "download_pdfs": True,
    "enable_price_enrichment": True,   # requires yfinance installed
    "enable_sector_enrichment": True,  # requires yfinance installed
    "strict_stock_codes": {"ST"},
    "broad_market_codes": {"ST", "OT", "MF", "EF"},
    "etf_name_hints": [
        "ETF", "SPDR", "ISHARES", "VANGUARD", "INVESCO", "TRUST", "FUND", "INDEX"
    ],
}

OUT_DIR = Path(CONFIG["output_dir"])
RAW_DIR = OUT_DIR / "raw"
PDF_DIR = OUT_DIR / "pdfs"
PARSED_DIR = OUT_DIR / "parsed"
CACHE_DIR = OUT_DIR / "cache"

for d in [OUT_DIR, RAW_DIR, PDF_DIR, PARSED_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class FilingIndexRow:
    prefix: str
    last: str
    first: str
    suffix: str
    filing_type: str
    state_dst: str
    year: str
    filing_date: str
    docid: str
    pdf_url: str


@dataclass
class ParsedTransaction:
    docid: str
    report_year: str
    filing_date_index: str
    member_name: str
    state_district: str
    filing_type: str
    pdf_url: str
    source_pdf: str

    asset_name_raw: str
    asset_name_clean: str
    ticker_raw: str
    ticker_clean: str
    cusip_or_identifier: str
    asset_code: str

    tx_type: str
    tx_partial: bool
    tx_date: str
    notification_date: str
    amount_range: str

    owner_code_raw: str
    owner_code_normalized: str
    owner_category: str

    filing_status_new: Optional[str]
    location_raw: Optional[str]
    detail_raw: Optional[str]

    is_purchase: bool
    is_strict_stock_purchase: bool
    is_broad_market_purchase: bool


# ============================================================
# REGEX / PARSING RULES
# ============================================================

AMOUNT_RE = re.compile(r"\$\d[\d,]*(?:\s*-\s*\$\d[\d,]*)?")
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
ASSET_CODE_RE = re.compile(r"\[([A-Z]{2})\]")
TICKER_PAREN_RE = re.compile(r"\(([A-Z][A-Z0-9.\-\/]{0,14})\)\s*\[[A-Z]{2}\]")
CUSIP_RE = re.compile(r"\(([0-9A-Z]{8,12})\)\s*\[[A-Z]{2}\]")

TRANSACTION_LINE_RE = re.compile(
    r"\b(?P<tx_type>P|S|E|X|C|G|R|I|T)"
    r"(?P<partial>\s+\(partial\))?\s+"
    r"(?P<tx_date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<notif_date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<amount>\$\d[\d,]*(?:\s*-\s*\$\d[\d,]*)?)"
)

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
    r"^I CERTIFY",
    r"^Digitally Signed:",
    r"^I V D$",
    r"^I P O$",
    r"^C  S$",
    r"^Yes No$",
    r"^F I$",
    r"^T$",
]

OWNER_CATEGORY_MAP = {
    "SP": "Spouse",
    "JT": "Joint",
    "DC": "Dependent Child",
    "S": "Self",
}

KNOWN_OWNER_ALIASES = {
    "PUTNAM INVESTMENTS": "Putnam Investments",
    "LIVTR": "LIVTR",
    "SCH1": "SCH1",
}


# ============================================================
# UTILS
# ============================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}")


def safe_get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def clean_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def parse_amount_bounds(amount_range: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    '$1,001 - $15,000' -> (1001, 15000, midpoint)
    """
    if not amount_range:
        return None, None, None

    nums = re.findall(r"\$([\d,]+)", amount_range)
    if not nums:
        return None, None, None

    vals = [float(x.replace(",", "")) for x in nums]
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    if len(vals) >= 2:
        lo, hi = vals[0], vals[1]
        return lo, hi, (lo + hi) / 2.0
    return None, None, None


def normalize_member_name(prefix: str, first: str, last: str, suffix: str) -> str:
    parts = [prefix.strip(), first.strip(), last.strip(), suffix.strip()]
    return clean_whitespace(" ".join([p for p in parts if p]))


def looks_like_noise(line: str) -> bool:
    for pat in HEADER_NOISE_PATTERNS:
        if re.search(pat, line, flags=re.I):
            return True
    return False


def normalize_ticker(raw_ticker: str) -> str:
    t = (raw_ticker or "").strip().upper()
    t = t.replace("/", ".")
    t = re.sub(r"[^A-Z0-9.\-]", "", t)
    return t


def extract_ticker(asset_text: str) -> str:
    m = TICKER_PAREN_RE.search(asset_text)
    if m:
        value = m.group(1)
        if re.fullmatch(r"[0-9A-Z]{8,12}", value):
            return ""
        return normalize_ticker(value)
    return ""


def extract_identifier(asset_text: str) -> str:
    m = CUSIP_RE.search(asset_text)
    return m.group(1) if m else ""


def strip_asset_code_and_ticker(asset_text: str) -> str:
    text = asset_text
    text = re.sub(r"\([A-Z0-9.\-\/]{1,15}\)\s*\[[A-Z]{2}\]", "", text)
    text = re.sub(r"\[[A-Z]{2}\]", "", text)
    text = clean_whitespace(text)
    return text


def normalize_asset_name(asset_text: str) -> str:
    text = strip_asset_code_and_ticker(asset_text)
    text = text.replace(" - Common Stock", "")
    text = text.replace(" Common Stock", "")
    text = clean_whitespace(text)
    return text


def normalize_owner_code(owner_raw: str) -> str:
    value = clean_whitespace(owner_raw).strip(">")
    upper = value.upper()
    if upper in KNOWN_OWNER_ALIASES:
        return KNOWN_OWNER_ALIASES[upper]
    return value


def classify_owner(owner_raw: str) -> str:
    text = (owner_raw or "").upper()
    for code, label in OWNER_CATEGORY_MAP.items():
        if f"(OWNER: {code})" in text or text.strip() == code:
            return label
    if "SPOUSE" in text or text.strip() == "SP":
        return "Spouse"
    if "JOINT" in text or text.strip() == "JT":
        return "Joint"
    if "DEPENDENT" in text or text.strip() == "DC":
        return "Dependent Child"
    if text.strip() in {"", "S"}:
        return "Self/Unknown"
    return "Other/Entity"


def is_recent_date_str(date_str: str, lookback_days: int) -> bool:
    try:
        dt = pd.to_datetime(date_str)
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=lookback_days))
        return dt >= cutoff
    except Exception:
        return False


def is_broad_market_asset(asset_code: str, asset_name_clean: str) -> bool:
    if asset_code in CONFIG["broad_market_codes"]:
        return True
    upper_name = asset_name_clean.upper()
    return any(term in upper_name for term in CONFIG["etf_name_hints"])


def is_strict_stock_asset(asset_code: str) -> bool:
    return asset_code in CONFIG["strict_stock_codes"]


# ============================================================
# ZIP / INDEX
# ============================================================

def download_zip_bytes(url: str) -> bytes:
    log(f"Downloading ZIP index: {url}")
    return safe_get(url, timeout=CONFIG["request_timeout"]).content


def extract_index_file(zip_bytes: bytes) -> Tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.lower().endswith((".txt", ".csv"))]
        if not candidates:
            raise RuntimeError(f"No TXT/CSV file found in ZIP. Contents: {names}")
        candidates.sort(key=lambda x: (not x.lower().endswith(".txt"), x.lower()))
        target = candidates[0]
        return target, zf.read(target)


def parse_index_bytes(tabular_bytes: bytes) -> pd.DataFrame:
    text = tabular_bytes.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df


def build_ptr_pdf_url(year: str, docid: str) -> str:
    return CONFIG["ptr_pdf_url_template"].format(year=year, docid=docid)


def index_to_filing_rows(df: pd.DataFrame) -> List[FilingIndexRow]:
    rows = []
    for _, r in df.iterrows():
        if str(r.get("FilingType", "")).strip().upper() != "P":
            continue

        filing_date = str(r.get("FilingDate", "")).strip()
        if not is_recent_date_str(filing_date, CONFIG["lookback_days"]):
            continue

        year = str(r.get("Year", "")).strip()
        docid = str(r.get("DocID", "")).strip()

        rows.append(
            FilingIndexRow(
                prefix=str(r.get("Prefix", "")).strip(),
                last=str(r.get("Last", "")).strip(),
                first=str(r.get("First", "")).strip(),
                suffix=str(r.get("Suffix", "")).strip(),
                filing_type="P",
                state_dst=str(r.get("StateDst", "")).strip(),
                year=year,
                filing_date=filing_date,
                docid=docid,
                pdf_url=build_ptr_pdf_url(year, docid),
            )
        )
    return rows


# ============================================================
# PDF DOWNLOAD / TEXT EXTRACTION
# ============================================================

def download_pdf(url: str, target_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=CONFIG["request_timeout"])
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
        for raw in text.splitlines():
            line = clean_whitespace(raw)
            if line:
                lines.append(line)
    return lines


def clean_pdf_lines(lines: List[str]) -> List[str]:
    out = []
    for line in lines:
        if not line:
            continue
        if looks_like_noise(line):
            continue
        out.append(line)
    return out


def extract_pdf_header_metadata(raw_lines: List[str]) -> Dict[str, str]:
    meta = {"member_name": "", "state_district": ""}
    for line in raw_lines:
        if line.startswith("Name:"):
            meta["member_name"] = clean_whitespace(line.replace("Name:", ""))
        elif line.startswith("State/District:"):
            meta["state_district"] = clean_whitespace(line.replace("State/District:", ""))
    return meta


# ============================================================
# PTR PARSING
# ============================================================

def parse_transaction_line(line: str) -> Optional[Dict[str, str]]:
    m = TRANSACTION_LINE_RE.search(line)
    if not m:
        return None
    return {
        "tx_type": m.group("tx_type"),
        "tx_partial": bool(m.group("partial")),
        "tx_date": m.group("tx_date"),
        "notification_date": m.group("notif_date"),
        "amount_range": m.group("amount"),
    }


def gather_asset_block(lines: List[str], idx: int) -> str:
    """
    Work backward from the transaction line and collect asset lines.
    """
    parts = []
    j = idx - 1

    while j >= 0:
        prev_line = lines[j]

        if prev_line.startswith(("F S:", "S O:", "D:", "L:")):
            break
        if TRANSACTION_LINE_RE.search(prev_line):
            break

        parts.insert(0, prev_line)
        joined = " ".join(parts)
        if ASSET_CODE_RE.search(joined):
            break

        j -= 1

    return clean_whitespace(" ".join(parts))


def gather_following_metadata(lines: List[str], idx: int) -> Dict[str, Optional[str]]:
    owner = None
    filing_status = None
    location = None
    detail_parts = []

    k = idx + 1
    while k < len(lines):
        next_line = lines[k]

        if TRANSACTION_LINE_RE.search(next_line):
            break

        if (
            ASSET_CODE_RE.search(next_line)
            and not next_line.startswith(("F S:", "S O:", "D:", "L:"))
            and not TRANSACTION_LINE_RE.search(next_line)
        ):
            break

        if next_line.startswith("S O:"):
            owner = clean_whitespace(next_line.replace("S O:", ""))
        elif next_line.startswith("F S:"):
            filing_status = clean_whitespace(next_line.replace("F S:", ""))
        elif next_line.startswith("L:"):
            location = clean_whitespace(next_line.replace("L:", ""))
        elif next_line.startswith("D:"):
            detail_parts.append(clean_whitespace(next_line.replace("D:", "")))
        else:
            if detail_parts:
                detail_parts.append(next_line)

        k += 1

    return {
        "owner_code_raw": owner,
        "filing_status_new": filing_status,
        "location_raw": location,
        "detail_raw": " ".join(detail_parts).strip() if detail_parts else None,
    }


def parse_transactions_from_pdf_lines(
    raw_lines: List[str],
    filing: FilingIndexRow,
    source_pdf_name: str
) -> List[ParsedTransaction]:
    header_meta = extract_pdf_header_metadata(raw_lines)
    lines = clean_pdf_lines(raw_lines)

    transactions: List[ParsedTransaction] = []

    i = 0
    while i < len(lines):
        tx_info = parse_transaction_line(lines[i])
        if not tx_info:
            i += 1
            continue

        asset_block = gather_asset_block(lines, i)
        metadata = gather_following_metadata(lines, i)

        asset_code_match = ASSET_CODE_RE.search(asset_block)
        asset_code = asset_code_match.group(1) if asset_code_match else ""

        ticker_raw = extract_ticker(asset_block)
        identifier = extract_identifier(asset_block)
        asset_name_clean = normalize_asset_name(asset_block)
        owner_code_raw = metadata.get("owner_code_raw") or ""
        owner_code_normalized = normalize_owner_code(owner_code_raw)
        owner_category = classify_owner(owner_code_raw)

        is_purchase = tx_info["tx_type"] == "P"
        is_strict_stock_purchase = is_purchase and is_strict_stock_asset(asset_code)
        is_broad_market_purchase = is_purchase and is_broad_market_asset(asset_code, asset_name_clean)

        transactions.append(
            ParsedTransaction(
                docid=filing.docid,
                report_year=filing.year,
                filing_date_index=filing.filing_date,
                member_name=header_meta["member_name"] or normalize_member_name(
                    filing.prefix, filing.first, filing.last, filing.suffix
                ),
                state_district=header_meta["state_district"] or filing.state_dst,
                filing_type=filing.filing_type,
                pdf_url=filing.pdf_url,
                source_pdf=source_pdf_name,

                asset_name_raw=asset_block,
                asset_name_clean=asset_name_clean,
                ticker_raw=ticker_raw,
                ticker_clean=normalize_ticker(ticker_raw),
                cusip_or_identifier=identifier,
                asset_code=asset_code,

                tx_type=tx_info["tx_type"],
                tx_partial=bool(tx_info["tx_partial"]),
                tx_date=tx_info["tx_date"],
                notification_date=tx_info["notification_date"],
                amount_range=tx_info["amount_range"],

                owner_code_raw=owner_code_raw,
                owner_code_normalized=owner_code_normalized,
                owner_category=owner_category,

                filing_status_new=metadata.get("filing_status_new"),
                location_raw=metadata.get("location_raw"),
                detail_raw=metadata.get("detail_raw"),

                is_purchase=is_purchase,
                is_strict_stock_purchase=is_strict_stock_purchase,
                is_broad_market_purchase=is_broad_market_purchase,
            )
        )

        i += 1

    return transactions


# ============================================================
# OPTIONAL PRICE / SECTOR ENRICHMENT
# ============================================================

def get_price_cache() -> Dict[str, Dict]:
    cache_path = CACHE_DIR / "price_cache.json"
    return load_json(cache_path, default={}) or {}


def save_price_cache(cache: Dict[str, Dict]) -> None:
    save_json(CACHE_DIR / "price_cache.json", cache)


def get_sector_cache() -> Dict[str, Dict]:
    cache_path = CACHE_DIR / "sector_cache.json"
    return load_json(cache_path, default={}) or {}


def save_sector_cache(cache: Dict[str, Dict]) -> None:
    save_json(CACHE_DIR / "sector_cache.json", cache)


def fetch_close_price_yfinance(ticker: str, tx_date: str) -> Optional[float]:
    if yf is None:
        return None
    try:
        dt = datetime.strptime(tx_date, "%m/%d/%Y")
        start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=6)).strftime("%Y-%m-%d")
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if hist is None or hist.empty:
            return None

        hist = hist.reset_index()
        hist["DateOnly"] = pd.to_datetime(hist["Date"]).dt.date
        target_date = dt.date()

        same_day = hist[hist["DateOnly"] == target_date]
        if not same_day.empty:
            return float(same_day.iloc[0]["Close"])

        hist["date_diff"] = hist["DateOnly"].apply(lambda d: abs((d - target_date).days))
        hist = hist.sort_values("date_diff")
        if not hist.empty:
            return float(hist.iloc[0]["Close"])
        return None
    except Exception:
        return None


def fetch_sector_yfinance(ticker: str) -> Optional[str]:
    if yf is None:
        return None
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or "Unknown"
    except Exception:
        return None


def enrich_prices(df: pd.DataFrame) -> pd.DataFrame:
    if not CONFIG["enable_price_enrichment"]:
        return df
    if yf is None:
        print("yfinance not installed; skipping price enrichment.")
        return df

    cache = get_price_cache()
    rows = []

    for _, row in df.iterrows():
        ticker = str(row.get("ticker_clean", "")).strip().upper()
        tx_date = str(row.get("tx_date", "")).strip()

        cache_key = f"{ticker}|{tx_date}"
        close_price = None

        if ticker and tx_date:
            if cache_key in cache:
                close_price = cache[cache_key].get("close_price")
            else:
                close_price = fetch_close_price_yfinance(ticker, tx_date)
                cache[cache_key] = {"close_price": close_price}
                save_price_cache(cache)
                time.sleep(CONFIG["sleep_between_price_queries"])

        lo, hi, mid = parse_amount_bounds(str(row.get("amount_range", "")))
        est_shares_low = None
        est_shares_high = None
        est_shares_mid = None

        if close_price and lo:
            try:
                est_shares_low = lo / close_price
                est_shares_high = hi / close_price if hi else lo / close_price
                est_shares_mid = mid / close_price if mid else lo / close_price
            except Exception:
                pass

        r = row.to_dict()
        r["close_price_near_tx_date"] = close_price
        r["amount_low"] = lo
        r["amount_high"] = hi
        r["amount_mid"] = mid
        r["est_shares_low"] = est_shares_low
        r["est_shares_high"] = est_shares_high
        r["est_shares_mid"] = est_shares_mid
        rows.append(r)

    return pd.DataFrame(rows)


def enrich_sectors(df: pd.DataFrame) -> pd.DataFrame:
    if not CONFIG["enable_sector_enrichment"]:
        df["sector"] = None
        return df
    if yf is None:
        print("yfinance not installed; skipping sector enrichment.")
        df["sector"] = None
        return df

    cache = get_sector_cache()
    sector_map = {}

    unique_tickers = sorted(set(df["ticker_clean"].fillna("").astype(str).str.upper()))
    unique_tickers = [t for t in unique_tickers if t]

    for ticker in unique_tickers:
        if ticker in cache:
            sector_map[ticker] = cache[ticker].get("sector")
        else:
            sector = fetch_sector_yfinance(ticker)
            cache[ticker] = {"sector": sector}
            sector_map[ticker] = sector
            save_sector_cache(cache)
            time.sleep(CONFIG["sleep_between_sector_queries"])

    out = df.copy()
    out["sector"] = out["ticker_clean"].map(lambda t: sector_map.get(str(t).upper(), None))
    return out


# ============================================================
# SIGNAL ENGINE
# ============================================================

def build_signals(df: pd.DataFrame) -> dict:
    out = {}

    df_sig = df.copy()
    df_sig = df_sig[df_sig["is_purchase"] == True].copy()
    df_sig = df_sig[df_sig["ticker_clean"].fillna("") != ""].copy()
    df_sig["tx_date_dt"] = pd.to_datetime(df_sig["tx_date"], errors="coerce")
    df_sig["filing_date_dt"] = pd.to_datetime(df_sig["filing_date_index"], errors="coerce")

    # 1) Clustered buying
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=CONFIG["cluster_window_days"]))
    recent = df_sig[df_sig["tx_date_dt"] >= cutoff].copy()

    clustered = (
        recent.groupby("ticker_clean")
        .agg(
            member_count=("member_name", "nunique"),
            trade_count=("docid", "count"),
            first_tx_date=("tx_date_dt", "min"),
            last_tx_date=("tx_date_dt", "max"),
            members=("member_name", lambda x: ", ".join(sorted(set(x)))),
            sectors=("sector", lambda x: ", ".join(sorted(set([str(v) for v in x if pd.notna(v) and str(v).strip()])))),
        )
        .reset_index()
    )

    clustered = clustered[clustered["member_count"] >= 2].copy()
    clustered = clustered.sort_values(["member_count", "trade_count", "last_tx_date"], ascending=[False, False, False])

    # 2) Repeat buyers
    repeat = (
        df_sig.groupby(["member_name", "ticker_clean"])
        .agg(
            purchase_count=("docid", "count"),
            first_date=("tx_date_dt", "min"),
            last_date=("tx_date_dt", "max"),
            sectors=("sector", lambda x: ", ".join(sorted(set([str(v) for v in x if pd.notna(v) and str(v).strip()])))),
            total_est_amount_mid=("amount_mid", "sum"),
        )
        .reset_index()
    )

    repeat = repeat[repeat["purchase_count"] >= 2].copy()
    repeat = repeat.sort_values(["purchase_count", "last_date", "total_est_amount_mid"], ascending=[False, False, False])

    # 3) Sector trends
    if "sector" in df_sig.columns:
        sector_trends = (
            df_sig.groupby("sector", dropna=False)
            .agg(
                purchase_count=("docid", "count"),
                member_count=("member_name", "nunique"),
                ticker_count=("ticker_clean", "nunique"),
                total_est_amount_mid=("amount_mid", "sum"),
            )
            .reset_index()
            .sort_values(["purchase_count", "member_count", "total_est_amount_mid"], ascending=[False, False, False])
        )
    else:
        sector_trends = pd.DataFrame()

    # 4) Weighted ticker signal score
    ticker_base = (
        df_sig.groupby("ticker_clean")
        .agg(
            member_count=("member_name", "nunique"),
            purchase_count=("docid", "count"),
            repeat_buyer_pairs=("member_name", lambda x: 0),  # placeholder
            latest_tx_date=("tx_date_dt", "max"),
            total_est_amount_mid=("amount_mid", "sum"),
            sector=("sector", lambda x: next((str(v) for v in x if pd.notna(v) and str(v).strip()), "")),
        )
        .reset_index()
    )

    repeat_pairs = (
        df_sig.groupby(["member_name", "ticker_clean"])
        .size()
        .reset_index(name="purchase_count")
    )
    repeat_pairs = repeat_pairs[repeat_pairs["purchase_count"] >= 2]
    repeat_pair_counts = (
        repeat_pairs.groupby("ticker_clean")
        .size()
        .reset_index(name="repeat_buyer_pairs")
    )

    ticker_base = ticker_base.drop(columns=["repeat_buyer_pairs"]).merge(
        repeat_pair_counts, on="ticker_clean", how="left"
    )
    ticker_base["repeat_buyer_pairs"] = ticker_base["repeat_buyer_pairs"].fillna(0)

    ticker_base["days_since_latest_tx"] = (pd.Timestamp(datetime.now()) - ticker_base["latest_tx_date"]).dt.days
    ticker_base["recency_score"] = ticker_base["days_since_latest_tx"].apply(
        lambda d: max(0, CONFIG["cluster_window_days"] - d) if pd.notna(d) else 0
    )

    ticker_base["signal_score"] = (
        ticker_base["member_count"] * 5
        + ticker_base["purchase_count"] * 2
        + ticker_base["repeat_buyer_pairs"] * 4
        + ticker_base["recency_score"] * 0.5
        + ticker_base["total_est_amount_mid"].fillna(0).apply(lambda x: min(x / 50000.0, 10))
    )

    weighted = ticker_base.sort_values(
        ["signal_score", "member_count", "purchase_count", "latest_tx_date"],
        ascending=[False, False, False, False]
    )

    out["clustered_buying"] = clustered
    out["repeat_buyers"] = repeat
    out["sector_trends"] = sector_trends
    out["weighted_ticker_signals"] = weighted

    return out


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    log("Starting pipeline")

    zip_bytes = download_zip_bytes(CONFIG["zip_url"])
    zip_path = RAW_DIR / Path(CONFIG["zip_url"]).name
    zip_path.write_bytes(zip_bytes)

    index_name, index_bytes = extract_index_file(zip_bytes)
    index_path = RAW_DIR / index_name
    index_path.write_bytes(index_bytes)

    log(f"Parsing index file: {index_name}")
    df_index = parse_index_bytes(index_bytes)

    filings = index_to_filing_rows(df_index)
    log(f"Recent PTR filings found: {len(filings)}")

    pdf_status_rows = []
    all_transactions: List[ParsedTransaction] = []

    for filing in sorted(filings, key=lambda x: x.filing_date, reverse=True):
        pdf_filename = f"{filing.year}_{filing.docid}.pdf"
        pdf_path = PDF_DIR / pdf_filename

        pdf_found = False
        if pdf_path.exists():
            pdf_found = True
        elif CONFIG["download_pdfs"]:
            log(f"Downloading PTR PDF {filing.docid}")
            pdf_found = download_pdf(filing.pdf_url, pdf_path)
            time.sleep(CONFIG["sleep_between_downloads"])

        pdf_status_rows.append({
            "docid": filing.docid,
            "year": filing.year,
            "filing_date": filing.filing_date,
            "member_name": normalize_member_name(filing.prefix, filing.first, filing.last, filing.suffix),
            "state_dst": filing.state_dst,
            "pdf_url": filing.pdf_url,
            "pdf_found": pdf_found,
        })

        if not pdf_found:
            continue

        try:
            raw_lines = extract_pdf_lines(pdf_path)
            txs = parse_transactions_from_pdf_lines(raw_lines, filing, pdf_filename)
            all_transactions.extend(txs)
            log(f"Parsed {len(txs)} transactions from {pdf_filename}")
        except Exception as e:
            log(f"Parse failure for {pdf_filename}: {e}")

    pd.DataFrame(pdf_status_rows).to_csv(PARSED_DIR / "pdf_status.csv", index=False)

    if not all_transactions:
        log("No transactions parsed.")
        return

    df_all = pd.DataFrame([asdict(tx) for tx in all_transactions])

    # enrichment
    df_all = enrich_prices(df_all)
    df_all = enrich_sectors(df_all)

    # derived views
    df_purchases = df_all[df_all["is_purchase"] == True].copy()
    df_strict = df_all[df_all["is_strict_stock_purchase"] == True].copy()
    df_broad = df_all[df_all["is_broad_market_purchase"] == True].copy()

    sort_cols = [c for c in ["filing_date_index", "tx_date", "member_name", "ticker_clean"] if c in df_all.columns]
    if sort_cols:
        for d in [df_all, df_purchases, df_strict, df_broad]:
            d.sort_values(sort_cols, ascending=False, inplace=True, ignore_index=True)

    # save base outputs
    df_all.to_csv(PARSED_DIR / "all_recent_ptr_transactions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    df_purchases.to_csv(PARSED_DIR / "all_recent_purchases.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    df_strict.to_csv(PARSED_DIR / "recent_strict_stock_purchases.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    df_broad.to_csv(PARSED_DIR / "recent_broad_market_purchases.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # signals
    signals = build_signals(df_all)
    signals["clustered_buying"].to_csv(PARSED_DIR / "signal_clustered_buying.csv", index=False)
    signals["repeat_buyers"].to_csv(PARSED_DIR / "signal_repeat_buyers.csv", index=False)
    signals["sector_trends"].to_csv(PARSED_DIR / "signal_sector_trends.csv", index=False)
    signals["weighted_ticker_signals"].to_csv(PARSED_DIR / "signal_weighted_ticker_signals.csv", index=False)

    summary = {
        "run_timestamp": now_ts(),
        "lookback_days": CONFIG["lookback_days"],
        "cluster_window_days": CONFIG["cluster_window_days"],
        "recent_ptr_filings": len(filings),
        "parsed_transactions": len(df_all),
        "purchase_rows": len(df_purchases),
        "strict_stock_purchase_rows": len(df_strict),
        "broad_market_purchase_rows": len(df_broad),
        "price_enrichment_enabled": CONFIG["enable_price_enrichment"],
        "sector_enrichment_enabled": CONFIG["enable_sector_enrichment"],
        "yfinance_available": yf is not None,
        "files": {
            "pdf_status": str(PARSED_DIR / "pdf_status.csv"),
            "all_recent_ptr_transactions": str(PARSED_DIR / "all_recent_ptr_transactions.csv"),
            "all_recent_purchases": str(PARSED_DIR / "all_recent_purchases.csv"),
            "recent_strict_stock_purchases": str(PARSED_DIR / "recent_strict_stock_purchases.csv"),
            "recent_broad_market_purchases": str(PARSED_DIR / "recent_broad_market_purchases.csv"),
            "signal_clustered_buying": str(PARSED_DIR / "signal_clustered_buying.csv"),
            "signal_repeat_buyers": str(PARSED_DIR / "signal_repeat_buyers.csv"),
            "signal_sector_trends": str(PARSED_DIR / "signal_sector_trends.csv"),
            "signal_weighted_ticker_signals": str(PARSED_DIR / "signal_weighted_ticker_signals.csv"),
        }
    }
    save_json(PARSED_DIR / "run_summary.json", summary)

    log("Done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_pipeline()