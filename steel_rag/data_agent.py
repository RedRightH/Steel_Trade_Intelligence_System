"""
data_agent.py - India Steel Export Data Agent

Parses 26 monthly TRADESTAT XLSX files (Jan 2024 - Feb 2026) into a
single DataFrame, then uses Groq to answer quantitative trade questions
and generate matplotlib charts.

Data: Commodity-wise iron & steel exports from India, by country.
      Values in USD Million.

Usage:
    from data_agent import query_export_data, load_export_data
    result = query_export_data("Which country had the highest steel export growth in Feb 2026?")

Or run directly: python data_agent.py
"""

import os
import re
import sys
import json
import textwrap
import traceback
from pathlib import Path
from io import StringIO

# ── Region / Continent Map ────────────────────────────────────────────────────
# Format: country_name_as_in_data -> (continent, sub_region)
# Sub-regions follow broadly recognised international groupings:
#   Asia       : East Asia | Southeast Asia | South Asia | West Asia | Central Asia
#   Europe     : Western Europe | Eastern Europe
#   Americas   : North America | Central America & Caribbean | South America
#   Africa     : North Africa | West Africa | East Africa | Central Africa | Southern Africa
#   Oceania    : Oceania
REGION_MAP: dict[str, tuple[str, str]] = {
    # ── East Asia ─────────────────────────────────────────────────────────────
    "CHINA P RP":       ("Asia", "East Asia"),
    "JAPAN":            ("Asia", "East Asia"),
    "KOREA RP":         ("Asia", "East Asia"),
    "TAIWAN":           ("Asia", "East Asia"),
    "HONG KONG":        ("Asia", "East Asia"),
    "MACAO":            ("Asia", "East Asia"),
    "MONGOLIA":         ("Asia", "East Asia"),

    # ── Southeast Asia ────────────────────────────────────────────────────────
    "VIETNAM SOC REP":  ("Asia", "Southeast Asia"),
    "THAILAND":         ("Asia", "Southeast Asia"),
    "MALAYSIA":         ("Asia", "Southeast Asia"),
    "INDONESIA":        ("Asia", "Southeast Asia"),
    "PHILIPPINES":      ("Asia", "Southeast Asia"),
    "SINGAPORE":        ("Asia", "Southeast Asia"),
    "MYANMAR":          ("Asia", "Southeast Asia"),
    "CAMBODIA":         ("Asia", "Southeast Asia"),
    "LAO PD RP":        ("Asia", "Southeast Asia"),
    "BRUNEI":           ("Asia", "Southeast Asia"),
    "TIMOR LESTE":      ("Asia", "Southeast Asia"),

    # ── South Asia ────────────────────────────────────────────────────────────
    "NEPAL":            ("Asia", "South Asia"),
    "BANGLADESH PR":    ("Asia", "South Asia"),
    "SRI LANKA DSR":    ("Asia", "South Asia"),
    "PAKISTAN IR":      ("Asia", "South Asia"),
    "BHUTAN":           ("Asia", "South Asia"),
    "MALDIVES":         ("Asia", "South Asia"),
    "AFGHANISTAN":      ("Asia", "South Asia"),

    # ── West Asia / Middle East ───────────────────────────────────────────────
    "U ARAB EMTS":      ("Asia", "West Asia"),
    "SAUDI ARAB":       ("Asia", "West Asia"),
    "TURKEY":           ("Asia", "West Asia"),
    "IRAN":             ("Asia", "West Asia"),
    "IRAQ":             ("Asia", "West Asia"),
    "ISRAEL":           ("Asia", "West Asia"),
    "JORDAN":           ("Asia", "West Asia"),
    "KUWAIT":           ("Asia", "West Asia"),
    "OMAN":             ("Asia", "West Asia"),
    "QATAR":            ("Asia", "West Asia"),
    "BAHARAIN IS":      ("Asia", "West Asia"),
    "YEMEN REPUBLC":    ("Asia", "West Asia"),
    "SYRIA":            ("Asia", "West Asia"),
    "LEBANON":          ("Asia", "West Asia"),
    "GEORGIA":          ("Asia", "West Asia"),
    "ARMENIA":          ("Asia", "West Asia"),
    "AZERBAIJAN":       ("Asia", "West Asia"),
    "DJIBOUTI":         ("Asia", "West Asia"),

    # ── Central Asia ──────────────────────────────────────────────────────────
    "KAZAKHSTAN":       ("Asia", "Central Asia"),
    "UZBEKISTAN":       ("Asia", "Central Asia"),
    "KYRGHYZSTAN":      ("Asia", "Central Asia"),
    "TAJIKISTAN":       ("Asia", "Central Asia"),
    "TURKMENISTAN":     ("Asia", "Central Asia"),

    # ── Western Europe ────────────────────────────────────────────────────────
    "ITALY":            ("Europe", "Western Europe"),
    "FRANCE":           ("Europe", "Western Europe"),
    "GERMANY":          ("Europe", "Western Europe"),
    "SPAIN":            ("Europe", "Western Europe"),
    "NETHERLAND":       ("Europe", "Western Europe"),
    "NETHERLANDANTIL":  ("Europe", "Western Europe"),
    "BELGIUM":          ("Europe", "Western Europe"),
    "U K":              ("Europe", "Western Europe"),
    "DENMARK":          ("Europe", "Western Europe"),
    "SWEDEN":           ("Europe", "Western Europe"),
    "NORWAY":           ("Europe", "Western Europe"),
    "FINLAND":          ("Europe", "Western Europe"),
    "AUSTRIA":          ("Europe", "Western Europe"),
    "SWITZERLAND":      ("Europe", "Western Europe"),
    "PORTUGAL":         ("Europe", "Western Europe"),
    "GREECE":           ("Europe", "Western Europe"),
    "IRELAND":          ("Europe", "Western Europe"),
    "LUXEMBOURG":       ("Europe", "Western Europe"),
    "ICELAND":          ("Europe", "Western Europe"),
    "MALTA":            ("Europe", "Western Europe"),
    "CYPRUS":           ("Europe", "Western Europe"),
    "LIECHTENSTEIN":    ("Europe", "Western Europe"),
    "MONACO":           ("Europe", "Western Europe"),
    "SAN MARINO":       ("Europe", "Western Europe"),

    # ── Eastern Europe ────────────────────────────────────────────────────────
    "POLAND":           ("Europe", "Eastern Europe"),
    "RUSSIA":           ("Europe", "Eastern Europe"),
    "UKRAINE":          ("Europe", "Eastern Europe"),
    "ROMANIA":          ("Europe", "Eastern Europe"),
    "CZECH REPUBLIC":   ("Europe", "Eastern Europe"),
    "HUNGARY":          ("Europe", "Eastern Europe"),
    "BULGARIA":         ("Europe", "Eastern Europe"),
    "SLOVAK REP":       ("Europe", "Eastern Europe"),
    "CROATIA":          ("Europe", "Eastern Europe"),
    "SERBIA":           ("Europe", "Eastern Europe"),
    "LATVIA":           ("Europe", "Eastern Europe"),
    "LITHUANIA":        ("Europe", "Eastern Europe"),
    "ESTONIA":          ("Europe", "Eastern Europe"),
    "SLOVENIA":         ("Europe", "Eastern Europe"),
    "BELARUS":          ("Europe", "Eastern Europe"),
    "MOLDOVA":          ("Europe", "Eastern Europe"),
    "ALBANIA":          ("Europe", "Eastern Europe"),
    "BOSNIA-HRZGOVIN":  ("Europe", "Eastern Europe"),
    "MONTENEGRO":       ("Europe", "Eastern Europe"),
    "MACEDONIA":        ("Europe", "Eastern Europe"),

    # ── North America ─────────────────────────────────────────────────────────
    "U S A":            ("Americas", "North America"),
    "CANADA":           ("Americas", "North America"),
    "MEXICO":           ("Americas", "North America"),
    "PUERTO RICO":      ("Americas", "North America"),

    # ── Central America & Caribbean ───────────────────────────────────────────
    "GUATEMALA":        ("Americas", "Central America & Caribbean"),
    "COSTA RICA":       ("Americas", "Central America & Caribbean"),
    "PANAMA REPUBLIC":  ("Americas", "Central America & Caribbean"),
    "CUBA":             ("Americas", "Central America & Caribbean"),
    "TRINIDAD":         ("Americas", "Central America & Caribbean"),
    "JAMAICA":          ("Americas", "Central America & Caribbean"),
    "DOMINIC REP":      ("Americas", "Central America & Caribbean"),
    "DOMINICA":         ("Americas", "Central America & Caribbean"),
    "BARBADOS":         ("Americas", "Central America & Caribbean"),
    "BELIZE":           ("Americas", "Central America & Caribbean"),
    "EL SALVADOR":      ("Americas", "Central America & Caribbean"),
    "HONDURAS":         ("Americas", "Central America & Caribbean"),
    "NICARAGUA":        ("Americas", "Central America & Caribbean"),
    "HAITI":            ("Americas", "Central America & Caribbean"),
    "GRENADA":          ("Americas", "Central America & Caribbean"),
    "ST LUCIA":         ("Americas", "Central America & Caribbean"),
    "ST VINCENT":       ("Americas", "Central America & Caribbean"),
    "ST KITT N A":      ("Americas", "Central America & Caribbean"),
    "ANTIGUA":          ("Americas", "Central America & Caribbean"),
    "MONTSERRAT":       ("Americas", "Central America & Caribbean"),
    "ANGUILLA":         ("Americas", "Central America & Caribbean"),
    "ARUBA":            ("Americas", "Central America & Caribbean"),
    "BAHAMAS":          ("Americas", "Central America & Caribbean"),
    "BERMUDA":          ("Americas", "Central America & Caribbean"),
    "CAYMAN IS":        ("Americas", "Central America & Caribbean"),
    "TURKS C IS":       ("Americas", "Central America & Caribbean"),
    "VIRGIN IS US":     ("Americas", "Central America & Caribbean"),
    "BR VIRGN IS":      ("Americas", "Central America & Caribbean"),
    "GUADELOUPE":       ("Americas", "Central America & Caribbean"),
    "MARTINIQUE":       ("Americas", "Central America & Caribbean"),
    "FR GUIANA":        ("Americas", "Central America & Caribbean"),
    "SURINAME":         ("Americas", "Central America & Caribbean"),
    "GUYANA":           ("Americas", "Central America & Caribbean"),
    "US MINOR OUTLYING ISLANDS": ("Americas", "Central America & Caribbean"),

    # ── South America ─────────────────────────────────────────────────────────
    "BRAZIL":           ("Americas", "South America"),
    "ARGENTINA":        ("Americas", "South America"),
    "CHILE":            ("Americas", "South America"),
    "COLOMBIA":         ("Americas", "South America"),
    "PERU":             ("Americas", "South America"),
    "VENEZUELA":        ("Americas", "South America"),
    "ECUADOR":          ("Americas", "South America"),
    "BOLIVIA":          ("Americas", "South America"),
    "URUGUAY":          ("Americas", "South America"),
    "PARAGUAY":         ("Americas", "South America"),

    # ── North Africa ──────────────────────────────────────────────────────────
    "EGYPT A RP":       ("Africa", "North Africa"),
    "ALGERIA":          ("Africa", "North Africa"),
    "MOROCCO":          ("Africa", "North Africa"),
    "TUNISIA":          ("Africa", "North Africa"),
    "LIBYA":            ("Africa", "North Africa"),
    "SUDAN":            ("Africa", "North Africa"),
    "SOUTH SUDAN":      ("Africa", "North Africa"),
    "MAURITANIA":       ("Africa", "North Africa"),
    "SAHARWI A.DM RP":  ("Africa", "North Africa"),

    # ── West Africa ───────────────────────────────────────────────────────────
    "NIGERIA":          ("Africa", "West Africa"),
    "GHANA":            ("Africa", "West Africa"),
    "SENEGAL":          ("Africa", "West Africa"),
    "COTE D' IVOIRE":   ("Africa", "West Africa"),
    "MALI":             ("Africa", "West Africa"),
    "BURKINA FASO":     ("Africa", "West Africa"),
    "GUINEA":           ("Africa", "West Africa"),
    "GUINEA BISSAU":    ("Africa", "West Africa"),
    "SIERRA LEONE":     ("Africa", "West Africa"),
    "LIBERIA":          ("Africa", "West Africa"),
    "TOGO":             ("Africa", "West Africa"),
    "BENIN":            ("Africa", "West Africa"),
    "NIGER":            ("Africa", "West Africa"),
    "CHAD":             ("Africa", "West Africa"),
    "GAMBIA":           ("Africa", "West Africa"),
    "CAPE VERDE IS":    ("Africa", "West Africa"),

    # ── East Africa ───────────────────────────────────────────────────────────
    "KENYA":            ("Africa", "East Africa"),
    "ETHIOPIA":         ("Africa", "East Africa"),
    "TANZANIA REP":     ("Africa", "East Africa"),
    "UGANDA":           ("Africa", "East Africa"),
    "RWANDA":           ("Africa", "East Africa"),
    "BURUNDI":          ("Africa", "East Africa"),
    "SOMALIA":          ("Africa", "East Africa"),
    "ERITREA":          ("Africa", "East Africa"),
    "SEYCHELLES":       ("Africa", "East Africa"),
    "COMOROS":          ("Africa", "East Africa"),
    "MADAGASCAR":       ("Africa", "East Africa"),
    "MAURITIUS":        ("Africa", "East Africa"),
    "MAYOTTE":          ("Africa", "East Africa"),
    "REUNION":          ("Africa", "East Africa"),
    "BRITISH INDIAN":   ("Africa", "East Africa"),

    # ── Central Africa ────────────────────────────────────────────────────────
    "CAMEROON":         ("Africa", "Central Africa"),
    "ANGOLA":           ("Africa", "Central Africa"),
    "CONGO P REP":      ("Africa", "Central Africa"),
    "CONGO D. REP.":    ("Africa", "Central Africa"),
    "C AFRI REP":       ("Africa", "Central Africa"),
    "GABON":            ("Africa", "Central Africa"),
    "EQUTL GUINEA":     ("Africa", "Central Africa"),
    "SAO TOME":         ("Africa", "Central Africa"),

    # ── Southern Africa ───────────────────────────────────────────────────────
    "SOUTH AFRICA":     ("Africa", "Southern Africa"),
    "MOZAMBIQUE":       ("Africa", "Southern Africa"),
    "ZAMBIA":           ("Africa", "Southern Africa"),
    "ZIMBABWE":         ("Africa", "Southern Africa"),
    "BOTSWANA":         ("Africa", "Southern Africa"),
    "NAMIBIA":          ("Africa", "Southern Africa"),
    "MALAWI":           ("Africa", "Southern Africa"),
    "LESOTHO":          ("Africa", "Southern Africa"),
    "SWAZILAND":        ("Africa", "Southern Africa"),

    # ── Oceania ───────────────────────────────────────────────────────────────
    "AUSTRALIA":        ("Oceania", "Oceania"),
    "NEW ZEALAND":      ("Oceania", "Oceania"),
    "PAPUA N GNA":      ("Oceania", "Oceania"),
    "FIJI IS":          ("Oceania", "Oceania"),
    "SOLOMON IS":       ("Oceania", "Oceania"),
    "VANUATU REP":      ("Oceania", "Oceania"),
    "SAMOA":            ("Oceania", "Oceania"),
    "TONGA":            ("Oceania", "Oceania"),
    "KIRIBATI REP":     ("Oceania", "Oceania"),
    "TUVALU":           ("Oceania", "Oceania"),
    "MICRONESIA":       ("Oceania", "Oceania"),
    "MARSHALL ISLAND":  ("Oceania", "Oceania"),
    "NEW CALEDONIA":    ("Oceania", "Oceania"),
    "FR POLYNESIA":     ("Oceania", "Oceania"),
}

def get_country_region(country: str) -> tuple[str, str]:
    """Return (continent, sub_region) for a country. Returns ('Other', 'Other') if unmapped."""
    return REGION_MAP.get(country.upper(), ("Other", "Other"))

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
EXPORTS_DIR  = Path(__file__).parent.parent / "Base documents" / "India_Steel_exports"
IMPORTS_DIR  = Path(__file__).parent.parent / "Base documents" / "India_Steel_imports"
CHARTS_DIR   = Path(__file__).parent / "charts"
GROQ_MODEL   = "llama-3.3-70b-versatile"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
# ─────────────────────────────────────────────────────────────────────────────

_df_exports: pd.DataFrame | None = None
_df_imports: pd.DataFrame | None = None
_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── XLSX Parser ───────────────────────────────────────────────────────────────

def _parse_month_year_from_filename(filename: str) -> tuple[int, int]:
    """
    Extract (month_num, year) from filenames like:
      TradeStat-..._feb26.xlsx  -> (2, 2026)
      TradeStat-..._sept24.xlsx -> (9, 2024)
    """
    stem = Path(filename).stem.lower()
    # last token after final underscore: e.g. "feb26", "sept24"
    token = stem.split("_")[-1]
    match = re.match(r"([a-z]+)(\d{2})$", token)
    if not match:
        raise ValueError(f"Cannot parse month/year from filename: {filename}")
    mon_str, yr_str = match.group(1), match.group(2)
    month_num = MONTH_MAP.get(mon_str)
    if month_num is None:
        raise ValueError(f"Unknown month abbreviation: {mon_str}")
    year = 2000 + int(yr_str)
    return month_num, year


def _fiscal_year(month_num: int, year: int) -> str:
    """
    Indian fiscal year runs Apr-Mar.
    Feb 2026 → FY2025-26, Apr 2025 → FY2025-26
    """
    if month_num >= 4:
        return f"FY{year}-{str(year+1)[-2:]}"
    else:
        return f"FY{year-1}-{str(year)[-2:]}"


def _parse_single_xlsx(filepath: Path) -> pd.DataFrame:
    """
    Parse one TRADESTAT XLSX file into a clean DataFrame.

    Returns columns:
        report_month_num, report_year, fiscal_year, country,
        monthly_curr_usd, monthly_prev_usd, monthly_growth_pct, monthly_share_pct,
        ytd_curr_usd, ytd_prev_usd, ytd_growth_pct, ytd_share_pct,
        source_file
    """
    month_num, year = _parse_month_year_from_filename(filepath.name)
    fy = _fiscal_year(month_num, year)

    raw = pd.read_excel(filepath, sheet_name=0, header=None)

    # Data rows: skip rows 0,1,2 (title, metadata, header) and last row (Commodity Total)
    data = raw.iloc[3:-1].copy()
    data.columns = [
        "sno", "country",
        "monthly_prev_usd", "monthly_prev_share",
        "monthly_curr_usd", "monthly_curr_share",
        "monthly_growth_pct",
        "ytd_prev_usd", "ytd_prev_share",
        "ytd_curr_usd", "ytd_curr_share",
        "ytd_growth_pct",
    ]

    # Keep only country + numeric cols
    keep = ["country", "monthly_curr_usd", "monthly_prev_usd", "monthly_growth_pct",
            "monthly_curr_share", "ytd_curr_usd", "ytd_prev_usd", "ytd_growth_pct",
            "ytd_curr_share"]
    df = data[keep].copy()

    # Clean country names
    df["country"] = df["country"].astype(str).str.strip().str.upper()

    # Coerce numeric columns — "-" and blanks become NaN
    num_cols = [c for c in df.columns if c != "country"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no country or all-zero data
    df = df[df["country"].notna() & (df["country"] != "")]
    df = df[df["country"] != "NAN"]

    # Add metadata
    df["report_month_num"] = month_num
    df["report_year"]      = year
    df["fiscal_year"]      = fy
    df["source_file"]      = filepath.name

    return df


def load_export_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load and combine all TRADESTAT XLSX files.
    Cached after first load.

    Returns a DataFrame with columns:
        report_month_num, report_year, fiscal_year, country,
        monthly_curr_usd, monthly_prev_usd, monthly_growth_pct, monthly_curr_share,
        ytd_curr_usd, ytd_prev_usd, ytd_growth_pct, ytd_curr_share, source_file
    """
    global _df_exports
    if _df_exports is not None and not force_reload:
        return _df_exports

    xlsx_files = sorted(EXPORTS_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found in {EXPORTS_DIR}")

    print(f"Loading {len(xlsx_files)} XLSX files from {EXPORTS_DIR.name}/...")
    frames = []
    for f in xlsx_files:
        try:
            frames.append(_parse_single_xlsx(f))
        except Exception as e:
            print(f"  [WARN] Skipping {f.name}: {e}")

    _df_exports = pd.concat(frames, ignore_index=True)

    # Sort by year, month
    _df_exports = _df_exports.sort_values(
        ["report_year", "report_month_num", "country"]
    ).reset_index(drop=True)

    print(f"  Loaded {len(_df_exports):,} rows | "
          f"{_df_exports['source_file'].nunique()} months | "
          f"{_df_exports['country'].nunique()} countries")
    return _df_exports


# ── Data summary for LLM context ──────────────────────────────────────────────

def _build_data_summary(df: pd.DataFrame) -> str:
    """Build a compact text summary of the DataFrame for the LLM system prompt."""
    months = df[["report_year", "report_month_num"]].drop_duplicates().sort_values(
        ["report_year", "report_month_num"]
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_list = [f"{month_names[r.report_month_num]}{r.report_year}"
                  for r in months.itertuples()]

    countries = sorted(df["country"].unique().tolist())
    fys = sorted(df["fiscal_year"].unique().tolist())

    latest = df[df["report_year"] == df["report_year"].max()]
    latest = latest[latest["report_month_num"] == latest["report_month_num"].max()]
    top5 = (latest.groupby("country")["monthly_curr_usd"]
            .sum().nlargest(5).round(2).to_dict())

    return f"""DataFrame name: df
Shape: {len(df):,} rows x {len(df.columns)} columns
Period: {month_list[0]} to {month_list[-1]} ({len(month_list)} monthly snapshots)
Fiscal years: {', '.join(fys)}
Countries: {len(countries)} unique

Columns:
  report_month_num  int   (1=Jan .. 12=Dec)
  report_year       int   (2024, 2025, 2026)
  fiscal_year       str   ('FY2024-25', 'FY2025-26')
  country           str   (UPPER CASE, e.g. 'CHINA PR', 'USA', 'VIETNAM SOC REP')
  monthly_curr_usd  float USD Million - exports in the report month (current year)
  monthly_prev_usd  float USD Million - exports same month previous year
  monthly_growth_pct float % YoY growth in monthly exports (NaN if no prev data)
  monthly_curr_share float % share of total monthly exports
  ytd_curr_usd      float USD Million - Apr to report_month cumulative (current FY)
  ytd_prev_usd      float USD Million - Apr to report_month cumulative (prev FY)
  ytd_growth_pct    float % YoY growth in YTD exports
  ytd_curr_share    float % share of total YTD exports
  source_file       str   original filename

Top 5 export destinations (latest month, USD Mn): {top5}

Notes:
- Values in USD Million. Use .sum() for totals, .mean() for averages.
- Missing data (no trade) appears as NaN — use .dropna() or fillna(0) appropriately.
- For "latest month" filter: df[(df.report_year==df.report_year.max()) & (df.report_month_num==df[df.report_year==df.report_year.max()].report_month_num.max())]
- For FY totals: use ytd_curr_usd from the last month of the FY (March = month 3 or last available).
- Charts: use matplotlib. Save to '{CHARTS_DIR}/<name>.png'. Return chart_path in your JSON.
"""


# ── Code-generation agent ─────────────────────────────────────────────────────

DATA_AGENT_SYSTEM = """You are a Python data analyst specialising in India's steel export trade data.
You have access to a pandas DataFrame called `df` with monthly TRADESTAT export data.

Your job:
1. Write Python code using pandas (and matplotlib if a chart is needed) to answer the question.
2. The code runs in a sandbox with: pandas as pd, numpy as np, matplotlib.pyplot as plt, df (the DataFrame).
3. Set `answer` (str) to your final text answer in the code.
4. If generating a chart, save to `CHART_PATH` and set `chart_generated = True`.

Return your response in EXACTLY this format (use the delimiters verbatim):

DESCRIPTION: <one sentence describing what the code does>
NEEDS_CHART: true or false
CODE:
<your python code here, no markdown fences>

Rules for correctness:
- Use df column names exactly as documented.
- Handle NaN with fillna(0) or dropna() as appropriate.
- `monthly_curr_usd` = exports in the report month (current year). Use this for monthly values.
- For a COUNTRY'S TREND over time: filter df by country, sort by (report_year, report_month_num), plot monthly_curr_usd. Each row is one month's data point.
- For TOTAL INDIA exports per month: group by (report_year, report_month_num) and sum monthly_curr_usd.
- For X-axis labels on time series: create a 'period' column = report_year*100 + report_month_num (e.g. 202502 for Feb 2025) for sorting, then format as "Jan 25", "Feb 25" etc. for display.
- For GROWING/SHRINKING markets: use numpy polyfit(t, values, 1)[0] to get slope over time. Sort by slope descending for growing, ascending for shrinking.
- For charts: plt.figure(figsize=(12,6)), add title, x/y labels, legend if multiple series, grid(True, alpha=0.4), tight_layout(), save to CHART_PATH.
- The variable `answer` MUST be a plain string summarising key findings (numbers, country names).
- Do NOT wrap code in ``` fences.
- import numpy as np inside the code if needed.
"""


def _execute_code(code: str, df: pd.DataFrame, chart_path: Path) -> dict:
    """
    Safely execute LLM-generated pandas/matplotlib code.
    Returns {answer, chart_generated, error}.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np
    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "plt": plt,
        "np": np,
        "CHART_PATH": str(chart_path),
        "answer": "No answer generated.",
        "chart_generated": False,
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    error = None
    try:
        exec(compile(code, "<data_agent>", "exec"), local_vars)
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        plt.close("all")

    stdout_output = captured.getvalue().strip()

    return {
        "answer":          str(local_vars.get("answer", "No answer generated.")),
        "chart_generated": bool(local_vars.get("chart_generated", False)),
        "stdout":          stdout_output,
        "error":           error,
    }


def query_export_data(question: str, save_chart_as: str | None = None) -> dict:
    """
    Answer a quantitative question about India's steel exports.

    Args:
        question:       Natural language question
        save_chart_as:  Filename stem for chart (e.g. "top10_feb26").
                        Auto-generated if None.

    Returns:
        {
          question, answer, chart_path (or None), needs_chart,
          code_used, description, error (or None)
        }
    """
    df = load_export_data()
    data_summary = _build_data_summary(df)

    # Build chart path
    safe_stem = re.sub(r"[^\w]", "_", question[:40].lower())
    chart_name = save_chart_as or safe_stem
    chart_path = CHARTS_DIR / f"{chart_name}.png"

    # Ask LLM to generate code
    user_msg = f"""DATA SCHEMA:
{data_summary}

QUESTION: {question}

Write pandas code to answer this question. If a bar/line chart would help visualise the answer, include it."""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": DATA_AGENT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse delimiter-based format:
        #   DESCRIPTION: ...
        #   NEEDS_CHART: true/false
        #   CODE:
        #   <python code>
        description  = ""
        needs_chart  = False
        code         = ""

        desc_match = re.search(r"DESCRIPTION:\s*(.+)", raw)
        if desc_match:
            description = desc_match.group(1).strip()

        chart_match = re.search(r"NEEDS_CHART:\s*(true|false)", raw, re.IGNORECASE)
        if chart_match:
            needs_chart = chart_match.group(1).lower() == "true"

        code_match = re.search(r"CODE:\s*\n(.*)", raw, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # Strip any trailing markdown fence
            code = re.sub(r"\n```\s*$", "", code).strip()

        if not code:
            raise ValueError(f"No CODE section found in response: {raw[:300]}")

    except Exception as e:
        return {
            "question": question, "answer": f"LLM error: {e}",
            "chart_path": None, "needs_chart": False,
            "code_used": "", "description": "", "error": str(e),
        }

    # Execute the generated code
    exec_result = _execute_code(code, df, chart_path)

    final_chart = str(chart_path) if (exec_result["chart_generated"] and chart_path.exists()) else None

    return {
        "question":    question,
        "answer":      exec_result["answer"],
        "chart_path":  final_chart,
        "needs_chart": needs_chart,
        "code_used":   code,
        "description": description,
        "error":       exec_result["error"],
        "stdout":      exec_result.get("stdout", ""),
    }


# ── Trend analytics helpers (no LLM needed — deterministic) ──────────────────

def _month_label(month_num: int, year: int) -> str:
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return f"{names[month_num]} {year}"


def get_country_trend(country: str) -> pd.DataFrame:
    """
    Return month-by-month export time series for a single country.

    Returns DataFrame with columns:
        period (str), report_year, report_month_num,
        monthly_usd, monthly_growth_pct, ytd_usd, ytd_growth_pct
    Sorted oldest → newest.
    """
    df = load_export_data()
    mask = df["country"].str.upper() == country.upper()
    if not mask.any():
        # fuzzy match — find closest country name
        candidates = df["country"].unique()
        matches = [c for c in candidates if country.upper() in c]
        if matches:
            mask = df["country"].isin(matches)
        else:
            raise ValueError(f"Country '{country}' not found. Try uppercase, e.g. 'VIETNAM SOC REP'")

    sub = (df[mask]
           .sort_values(["report_year", "report_month_num"])
           .copy())
    sub["period"] = sub.apply(
        lambda r: _month_label(int(r.report_month_num), int(r.report_year)), axis=1
    )
    return sub[["period", "report_year", "report_month_num",
                "monthly_curr_usd", "monthly_growth_pct",
                "ytd_curr_usd", "ytd_growth_pct"]].rename(
        columns={"monthly_curr_usd": "monthly_usd",
                 "ytd_curr_usd":     "ytd_usd"})


def get_market_trends(lookback_months: int = 6, min_avg_usd: float = 1.0,
                      n: int = 10) -> dict:
    """
    Identify growing and shrinking markets using linear trend slope
    over the last `lookback_months` months.

    Args:
        lookback_months: How many recent months to use for trend calculation.
        min_avg_usd:     Minimum average monthly export (USD Mn) to be included
                         (filters out near-zero noise).
        n:               Number of top growing / shrinking markets to return.

    Returns:
        {
          "period":           "Aug 2025 - Feb 2026"  (the lookback window),
          "growing":          DataFrame [country, avg_usd, slope, growth_pct_latest],
          "shrinking":        DataFrame [country, avg_usd, slope, growth_pct_latest],
          "all_trends":       DataFrame with slope for every country,
        }
    """
    from numpy.polynomial import polynomial as P

    df = load_export_data()

    # Get the last `lookback_months` unique (year, month) pairs
    periods = (df[["report_year", "report_month_num"]]
               .drop_duplicates()
               .sort_values(["report_year", "report_month_num"])
               .tail(lookback_months))
    # tuples are (month_num, year) — matching _month_label signature
    period_tuples = list(zip(periods["report_month_num"].astype(int),
                             periods["report_year"].astype(int)))

    window = df[df.apply(
        lambda r: (int(r.report_month_num), int(r.report_year)) in period_tuples, axis=1
    )].copy()

    # Create a sequential time index (0, 1, 2, ...) for regression
    period_idx = {t: i for i, t in enumerate(period_tuples)}
    window["t"] = window.apply(
        lambda r: period_idx[(int(r.report_month_num), int(r.report_year))], axis=1
    )

    results = []
    for country, grp in window.groupby("country"):
        grp = grp.sort_values("t")
        vals = grp["monthly_curr_usd"].fillna(0).values
        avg  = vals.mean()
        if avg < min_avg_usd:
            continue

        # Linear slope via numpy polyfit (degree 1)
        t_vals = grp["t"].values
        if len(t_vals) < 2:
            continue
        try:
            import numpy as np
            slope = float(np.polyfit(t_vals, vals, 1)[0])
        except Exception:
            slope = 0.0

        latest_growth = grp["monthly_growth_pct"].iloc[-1]

        results.append({
            "country":            country,
            "avg_monthly_usd":    round(avg, 2),
            "trend_slope":        round(slope, 3),   # USD Mn / month
            "growth_pct_latest":  round(float(latest_growth), 1) if pd.notna(latest_growth) else None,
        })

    all_trends = pd.DataFrame(results).sort_values("trend_slope", ascending=False)

    start_label = _month_label(*period_tuples[0])
    end_label   = _month_label(*period_tuples[-1])

    return {
        "period":     f"{start_label} - {end_label}",
        "growing":    all_trends.nlargest(n, "trend_slope").reset_index(drop=True),
        "shrinking":  all_trends.nsmallest(n, "trend_slope").reset_index(drop=True),
        "all_trends": all_trends.reset_index(drop=True),
    }


def plot_country_comparison(countries: list[str],
                            save_as: str = "country_comparison") -> str:
    """
    Plot monthly export trends for multiple countries on one chart.

    Args:
        countries:  List of country names (case-insensitive, partial match OK).
        save_as:    Filename stem for the chart.

    Returns:
        Path to saved chart PNG.
    """
    df = load_export_data()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    plotted = []
    for country in countries:
        mask = df["country"].str.upper().str.contains(country.upper())
        sub  = df[mask].groupby(["report_year", "report_month_num"])["monthly_curr_usd"].sum().reset_index()
        sub  = sub.sort_values(["report_year", "report_month_num"])
        if sub.empty:
            continue
        labels = [f"{month_names[int(r.report_month_num)]}\n{str(int(r.report_year))[2:]}"
                  for r in sub.itertuples()]
        ax.plot(range(len(sub)), sub["monthly_curr_usd"].values,
                marker="o", linewidth=2, markersize=4, label=country.title())
        if not plotted:
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(labels, fontsize=8, rotation=45)
        plotted.append(country)

    ax.set_title(f"India Steel Exports: {', '.join(c.title() for c in plotted)}", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("USD Million")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = CHARTS_DIR / f"{save_as}.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    return str(out)


def compare_countries(countries: list[str],
                      save_as: str = "country_comparison") -> dict:
    """
    Full country-by-country comparison: stats table + two charts.

    Charts:
      1. Line chart — monthly export trend for each country (all months).
      2. Bar chart  — latest month side-by-side.

    Returns:
        {
          "chart_trend":  path to line chart PNG,
          "chart_latest": path to bar chart PNG,
          "stats":        DataFrame with per-country summary stats,
        }
    """
    import numpy as np

    df = load_export_data()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    # Resolve countries (case-insensitive, partial match)
    resolved = {}
    for c in countries:
        mask = df["country"].str.upper().str.contains(c.upper(), regex=False)
        if mask.any():
            # Use the most common exact match
            best = df[mask]["country"].value_counts().index[0]
            resolved[c] = best
        else:
            resolved[c] = None

    stats_rows = []

    # ── Chart 1: Monthly trend line chart ────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(13, 6))
    all_periods = []

    for label, exact in resolved.items():
        if exact is None:
            continue
        sub = (df[df["country"] == exact]
               .groupby(["report_year", "report_month_num"])["monthly_curr_usd"]
               .sum().reset_index()
               .sort_values(["report_year", "report_month_num"]))
        if sub.empty:
            continue

        periods = [f"{month_names[int(r.report_month_num)]}\n{str(int(r.report_year))[2:]}"
                   for r in sub.itertuples()]
        vals = sub["monthly_curr_usd"].fillna(0).values

        ax1.plot(range(len(sub)), vals, marker="o", linewidth=2,
                 markersize=4, label=exact.title())
        if len(periods) > len(all_periods):
            all_periods = periods

        # Compute stats
        t = np.arange(len(vals))
        slope = float(np.polyfit(t, vals, 1)[0]) if len(vals) > 1 else 0.0
        latest_growth = sub.merge(
            df[df["country"] == exact][["report_year", "report_month_num",
                                        "monthly_growth_pct"]],
            on=["report_year", "report_month_num"]
        )["monthly_growth_pct"].iloc[-1] if not sub.empty else None

        continent, subregion = get_country_region(exact)
        stats_rows.append({
            "country":         exact,
            "continent":       continent,
            "sub_region":      subregion,
            "avg_monthly_usd": round(float(vals.mean()), 2),
            "peak_usd":        round(float(vals.max()), 2),
            "latest_usd":      round(float(vals[-1]), 2),
            "trend_slope":     round(slope, 3),
            "direction":       "Growing" if slope > 0.2 else ("Shrinking" if slope < -0.2 else "Stable"),
            "latest_yoy_pct":  round(float(latest_growth), 1) if pd.notna(latest_growth) else None,
        })

    if all_periods:
        ax1.set_xticks(range(len(all_periods)))
        ax1.set_xticklabels(all_periods, fontsize=8)
    ax1.set_title("India Steel Exports — Country Trend Comparison", fontsize=13)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("USD Million")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    trend_path = CHARTS_DIR / f"{save_as}_trend.png"
    plt.savefig(str(trend_path), dpi=120)
    plt.close()

    # ── Chart 2: Latest month bar chart ──────────────────────────────────────
    stats_df = pd.DataFrame(stats_rows).sort_values("latest_usd", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ["#27ae60" if d == "Growing" else ("#e74c3c" if d == "Shrinking" else "#3498db")
              for d in stats_df["direction"]]
    bars = ax2.bar(stats_df["country"].str.title(), stats_df["latest_usd"], color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, stats_df["latest_usd"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"${val:.1f}M", ha="center", va="bottom", fontsize=9)

    ax2.set_title("Latest Month Steel Exports by Country\n"
                  "(Green = Growing trend | Red = Shrinking | Blue = Stable)", fontsize=11)
    ax2.set_ylabel("USD Million")
    ax2.set_xlabel("Country")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    latest_path = CHARTS_DIR / f"{save_as}_latest.png"
    plt.savefig(str(latest_path), dpi=120)
    plt.close()

    return {
        "chart_trend":  str(trend_path),
        "chart_latest": str(latest_path),
        "stats":        stats_df.reset_index(drop=True),
    }


def get_regional_summary(period: str = "latest_month") -> pd.DataFrame:
    """
    Aggregate exports by continent and sub-region.

    Args:
        period: "latest_month" | "ytd" | "all_time"

    Returns DataFrame with columns:
        continent, sub_region, usd_million, country_count,
        top_country, top_country_usd
    """
    df = load_export_data()

    # Add region columns
    df = df.copy()
    df["continent"]  = df["country"].map(lambda c: REGION_MAP.get(c, ("Other", "Other"))[0])
    df["sub_region"] = df["country"].map(lambda c: REGION_MAP.get(c, ("Other", "Other"))[1])

    if period == "latest_month":
        yr  = df["report_year"].max()
        mo  = df[df["report_year"] == yr]["report_month_num"].max()
        sub = df[(df["report_year"] == yr) & (df["report_month_num"] == mo)]
        val_col = "monthly_curr_usd"
    elif period == "ytd":
        yr  = df["report_year"].max()
        mo  = df[df["report_year"] == yr]["report_month_num"].max()
        sub = df[(df["report_year"] == yr) & (df["report_month_num"] == mo)]
        val_col = "ytd_curr_usd"
    else:  # all_time: sum monthly_curr_usd across all report months
        sub = df
        val_col = "monthly_curr_usd"

    rows = []
    for (cont, reg), grp in sub.groupby(["continent", "sub_region"]):
        total  = grp[val_col].sum()
        top    = grp.groupby("country")[val_col].sum().idxmax()
        top_v  = grp.groupby("country")[val_col].sum().max()
        rows.append({
            "continent":       cont,
            "sub_region":      reg,
            "usd_million":     round(float(total), 2),
            "country_count":   grp["country"].nunique(),
            "top_country":     top,
            "top_country_usd": round(float(top_v), 2),
        })

    return (pd.DataFrame(rows)
            .sort_values(["continent", "usd_million"], ascending=[True, False])
            .reset_index(drop=True))


def plot_regional_breakdown(period: str = "latest_month",
                            save_as: str = "regional_breakdown") -> dict:
    """
    Two-panel chart:
      Left:  Continent-level pie / bar.
      Right: Sub-region breakdown within each continent.

    Returns {"chart_path": str, "summary": DataFrame}
    """
    summary = get_regional_summary(period)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: Continent totals (horizontal bar) ───────────────────────────────
    cont = (summary.groupby("continent")["usd_million"]
            .sum().sort_values(ascending=True))
    colors_cont = {
        "Asia":     "#3498db",
        "Europe":   "#2ecc71",
        "Americas": "#e67e22",
        "Africa":   "#e74c3c",
        "Oceania":  "#9b59b6",
        "Other":    "#95a5a6",
    }
    bar_colors = [colors_cont.get(c, "#95a5a6") for c in cont.index]
    bars = ax1.barh(cont.index, cont.values, color=bar_colors)
    for bar, val in zip(bars, cont.values):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"${val:.1f}M", va="center", fontsize=9)
    ax1.set_title("Exports by Continent", fontsize=12)
    ax1.set_xlabel("USD Million")

    # ── Right: Sub-region stacked view ───────────────────────────────────────
    pivot = summary.pivot_table(
        index="continent", columns="sub_region",
        values="usd_million", aggfunc="sum", fill_value=0
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    import numpy as np
    bottom = np.zeros(len(pivot))
    cmap   = plt.cm.get_cmap("tab20", len(pivot.columns))
    for i, col in enumerate(pivot.columns):
        ax2.bar(pivot.index, pivot[col].values, bottom=bottom,
                label=col, color=cmap(i))
        bottom += pivot[col].values

    ax2.set_title("Sub-region Breakdown by Continent", fontsize=12)
    ax2.set_ylabel("USD Million")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    plt.xticks(rotation=15)

    period_label = {"latest_month": "Latest Month", "ytd": "YTD", "all_time": "All Time"}.get(period, period)
    fig.suptitle(f"India Steel Exports — Regional Breakdown  ({period_label})", fontsize=14)
    plt.tight_layout()

    out = CHARTS_DIR / f"{save_as}.png"
    plt.savefig(str(out), dpi=120)
    plt.close()

    return {"chart_path": str(out), "summary": summary}


def plot_market_trends(lookback_months: int = 6, n: int = 10,
                       save_as: str = "market_trends") -> dict:
    """
    Generate a side-by-side bar chart of top growing vs shrinking markets.

    Returns:
        {"chart_path": str, "growing": DataFrame, "shrinking": DataFrame, "period": str}
    """
    trends = get_market_trends(lookback_months=lookback_months, n=n)
    growing  = trends["growing"]
    shrinking = trends["shrinking"]

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Growing markets
    ax1.barh(growing["country"][::-1], growing["trend_slope"][::-1], color="#2ecc71")
    ax1.set_title(f"Top {n} Growing Markets\n(USD Mn/month slope)", fontsize=11)
    ax1.set_xlabel("Export Trend Slope (USD Mn/month)")
    ax1.axvline(0, color="black", linewidth=0.8)

    # Shrinking markets
    ax2.barh(shrinking["country"], shrinking["trend_slope"], color="#e74c3c")
    ax2.set_title(f"Top {n} Shrinking Markets\n(USD Mn/month slope)", fontsize=11)
    ax2.set_xlabel("Export Trend Slope (USD Mn/month)")
    ax2.axvline(0, color="black", linewidth=0.8)

    fig.suptitle(f"India Steel Export Market Trends  |  {trends['period']}", fontsize=13)
    plt.tight_layout()

    out = CHARTS_DIR / f"{save_as}.png"
    plt.savefig(str(out), dpi=120)
    plt.close()

    return {
        "chart_path": str(out),
        "period":     trends["period"],
        "growing":    growing,
        "shrinking":  shrinking,
    }


# ── Quick stats helpers (no LLM needed) ───────────────────────────────────────

def get_latest_top_destinations(n: int = 10) -> pd.DataFrame:
    """Return top-n export destinations for the latest available month."""
    df = load_export_data()
    latest_year  = df["report_year"].max()
    latest_month = df[df["report_year"] == latest_year]["report_month_num"].max()
    latest = df[(df["report_year"] == latest_year) & (df["report_month_num"] == latest_month)]
    return (latest.groupby("country")["monthly_curr_usd"]
            .sum().nlargest(n).reset_index()
            .rename(columns={"monthly_curr_usd": "usd_million"}))


def get_yoy_summary() -> dict:
    """Return overall YoY export growth for latest month and YTD."""
    df = load_export_data()
    latest_year  = df["report_year"].max()
    latest_month = df[df["report_year"] == latest_year]["report_month_num"].max()
    latest = df[(df["report_year"] == latest_year) & (df["report_month_num"] == latest_month)]

    total_curr = latest["monthly_curr_usd"].sum()
    total_prev = latest["monthly_prev_usd"].sum()
    ytd_curr   = latest["ytd_curr_usd"].sum()
    ytd_prev   = latest["ytd_prev_usd"].sum()

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return {
        "latest_month": f"{month_names[latest_month]} {latest_year}",
        "monthly_curr_usd_mn":  round(total_curr, 2),
        "monthly_prev_usd_mn":  round(total_prev, 2),
        "monthly_growth_pct":   round((total_curr - total_prev) / total_prev * 100, 2) if total_prev else None,
        "ytd_curr_usd_mn":      round(ytd_curr, 2),
        "ytd_prev_usd_mn":      round(ytd_prev, 2),
        "ytd_growth_pct":       round((ytd_curr - ytd_prev) / ytd_prev * 100, 2) if ytd_prev else None,
    }


# ── Import data functions ─────────────────────────────────────────────────────

def load_import_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load and combine all TRADESTAT Import XLSX files.
    Cached after first load. Same schema as export data.

    Returns a DataFrame with columns:
        report_month_num, report_year, fiscal_year, country,
        monthly_curr_usd, monthly_prev_usd, monthly_growth_pct, monthly_curr_share,
        ytd_curr_usd, ytd_prev_usd, ytd_growth_pct, ytd_curr_share, source_file
    """
    global _df_imports
    if _df_imports is not None and not force_reload:
        return _df_imports

    xlsx_files = sorted(IMPORTS_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found in {IMPORTS_DIR}")

    print(f"Loading {len(xlsx_files)} import XLSX files from {IMPORTS_DIR.name}/...")
    frames = []
    for f in xlsx_files:
        try:
            frames.append(_parse_single_xlsx(f))
        except Exception as e:
            print(f"  [WARN] Skipping {f.name}: {e}")

    _df_imports = pd.concat(frames, ignore_index=True)
    _df_imports = _df_imports.sort_values(
        ["report_year", "report_month_num", "country"]
    ).reset_index(drop=True)

    print(f"  Loaded {len(_df_imports):,} rows | "
          f"{_df_imports['source_file'].nunique()} months | "
          f"{_df_imports['country'].nunique()} countries")
    return _df_imports


def get_latest_top_sources(n: int = 10) -> pd.DataFrame:
    """Return top-n import source countries for the latest available month."""
    df = load_import_data()
    latest_year  = df["report_year"].max()
    latest_month = df[df["report_year"] == latest_year]["report_month_num"].max()
    latest = df[(df["report_year"] == latest_year) & (df["report_month_num"] == latest_month)]
    return (latest.groupby("country")["monthly_curr_usd"]
            .sum().nlargest(n).reset_index()
            .rename(columns={"monthly_curr_usd": "usd_million"}))


def get_import_yoy_summary() -> dict:
    """Return overall YoY import growth for latest month and YTD."""
    df = load_import_data()
    latest_year  = df["report_year"].max()
    latest_month = df[df["report_year"] == latest_year]["report_month_num"].max()
    latest = df[(df["report_year"] == latest_year) & (df["report_month_num"] == latest_month)]

    total_curr = latest["monthly_curr_usd"].sum()
    total_prev = latest["monthly_prev_usd"].sum()
    ytd_curr   = latest["ytd_curr_usd"].sum()
    ytd_prev   = latest["ytd_prev_usd"].sum()

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return {
        "latest_month":         f"{month_names[latest_month]} {latest_year}",
        "monthly_curr_usd_mn":  round(total_curr, 2),
        "monthly_prev_usd_mn":  round(total_prev, 2),
        "monthly_growth_pct":   round((total_curr - total_prev) / total_prev * 100, 2) if total_prev else None,
        "ytd_curr_usd_mn":      round(ytd_curr, 2),
        "ytd_prev_usd_mn":      round(ytd_prev, 2),
        "ytd_growth_pct":       round((ytd_curr - ytd_prev) / ytd_prev * 100, 2) if ytd_prev else None,
    }


def get_trade_balance(period: str = "latest_month") -> pd.DataFrame:
    """
    Compute trade balance (exports - imports) by country for a given period.

    Args:
        period: "latest_month" | "ytd"

    Returns DataFrame with columns:
        country, exports_usd, imports_usd, balance_usd, continent
    Sorted by balance_usd descending (largest surplus first).
    """
    df_exp = load_export_data()
    df_imp = load_import_data()

    # Use the most recent month common to both datasets
    exp_latest_yr = df_exp["report_year"].max()
    exp_latest_mo = df_exp[df_exp["report_year"] == exp_latest_yr]["report_month_num"].max()
    imp_latest_yr = df_imp["report_year"].max()
    imp_latest_mo = df_imp[df_imp["report_year"] == imp_latest_yr]["report_month_num"].max()

    # Use the earlier of the two latest months (safest common period)
    if (exp_latest_yr, exp_latest_mo) <= (imp_latest_yr, imp_latest_mo):
        yr, mo = exp_latest_yr, exp_latest_mo
    else:
        yr, mo = imp_latest_yr, imp_latest_mo

    val_col = "monthly_curr_usd" if period == "latest_month" else "ytd_curr_usd"

    exp_sub = df_exp[(df_exp["report_year"] == yr) & (df_exp["report_month_num"] == mo)]
    imp_sub = df_imp[(df_imp["report_year"] == yr) & (df_imp["report_month_num"] == mo)]

    exp_agg = exp_sub.groupby("country")[val_col].sum().reset_index().rename(columns={val_col: "exports_usd"})
    imp_agg = imp_sub.groupby("country")[val_col].sum().reset_index().rename(columns={val_col: "imports_usd"})

    merged = pd.merge(exp_agg, imp_agg, on="country", how="outer").fillna(0)
    merged["balance_usd"] = merged["exports_usd"] - merged["imports_usd"]
    merged["continent"]   = merged["country"].map(lambda c: REGION_MAP.get(c, ("Other","Other"))[0])
    merged = merged.sort_values("balance_usd", ascending=False).reset_index(drop=True)
    return merged


def query_import_data(question: str, save_chart_as: str | None = None) -> dict:
    """
    Answer a quantitative question about India's steel imports using LLM code-gen.
    Same interface as query_export_data() but operates on the imports DataFrame.
    """
    df = load_import_data()

    months = df[["report_year", "report_month_num"]].drop_duplicates().sort_values(
        ["report_year", "report_month_num"]
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_list = [f"{month_names[r.report_month_num]}{r.report_year}"
                  for r in months.itertuples()]
    latest = df[df["report_year"] == df["report_year"].max()]
    latest = latest[latest["report_month_num"] == latest["report_month_num"].max()]
    top5 = (latest.groupby("country")["monthly_curr_usd"]
            .sum().nlargest(5).round(2).to_dict())

    data_summary = f"""DataFrame name: df
Shape: {len(df):,} rows x {len(df.columns)} columns
Period: {month_list[0]} to {month_list[-1]} ({len(month_list)} monthly snapshots)
Countries: {df['country'].nunique()} unique

Columns:
  report_month_num  int   (1=Jan .. 12=Dec)
  report_year       int   (2024, 2025, 2026)
  fiscal_year       str   ('FY2024-25', 'FY2025-26')
  country           str   (UPPER CASE, source country of steel imports)
  monthly_curr_usd  float USD Million - imports in the report month (current year)
  monthly_prev_usd  float USD Million - imports same month previous year
  monthly_growth_pct float % YoY growth in monthly imports
  monthly_curr_share float % share of total monthly imports
  ytd_curr_usd      float USD Million - Apr to report_month cumulative (current FY)
  ytd_prev_usd      float USD Million - Apr to report_month cumulative (prev FY)
  ytd_growth_pct    float % YoY growth in YTD imports
  ytd_curr_share    float % share of total YTD imports

Top 5 import sources (latest month, USD Mn): {top5}

Notes:
- Values in USD Million. This is IMPORT data — 'country' is the origin of steel coming INTO India.
- Missing data appears as NaN — use fillna(0) or dropna() appropriately.
- Charts: use matplotlib. Save to '{CHARTS_DIR}/<name>.png'. Return chart_path in your JSON.
"""

    safe_stem = re.sub(r"[^\w]", "_", question[:40].lower())
    chart_name = save_chart_as or safe_stem
    chart_path = CHARTS_DIR / f"{chart_name}.png"

    import_system = DATA_AGENT_SYSTEM.replace(
        "specialising in India's steel export trade data",
        "specialising in India's steel import trade data"
    ).replace(
        "monthly TRADESTAT export data",
        "monthly TRADESTAT import data"
    ).replace(
        "`monthly_curr_usd` = exports in the report month",
        "`monthly_curr_usd` = imports in the report month"
    )

    user_msg = f"""DATA SCHEMA:
{data_summary}

QUESTION: {question}

Write pandas code to answer this question. If a chart would help visualise the answer, include it."""

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": import_system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content.strip()

        description, needs_chart, code = "", False, ""
        desc_match  = re.search(r"DESCRIPTION:\s*(.+)", raw)
        chart_match = re.search(r"NEEDS_CHART:\s*(true|false)", raw, re.IGNORECASE)
        code_match  = re.search(r"CODE:\s*\n(.*)", raw, re.DOTALL)

        if desc_match:  description = desc_match.group(1).strip()
        if chart_match: needs_chart = chart_match.group(1).lower() == "true"
        if code_match:
            code = code_match.group(1).strip()
            code = re.sub(r"\n```\s*$", "", code).strip()
        if not code:
            raise ValueError(f"No CODE section found: {raw[:300]}")

    except Exception as e:
        return {
            "question": question, "answer": f"LLM error: {e}",
            "chart_path": None, "needs_chart": False,
            "code_used": "", "description": "", "error": str(e),
        }

    exec_result = _execute_code(code, df, chart_path)
    final_chart = str(chart_path) if (exec_result["chart_generated"] and chart_path.exists()) else None

    return {
        "question":    question,
        "answer":      exec_result["answer"],
        "chart_path":  final_chart,
        "needs_chart": needs_chart,
        "code_used":   code,
        "description": description,
        "error":       exec_result["error"],
        "stdout":      exec_result.get("stdout", ""),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEEL EXPORT DATA AGENT - Quick Test")
    print("=" * 60)

    # 1. Load data and show summary stats
    df = load_export_data()
    print()

    summary = get_yoy_summary()
    print(f"Latest month: {summary['latest_month']}")
    print(f"  Monthly exports : USD {summary['monthly_curr_usd_mn']}M  (prev yr: {summary['monthly_prev_usd_mn']}M)  Growth: {summary['monthly_growth_pct']}%")
    print(f"  YTD exports     : USD {summary['ytd_curr_usd_mn']}M  (prev yr: {summary['ytd_prev_usd_mn']}M)  Growth: {summary['ytd_growth_pct']}%")
    print()

    top = get_latest_top_destinations(10)
    print("Top 10 export destinations (latest month):")
    for _, row in top.iterrows():
        print(f"  {row['country']:<30} USD {row['usd_million']:.2f}M")
    print()

    # 2. Market trend analysis (deterministic — no LLM)
    print("=" * 60)
    print("MARKET TREND ANALYSIS (last 6 months)")
    print("=" * 60)
    trends = get_market_trends(lookback_months=6, n=5)
    print(f"Period: {trends['period']}")
    print()
    print("Top 5 GROWING markets (by export trend slope):")
    for _, row in trends["growing"].iterrows():
        print(f"  {row['country']:<25} slope={row['trend_slope']:+.2f} USD Mn/mo  "
              f"avg={row['avg_monthly_usd']:.1f}M  "
              f"latest_growth={row['growth_pct_latest']}%")
    print()
    print("Top 5 SHRINKING markets:")
    for _, row in trends["shrinking"].iterrows():
        print(f"  {row['country']:<25} slope={row['trend_slope']:+.2f} USD Mn/mo  "
              f"avg={row['avg_monthly_usd']:.1f}M  "
              f"latest_growth={row['growth_pct_latest']}%")
    print()

    # 3. Country trend
    print("Vietnam trend (last 12 months):")
    vn = get_country_trend("VIETNAM SOC REP")
    for _, row in vn.tail(6).iterrows():
        print(f"  {row['period']:<12} USD {row['monthly_usd']:.2f}M  "
              f"YoY: {row['monthly_growth_pct']}%")
    print()

    # 4. Country-by-country comparison (stats + dual charts)
    print("=" * 60)
    print("COUNTRY COMPARISON")
    print("=" * 60)
    comp = compare_countries(
        ["ITALY", "NEPAL", "VIETNAM SOC REP", "U K", "TURKEY", "BELGIUM"],
        save_as="country_comp"
    )
    print("Stats:")
    print(comp["stats"][["country", "continent", "sub_region",
                          "avg_monthly_usd", "latest_usd",
                          "direction", "latest_yoy_pct"]].to_string(index=False))
    print(f"\nTrend chart : {comp['chart_trend']}")
    print(f"Latest chart: {comp['chart_latest']}")
    print()

    # 5. Regional breakdown
    print("=" * 60)
    print("REGIONAL BREAKDOWN (latest month)")
    print("=" * 60)
    reg = plot_regional_breakdown(period="latest_month", save_as="regional_latest")
    print(reg["summary"].to_string(index=False))
    print(f"\nChart: {reg['chart_path']}")
    print()

    # 6. Growing vs shrinking markets chart
    trend_result = plot_market_trends(lookback_months=6, n=8, save_as="market_trends_6mo")
    print(f"Market trends chart: {trend_result['chart_path']}")
    print()

    # 6. LLM-powered queries
    print("=" * 60)
    print("LLM-POWERED QUERIES")
    print("=" * 60)
    test_questions = [
        "Which 5 countries had the highest YoY growth in monthly steel exports in the latest month?",
        "Show a bar chart of India's top 10 steel export destinations for the latest month by USD value.",
        "Compare Vietnam and Italy steel export trends from Jan 2025 to latest month on a line chart.",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = query_export_data(q)
        print(f"A: {result['answer']}")
        if result["chart_path"]:
            print(f"   Chart saved: {result['chart_path']}")
        if result["error"]:
            print(f"   [ERROR] {result['error'][:200]}")
        print()
