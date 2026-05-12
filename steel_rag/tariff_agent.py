"""
tariff_agent.py - India MFN Tariff Analysis Agent (HS 72 & 73, 2010-2023)

Source: WTO WITS MFN applied tariffs for India (Reporter ISO 356)
Data:   14 annual CSV files, HS-6 level, SimpleAverage MFN rate

Capabilities:
  - Lookup MFN rate for any steel HS code in any year
  - Trend analysis for a product across 14 years
  - Chapter-level summaries (HS 72 vs 73)
  - Highest-tariff product rankings
  - Tariff change detection (years where rates shifted)
  - Charts: trend lines, heatmaps, comparisons

Usage:
    from tariff_agent import query_tariff, get_tariff_trend, get_chapter_summary
    result = query_tariff("What is India's MFN duty on hot-rolled steel coils?")

Or run directly: python tariff_agent.py
"""

import os
import re
import sys
import json
import traceback
from pathlib import Path
from io import StringIO

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
MFN_DIR    = Path(__file__).parent.parent / "Base documents" / "MFN_India"
CHARTS_DIR = Path(__file__).parent / "charts"
GROQ_MODEL = "llama-3.3-70b-versatile"
# ─────────────────────────────────────────────────────────────────────────────

# ── HS Code Descriptions (HS 72 & 73 — Iron & Steel) ─────────────────────────
HS_DESCRIPTIONS = {
    # ── Chapter 72: Iron and Steel ────────────────────────────────────────────
    "7201": "Pig iron and spiegeleisen in pigs, blocks or other primary forms",
    "720110": "Pig iron, non-alloy (phosphorus ≥ 0.5%)",
    "720120": "Pig iron, non-alloy (phosphorus < 0.5%)",
    "720150": "Alloy pig iron; spiegeleisen",
    "7202": "Ferro-alloys",
    "720211": "Ferro-manganese (carbon > 2%)",
    "720219": "Ferro-manganese (carbon ≤ 2%)",
    "720221": "Ferro-silicon (silicon > 55%)",
    "720229": "Ferro-silicon (silicon ≤ 55%)",
    "720230": "Ferro-silico-manganese",
    "720241": "Ferro-chromium (carbon > 4%)",
    "720249": "Ferro-chromium (carbon ≤ 4%)",
    "720250": "Ferro-silico-chromium",
    "720260": "Ferro-nickel",
    "720270": "Ferro-molybdenum",
    "720280": "Ferro-tungsten and ferro-silico-tungsten",
    "720291": "Ferro-titanium and ferro-silico-titanium",
    "720292": "Ferro-vanadium",
    "720293": "Ferro-niobium",
    "720299": "Other ferro-alloys",
    "7203": "Ferrous products obtained by direct reduction of iron ore",
    "7204": "Ferrous waste and scrap; remelting scrap ingots of iron or steel",
    "720410": "Waste and scrap of cast iron",
    "720421": "Waste and scrap of stainless steel",
    "720429": "Waste and scrap of other alloy steel",
    "720430": "Waste and scrap of tinned iron or steel",
    "720441": "Turnings, shavings, chips, milling waste",
    "720449": "Other ferrous waste and scrap",
    "720450": "Remelting scrap ingots",
    "7205": "Granules and powders of pig iron, spiegeleisen, iron or steel",
    "7206": "Iron and non-alloy steel in ingots or other primary forms",
    "7207": "Semi-finished products of iron or non-alloy steel",
    "720711": "Billets — rectangular cross-section (carbon < 0.25%)",
    "720712": "Other billets — rectangular cross-section (carbon < 0.25%)",
    "720719": "Other semi-finished iron/non-alloy steel",
    "720720": "Semi-finished products of iron/non-alloy steel (carbon ≥ 0.25%)",
    "7208": "Flat-rolled iron/non-alloy steel ≥600mm wide, hot-rolled (HR coils/sheets/plates)",
    "720810": "HR flat-rolled, with patterns in relief",
    "720825": "HR flat-rolled, coiled, thickness ≥ 4.75mm",
    "720826": "HR flat-rolled, coiled, thickness 3–4.75mm",
    "720827": "HR flat-rolled, coiled, thickness < 3mm",
    "720836": "HR flat-rolled, not coiled, thickness > 10mm",
    "720837": "HR flat-rolled, not coiled, thickness 4.75–10mm",
    "720838": "HR flat-rolled, not coiled, thickness 3–4.75mm",
    "720839": "HR flat-rolled, not coiled, thickness < 3mm",
    "720840": "HR flat-rolled, not coiled, not clad/plated",
    "720851": "HR plates, thickness ≥ 10mm",
    "720852": "HR plates, thickness 4.75–10mm",
    "720853": "HR plates, thickness 3–4.75mm",
    "720854": "HR plates, thickness < 3mm",
    "7209": "Flat-rolled iron/non-alloy steel ≥600mm wide, cold-rolled (CR coils/sheets)",
    "720915": "CR flat-rolled, coiled, thickness ≥ 3mm",
    "720916": "CR flat-rolled, coiled, thickness 1–3mm",
    "720917": "CR flat-rolled, coiled, thickness 0.5–1mm",
    "720918": "CR flat-rolled, coiled, thickness < 0.5mm",
    "720925": "CR flat-rolled, not coiled, thickness ≥ 3mm",
    "720926": "CR flat-rolled, not coiled, thickness 1–3mm",
    "720927": "CR flat-rolled, not coiled, thickness 0.5–1mm",
    "720928": "CR flat-rolled, not coiled, thickness < 0.5mm",
    "720990": "Other cold-rolled flat products",
    "7210": "Flat-rolled iron/non-alloy steel ≥600mm, clad/plated/coated (galvanized/electrogalvanized)",
    "721011": "Tin-plated, thickness ≥ 0.5mm",
    "721012": "Tin-plated, thickness < 0.5mm",
    "721020": "Lead-coated or lead-tin alloy coated",
    "721030": "Electrolytically plated/coated with zinc (electrogalvanized)",
    "721041": "Zinc-coated, corrugated (hot-dip galvanized)",
    "721049": "Other zinc-coated (hot-dip galvanized)",
    "721050": "Chromium oxide / chromium and chromium oxide coated",
    "721061": "Aluminium-zinc alloy coated",
    "721069": "Other aluminium-coated",
    "721070": "Painted, varnished or coated with plastics (colour-coated)",
    "721090": "Other clad/plated/coated flat-rolled",
    "7211": "Flat-rolled iron/non-alloy steel <600mm wide, not clad/plated/coated",
    "7212": "Flat-rolled iron/non-alloy steel <600mm wide, clad/plated/coated",
    "7213": "Wire rod, iron or non-alloy steel, hot-rolled in irregular coils",
    "7214": "Other bars and rods of iron/non-alloy steel, not further worked",
    "7215": "Other bars and rods of iron/non-alloy steel",
    "7216": "Angles, shapes and sections of iron/non-alloy steel",
    "7217": "Wire of iron or non-alloy steel",
    "7218": "Stainless steel in ingots or other primary forms; semi-finished products",
    "7219": "Flat-rolled products of stainless steel, width ≥ 600mm",
    "721911": "HR stainless, coiled, thickness > 10mm",
    "721912": "HR stainless, coiled, thickness 4.75–10mm",
    "721913": "HR stainless, coiled, thickness 3–4.75mm",
    "721914": "HR stainless, coiled, thickness < 3mm",
    "721921": "HR stainless, not coiled, thickness > 10mm",
    "721922": "HR stainless, not coiled, thickness 4.75–10mm",
    "721923": "HR stainless, not coiled, thickness 3–4.75mm",
    "721924": "HR stainless, not coiled, thickness < 3mm",
    "721931": "CR stainless, thickness ≥ 4.75mm",
    "721932": "CR stainless, thickness 3–4.75mm",
    "721933": "CR stainless, thickness 1–3mm",
    "721934": "CR stainless, thickness 0.5–1mm",
    "721935": "CR stainless, thickness < 0.5mm",
    "721990": "Other stainless flat-rolled ≥600mm",
    "7220": "Flat-rolled products of stainless steel, width < 600mm",
    "7221": "Bars and rods of stainless steel, hot-rolled",
    "7222": "Other bars, rods, angles, shapes and sections of stainless steel",
    "7223": "Wire of stainless steel",
    "7224": "Other alloy steel in ingots or other primary forms",
    "7225": "Flat-rolled products of other alloy steel, width ≥ 600mm",
    "722511": "HR silicon-electrical steel, grain-oriented",
    "722519": "Other HR silicon-electrical steel",
    "722530": "HR flat-rolled, other alloy steel (coiled)",
    "722540": "HR flat-rolled, other alloy steel (not coiled)",
    "722550": "CR flat-rolled, other alloy steel",
    "722591": "Electrolytically plated/coated, other alloy",
    "722592": "Otherwise coated, other alloy",
    "722599": "Other flat-rolled alloy steel ≥600mm",
    "7226": "Flat-rolled products of other alloy steel, width < 600mm",
    "7227": "Bars and rods of other alloy steel, hot-rolled in irregular coils",
    "7228": "Other bars, rods, angles, shapes and sections of other alloy steel",
    "7229": "Wire of other alloy steel",

    # ── Chapter 73: Articles of Iron or Steel ─────────────────────────────────
    "7301": "Sheet piling of iron or steel; welded angles, shapes and sections",
    "7302": "Railway or tramway track construction material of iron or steel",
    "7303": "Tubes, pipes and hollow profiles of cast iron",
    "7304": "Seamless tubes, pipes and hollow profiles of iron or steel",
    "730411": "Seamless, line pipe, stainless steel",
    "730419": "Seamless, line pipe, other steel",
    "730421": "Seamless, casing/tubing, stainless (drill pipe)",
    "730429": "Seamless, casing/tubing, other steel",
    "730431": "Seamless, cold-drawn/rolled, stainless",
    "730439": "Seamless, cold-drawn/rolled, other steel",
    "730441": "Seamless, other, circular cross-section, stainless",
    "730449": "Seamless, other, circular cross-section, other steel",
    "730451": "Seamless, other, circular, OD ≤ 168.3mm, alloy steel",
    "730459": "Seamless, other, circular, OD > 168.3mm",
    "730490": "Seamless tubes, other cross-section",
    "7305": "Tubes, pipes, hollow profiles of iron/steel (welded, large diameter ≥ 406.4mm OD)",
    "7306": "Other tubes, pipes and hollow profiles of iron or steel (welded)",
    "730611": "Welded line pipe, stainless steel",
    "730619": "Welded line pipe, other steel",
    "730621": "Welded casing/tubing, stainless steel",
    "730629": "Welded casing/tubing, other steel",
    "730630": "Welded, circular cross-section, iron/non-alloy",
    "730640": "Welded, circular cross-section, stainless",
    "730650": "Welded, circular cross-section, other alloy",
    "730661": "Welded, square/rectangular, stainless",
    "730669": "Welded, square/rectangular, other",
    "730690": "Other welded tubes/pipes",
    "7307": "Tube or pipe fittings of iron or steel",
    "7308": "Structures and parts of structures of iron or steel",
    "7309": "Reservoirs, tanks, vats and similar containers of iron or steel",
    "7310": "Tanks, casks, drums, cans, boxes of iron or steel (capacity ≤ 300L)",
    "7311": "Containers for compressed or liquefied gas of iron or steel",
    "7312": "Stranded wire, ropes, cables, plaited bands of iron or steel",
    "7313": "Barbed wire of iron or steel",
    "7314": "Cloth, grill, netting and fencing of iron or steel wire",
    "7315": "Chain and parts thereof of iron or steel",
    "7316": "Anchors, grapnels and parts thereof",
    "7317": "Nails, tacks, drawing pins, corrugated nails",
    "7318": "Screws, bolts, nuts, coach screws, screw hooks, rivets, washers",
    "7319": "Sewing needles, knitting needles, bodkins, crochet hooks",
    "7320": "Springs and leaves for springs of iron or steel",
    "7321": "Stoves, ranges, grates, cookers, barbecues",
    "7322": "Radiators for central heating, not electrically heated",
    "7323": "Table, kitchen or other household articles of iron or steel",
    "7324": "Sanitary ware and parts of iron or steel",
    "7325": "Other cast articles of iron or steel",
    "7326": "Other articles of iron or steel",
}

def hs_description(code: str) -> str:
    """Return product description for an HS code (tries 6-digit, then 4-digit)."""
    code = str(code).strip()
    # Try exact match first (6-digit)
    if code in HS_DESCRIPTIONS:
        return HS_DESCRIPTIONS[code]
    # Try heading (4-digit)
    heading = code[:4]
    if heading in HS_DESCRIPTIONS:
        return HS_DESCRIPTIONS[heading]
    # Try chapter (2-digit)
    chapter = code[:2]
    chapter_map = {"72": "Iron and Steel (HS Chapter 72)", "73": "Articles of Iron or Steel (HS Chapter 73)"}
    return chapter_map.get(chapter, f"HS {code}")


# ── Data Loader ───────────────────────────────────────────────────────────────

_df_tariff: pd.DataFrame | None = None
_groq_client = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def load_tariff_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load all MFN CSV files, filter for HS 72 & 73, return combined DataFrame.

    Columns:
        year, hs6 (str, zero-padded), chapter (int), heading (str),
        min_rate, max_rate, avg_rate (SimpleAverage), dutiable_lines,
        free_lines, total_lines, description
    """
    global _df_tariff
    if _df_tariff is not None and not force_reload:
        return _df_tariff

    csv_files = sorted(MFN_DIR.glob("*.CSV"))
    if not csv_files:
        raise FileNotFoundError(f"No MFN CSV files found in {MFN_DIR}")

    print(f"Loading {len(csv_files)} MFN tariff files...")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="latin-1")
            # Filter for HS 72 and 73
            mask = df["ProductCode"].astype(str).str.startswith(("72", "73"))
            steel = df[mask].copy()
            if steel.empty:
                continue

            steel["year"]     = df["Year"].iloc[0]
            steel["hs6"]      = steel["ProductCode"].astype(str).str.zfill(6)
            steel["chapter"]  = steel["hs6"].str[:2].astype(int)
            steel["heading"]  = steel["hs6"].str[:4]
            steel["avg_rate"] = pd.to_numeric(steel["SimpleAverage"], errors="coerce")
            steel["min_rate"] = pd.to_numeric(steel["Min_Rate"],       errors="coerce")
            steel["max_rate"] = pd.to_numeric(steel["Max_Rate"],       errors="coerce")
            steel["dutiable_lines"] = pd.to_numeric(steel["Nbr_Dutiable_Lines"], errors="coerce")
            steel["free_lines"]     = pd.to_numeric(steel["Nbr_Free_Lines"],     errors="coerce")
            steel["total_lines"]    = pd.to_numeric(steel["TotalNoOfValidLines"], errors="coerce")

            steel["description"] = steel["hs6"].apply(hs_description)

            frames.append(steel[[
                "year", "hs6", "chapter", "heading",
                "min_rate", "max_rate", "avg_rate",
                "dutiable_lines", "free_lines", "total_lines",
                "description",
            ]])
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    _df_tariff = (pd.concat(frames, ignore_index=True)
                  .sort_values(["year", "hs6"])
                  .reset_index(drop=True))

    print(f"  Loaded {len(_df_tariff):,} rows | "
          f"{_df_tariff['year'].nunique()} years ({_df_tariff['year'].min()}-{_df_tariff['year'].max()}) | "
          f"{_df_tariff['hs6'].nunique()} unique HS codes")
    return _df_tariff


# ── Core Query Functions ──────────────────────────────────────────────────────

def get_tariff(hs_code: str, year: int = None) -> pd.DataFrame:
    """
    Look up MFN rate for a specific HS code.
    If year is None, returns all years.
    Supports partial codes: '7208' returns all 7208xx codes.
    """
    df = load_tariff_data()
    hs_code = str(hs_code).strip().replace(".", "").zfill(len(hs_code.replace(".", "")))

    if len(hs_code) <= 4:
        mask = df["hs6"].str.startswith(hs_code)
    else:
        mask = df["hs6"] == hs_code.zfill(6)

    result = df[mask].copy()
    if year:
        result = result[result["year"] == year]
    return result.sort_values(["year", "hs6"])


def get_tariff_trend(hs_code: str) -> pd.DataFrame:
    """
    Return year-by-year MFN rate trend for a specific HS code (or heading).
    Groups multiple sub-codes under a heading into a weighted average.
    """
    df = load_tariff_data()
    hs_code = str(hs_code).strip().replace(".", "")

    if len(hs_code) <= 4:
        mask = df["hs6"].str.startswith(hs_code.zfill(4))
        grp  = df[mask].groupby("year").agg(
            avg_rate=("avg_rate", "mean"),
            min_rate=("min_rate", "min"),
            max_rate=("max_rate", "max"),
        ).round(2).reset_index()
        grp["hs_code"] = hs_code
        grp["description"] = hs_description(hs_code)
    else:
        mask = df["hs6"] == hs_code.zfill(6)
        grp  = df[mask][["year", "avg_rate", "min_rate", "max_rate"]].copy()
        grp["hs_code"]     = hs_code
        grp["description"] = hs_description(hs_code)

    return grp.sort_values("year").reset_index(drop=True)


def get_chapter_summary(year: int = None) -> pd.DataFrame:
    """
    Chapter-level summary: average MFN rate for HS 72 vs HS 73.
    If year is None, returns all years.
    """
    df = load_tariff_data()
    if year:
        df = df[df["year"] == year]

    result = (df.groupby(["year", "chapter"])
              .agg(avg_rate=("avg_rate", "mean"),
                   max_rate=("max_rate", "max"),
                   free_pct=("free_lines", lambda x: 100 * x.sum() /
                             df.loc[x.index, "total_lines"].sum()))
              .round(2).reset_index())
    result["chapter_name"] = result["chapter"].map(
        {72: "Iron & Steel", 73: "Articles of Iron/Steel"})
    return result.sort_values(["year", "chapter"])


def get_top_tariff_products(year: int, n: int = 10, chapter: int = None) -> pd.DataFrame:
    """Return the n products with highest MFN average rate in a given year."""
    df = load_tariff_data()
    sub = df[df["year"] == year]
    if chapter:
        sub = sub[sub["chapter"] == chapter]
    return (sub.nlargest(n, "avg_rate")
            [["hs6", "heading", "avg_rate", "min_rate", "max_rate", "description"]]
            .reset_index(drop=True))


def get_tariff_changes(hs_code: str = None) -> pd.DataFrame:
    """
    Identify years where MFN rates changed significantly (> 0.5% shift).
    If hs_code is None, returns chapter-level changes.
    """
    df = load_tariff_data()

    if hs_code:
        trend = get_tariff_trend(hs_code)
        trend["prev_rate"] = trend["avg_rate"].shift(1)
        trend["change"]    = (trend["avg_rate"] - trend["prev_rate"]).round(2)
        return trend[trend["change"].abs() > 0.5].dropna()
    else:
        # Chapter-level changes
        ch = get_chapter_summary()
        ch["prev_rate"] = ch.groupby("chapter")["avg_rate"].shift(1)
        ch["change"]    = (ch["avg_rate"] - ch["prev_rate"]).round(2)
        return ch[ch["change"].abs() > 0.5].dropna().sort_values("year")


def search_by_product(keyword: str) -> pd.DataFrame:
    """
    Search HS codes by product description keyword.
    Returns matching HS codes with their descriptions and latest rate.
    """
    df = load_tariff_data()
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]

    mask = latest["description"].str.lower().str.contains(keyword.lower(), regex=False)
    results = latest[mask][["hs6", "heading", "avg_rate", "max_rate", "description"]].copy()

    # Also search in HS_DESCRIPTIONS
    extra_codes = [
        code for code, desc in HS_DESCRIPTIONS.items()
        if keyword.lower() in desc.lower() and len(code) == 6
    ]
    for code in extra_codes:
        if code not in results["hs6"].values:
            row = latest[latest["hs6"] == code]
            if not row.empty:
                results = pd.concat([results, row[["hs6","heading","avg_rate","max_rate","description"]]])

    return results.drop_duplicates("hs6").sort_values("avg_rate", ascending=False).reset_index(drop=True)


# ── Chart Functions ───────────────────────────────────────────────────────────

def plot_tariff_trend(hs_codes: list[str],
                      save_as: str = "tariff_trend") -> str:
    """
    Line chart showing MFN rate trend for one or more HS codes over all years.
    Returns path to saved chart.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    for code in hs_codes:
        trend = get_tariff_trend(code)
        if trend.empty:
            continue
        label = f"HS {code}: {hs_description(code)[:40]}"
        ax.plot(trend["year"], trend["avg_rate"],
                marker="o", linewidth=2, markersize=5, label=label)

    ax.set_title("India MFN Applied Tariff Trend — Steel Products", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average MFN Rate (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(range(2010, 2024))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out = CHARTS_DIR / f"{save_as}.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    return str(out)


def plot_chapter_comparison(save_as: str = "tariff_chapter") -> str:
    """
    Dual-line chart: HS 72 vs HS 73 average MFN rate over all years.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ch = get_chapter_summary()

    fig, ax = plt.subplots(figsize=(12, 6))
    for chapter, grp in ch.groupby("chapter"):
        grp = grp.sort_values("year")
        label = f"HS {chapter}: {grp['chapter_name'].iloc[0]}"
        ax.plot(grp["year"], grp["avg_rate"],
                marker="o", linewidth=2.5, markersize=6, label=label)

    ax.set_title("India MFN Tariff: HS 72 (Iron & Steel) vs HS 73 (Articles)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Simple Average MFN Rate (%)")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(range(2010, 2024))
    plt.xticks(rotation=45)

    # Annotate key events
    ax.axvline(2016, color="orange", linestyle=":", alpha=0.7)
    ax.text(2016.1, ax.get_ylim()[1]*0.95, "2016: Tariff hike\n(5%→7.5%)", fontsize=7, color="orange")
    ax.axvline(2019, color="red", linestyle=":", alpha=0.7)
    ax.text(2019.1, ax.get_ylim()[1]*0.85, "2019: Max rate\nraised to 25%", fontsize=7, color="red")

    plt.tight_layout()
    out = CHARTS_DIR / f"{save_as}.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    return str(out)


def plot_tariff_heatmap(year: int = 2023, chapter: int = 72,
                        save_as: str = "tariff_heatmap") -> str:
    """
    Heatmap of MFN rates by heading within a chapter for a given year.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_tariff_data()
    sub = df[(df["year"] == year) & (df["chapter"] == chapter)]

    pivot = (sub.groupby("heading")["avg_rate"]
             .mean().round(2).reset_index()
             .sort_values("avg_rate", ascending=False))

    pivot["desc"] = pivot["heading"].apply(lambda h: hs_description(h)[:45])

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    bars = ax.barh(pivot["desc"][::-1], pivot["avg_rate"][::-1],
                   color=plt.cm.RdYlGn_r(pivot["avg_rate"][::-1] / pivot["avg_rate"].max()))
    for bar, val in zip(bars, pivot["avg_rate"][::-1]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)

    chapter_name = {72: "Iron & Steel", 73: "Articles of Iron/Steel"}.get(chapter, str(chapter))
    ax.set_title(f"India MFN Tariff Rates by Heading — HS {chapter} ({chapter_name}), {year}", fontsize=12)
    ax.set_xlabel("Average MFN Rate (%)")
    plt.tight_layout()

    out = CHARTS_DIR / f"{save_as}.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    return str(out)


# ── LLM-powered tariff query ──────────────────────────────────────────────────

TARIFF_AGENT_SYSTEM = """You are a Python analyst for India's MFN steel tariff data.
You have a pandas DataFrame `df` with these columns:
  year (int), hs6 (str 6-digit), chapter (int 72 or 73), heading (str 4-digit),
  avg_rate (float %), min_rate (float %), max_rate (float %),
  dutiable_lines (float), free_lines (float), total_lines (float),
  description (str — product name)

Available helper functions (already imported, use them):
  get_tariff(hs_code, year=None)   — lookup by code
  get_tariff_trend(hs_code)        — year-by-year trend
  get_chapter_summary(year=None)   — chapter totals
  get_top_tariff_products(year, n) — highest-rate products
  get_tariff_changes(hs_code=None) — where rates changed
  search_by_product(keyword)       — find codes by name
  plot_tariff_trend(hs_codes, save_as=...)  — chart
  plot_chapter_comparison(save_as=...)      — HS72 vs HS73 chart
  plot_tariff_heatmap(year, chapter, ...)   — heading-level heatmap

Return your response in EXACTLY this format:

DESCRIPTION: <one sentence>
NEEDS_CHART: true or false
CODE:
<python code — set `answer` str and optionally `chart_generated=True`>

Rules:
- Set `answer` as a plain string. Include key numbers inline.
  For a trend, write something like: "HS 7208 MFN tariff was 5.0% in 2015, rose to 7.5% by 2016, and remains 7.5% in 2023 (stable since 2016)."
  If you have a DataFrame, extract the key rows and describe them in prose — do NOT just dump the whole table.
- CHART_PATH is already a variable containing the full file path string — use it directly:
    plt.savefig(CHART_PATH, bbox_inches='tight', dpi=120)
    chart_generated = True
  Do NOT append filenames to CHART_PATH. Do NOT quote CHART_PATH as a string.
- Do NOT wrap code in ``` fences.
- Latest year available is 2023.
"""


def query_tariff(question: str, save_chart_as: str = None) -> dict:
    """
    Answer any tariff question using LLM-generated pandas code.
    Returns {question, answer, chart_path, code_used, error}.
    """
    df = load_tariff_data()

    safe_stem  = re.sub(r"[^\w]", "_", question[:40].lower())
    chart_name = save_chart_as or safe_stem
    chart_path = CHARTS_DIR / f"{chart_name}.png"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    user_msg = (
        f"Latest year: {df['year'].max()}\n"
        f"Available years: {sorted(df['year'].unique().tolist())}\n"
        f"Steel HS codes: {df['hs6'].nunique()} unique codes in chapters 72 & 73\n\n"
        f"QUESTION: {question}"
    )

    try:
        resp = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": TARIFF_AGENT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content.strip()

        desc_match  = re.search(r"DESCRIPTION:\s*(.+)", raw)
        chart_match = re.search(r"NEEDS_CHART:\s*(true|false)", raw, re.IGNORECASE)
        code_match  = re.search(r"CODE:\s*\n(.*)", raw, re.DOTALL)

        description = desc_match.group(1).strip()  if desc_match  else ""
        needs_chart = chart_match.group(1).lower() == "true" if chart_match else False
        code        = code_match.group(1).strip()  if code_match  else ""
        code        = re.sub(r"\n```\s*$", "", code).strip()

        if not code:
            raise ValueError("No CODE section in response")

    except Exception as e:
        return {"question": question, "answer": f"LLM error: {e}",
                "chart_path": None, "code_used": "", "error": str(e)}

    # Execute code
    import numpy as np
    local_vars = {
        "df": df.copy(), "pd": pd, "plt": plt, "np": np,
        "CHART_PATH": str(chart_path),
        "CHARTS_DIR": CHARTS_DIR,
        "get_tariff":            get_tariff,
        "get_tariff_trend":      get_tariff_trend,
        "get_chapter_summary":   get_chapter_summary,
        "get_top_tariff_products": get_top_tariff_products,
        "get_tariff_changes":    get_tariff_changes,
        "search_by_product":     search_by_product,
        "plot_tariff_trend":     plot_tariff_trend,
        "plot_chapter_comparison": plot_chapter_comparison,
        "plot_tariff_heatmap":   plot_tariff_heatmap,
        "hs_description":        hs_description,
        "answer":          "No answer generated.",
        "chart_generated": False,
        "chart_path":      None,
    }

    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    error = None
    try:
        exec(compile(code, "<tariff_agent>", "exec"), local_vars)
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        plt.close("all")

    final_chart = str(chart_path) if (local_vars.get("chart_generated") and chart_path.exists()) else None

    return {
        "question":   question,
        "answer":     str(local_vars.get("answer", "No answer generated.")),
        "chart_path": final_chart,
        "code_used":  code,
        "error":      error,
        "stdout":     captured.getvalue().strip(),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("INDIA MFN TARIFF AGENT — Steel (HS 72 & 73), 2010-2023")
    print("=" * 65)

    df = load_tariff_data()
    print()

    # 1. Chapter summary (latest year)
    print("Chapter summary (2023):")
    ch = get_chapter_summary(year=2023)
    for _, row in ch.iterrows():
        print(f"  HS {int(row.chapter)} — {row.chapter_name:<30} "
              f"avg={row.avg_rate:.2f}%  max={row.max_rate:.1f}%")
    print()

    # 2. Key product lookups
    products = [
        ("7208", "HR flat-rolled coils/sheets/plates"),
        ("7209", "CR flat-rolled coils/sheets"),
        ("7210", "Galvanized/coated flat-rolled (electrogalvanized)"),
        ("7304", "Seamless tubes and pipes"),
        ("7306", "Welded tubes and pipes"),
    ]
    print("Key product MFN rates (2023):")
    for code, label in products:
        t = get_tariff(code, year=2023)
        if not t.empty:
            avg = t["avg_rate"].mean()
            mx  = t["max_rate"].max()
            print(f"  HS {code}: {label:<45} avg={avg:.1f}%  max={mx:.1f}%")
    print()

    # 3. Tariff change history (chapter level)
    print("Years with significant tariff changes:")
    changes = get_tariff_changes()
    for _, row in changes.iterrows():
        direction = "UP" if row["change"] > 0 else "DOWN"
        print(f"  {int(row.year)}  HS {int(row.chapter)} ({row.chapter_name:<22})  "
              f"{row.prev_rate:.2f}% → {row.avg_rate:.2f}%  [{direction} {abs(row.change):.2f}%]")
    print()

    # 4. Top 5 highest-tariff products (2023, HS 73)
    print("Top 5 highest MFN rates in HS 73 (2023):")
    top = get_top_tariff_products(year=2023, n=5, chapter=73)
    for _, row in top.iterrows():
        print(f"  HS {row.hs6}  {row.description[:50]:<50}  {row.avg_rate:.1f}%")
    print()

    # 5. Charts
    print("Generating charts...")
    c1 = plot_chapter_comparison(save_as="mfn_chapter_trend")
    print(f"  Chapter trend chart : {c1}")

    c2 = plot_tariff_trend(["7208", "7209", "7210", "7304"],
                            save_as="mfn_key_products")
    print(f"  Key products chart  : {c2}")

    c3 = plot_tariff_heatmap(year=2023, chapter=72, save_as="mfn_heatmap_72_2023")
    print(f"  HS 72 heatmap       : {c3}")
    print()

    # 6. LLM query
    print("=" * 65)
    print("LLM-POWERED TARIFF QUERIES")
    print("=" * 65)
    questions = [
        "How has India's MFN tariff on seamless tubes (HS 7304) changed from 2010 to 2023?",
        "Which steel products have the highest MFN tariff in India in 2023?",
    ]
    for q in questions:
        print(f"Q: {q}")
        r = query_tariff(q)
        print(f"A: {r['answer']}")
        if r["chart_path"]:
            print(f"   Chart: {r['chart_path']}")
        if r["error"]:
            print(f"   [ERROR] {r['error'][:150]}")
        print()
