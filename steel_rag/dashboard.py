"""
dashboard.py - India Steel Trade Intelligence Platform
Streamlit dashboard — 4 tabs:
  1. Intelligence Query   — route any question through the multi-agent router
  2. Export Trends        — data_agent charts (top destinations, trends, regions)
  3. Tariff Lookup        — MFN tariff rates (HS 72 & 73, 2010-2023)
  4. Eval Report          — baseline_v1 vs v1b comparison

Run:
    streamlit run dashboard.py
"""

import sys
import json
import time
from pathlib import Path

# ── resolve project root ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT.parent / ".env")

import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Steel Trade Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .badge {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:12px; font-weight:600; margin-right:6px;
  }
  .badge-ad      { background:#fee2e2; color:#991b1b; }
  .badge-sg      { background:#fef3c7; color:#92400e; }
  .badge-policy  { background:#dbeafe; color:#1e40af; }
  .badge-data    { background:#d1fae5; color:#065f46; }
  .badge-tariff  { background:#ede9fe; color:#5b21b6; }
  .badge-risk    { background:#fff7ed; color:#9a3412; }
  .metric-box {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:8px; padding:12px 16px; margin:4px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.title("🏭 India Steel Trade Intelligence Platform")
st.caption("RAG · Multi-Agent Router · Export Analytics · MFN Tariff Analysis")
st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Intelligence Query",
    "📈 Export Trends",
    "📊 Tariff Lookup",
    "🧪 Eval Report",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Intelligence Query
# ═══════════════════════════════════════════════════════════════════════════════

BADGE_MAP = {
    "ANTI_DUMPING":       ("badge-ad",     "⚖️ Anti-Dumping"),
    "SAFEGUARD":          ("badge-sg",     "🛡️ Safeguard"),
    "POLICY_OPPORTUNITY": ("badge-policy", "📋 Policy"),
    "RAW_MATERIAL":       ("badge-risk",   "⛏️ Raw Material"),
    "CBAM_COMPLIANCE":    ("badge-risk",   "🌿 CBAM"),
    "DATA_ANALYSIS":      ("badge-data",   "📊 Data"),
    "TARIFF_ANALYSIS":    ("badge-tariff", "🔢 Tariff"),
}

EXAMPLE_QUESTIONS = [
    "Select an example…",
    "What anti-dumping duty was imposed on seamless tubes from China?",
    "What products are covered by India's safeguard investigation on steel flat products?",
    "Which 5 countries receive the most Indian steel exports by value?",
    "What are the growing markets for Indian steel exports in the last 6 months?",
    "What is India's MFN tariff on hot-rolled steel coils HS 7208?",
    "How does the EU CBAM affect Indian steel exporters?",
    "Compare Vietnam and UAE steel export trends over the last year",
]

with tab1:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        example = st.selectbox("Example questions", EXAMPLE_QUESTIONS,
                               label_visibility="collapsed")
        default_q = "" if example.startswith("Select") else example
        question = st.text_area(
            "Ask anything about India's steel trade",
            value=default_q,
            height=80,
            placeholder="e.g. Which countries are subject to anti-dumping on electrogalvanized steel?",
        )
    with col_btn:
        st.write("")
        st.write("")
        run = st.button("Ask ›", type="primary", use_container_width=True)

    if run and question.strip():
        with st.spinner("Routing question to the right agent…"):
            from router import route_query, PolicyAnalystOutput, SupplyChainRiskOutput, DataAnalysisOutput, TariffAnalysisOutput
            ro = route_query(question.strip())

        # ── type badge + latency ──────────────────────────────────────────────
        badge_cls, badge_label = BADGE_MAP.get(ro.question_type, ("badge-data", ro.question_type))
        st.markdown(
            f'<span class="badge {badge_cls}">{badge_label}</span>'
            f'<span style="color:#94a3b8;font-size:12px;">via {ro.agent_used} · {ro.latency_ms:,}ms</span>',
            unsafe_allow_html=True,
        )
        st.write("")

        r = ro.result

        # ── Policy / AD / Safeguard ───────────────────────────────────────────
        if isinstance(r, PolicyAnalystOutput):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duty Type",      r.duty_type)
            col2.metric("Duty Rate",      r.duty_rate)
            col3.metric("Effective Date", r.effective_date)
            col4.metric("Confidence",     f"{r.confidence:.0%}")

            if r.countries:
                st.write("**Countries involved:**", ", ".join(r.countries))
            if r.product:
                st.write("**Product:**", r.product)

            st.divider()
            st.subheader("Answer")
            st.write(r.answer_text)

            if r.source_docs:
                with st.expander("📄 Source documents"):
                    for doc in r.source_docs:
                        st.write(f"• {doc}")

        # ── Supply Chain / CBAM ───────────────────────────────────────────────
        elif isinstance(r, SupplyChainRiskOutput):
            risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.risk_level, "⚪")
            col1, col2 = st.columns(2)
            col1.metric("Risk Level", f"{risk_color} {r.risk_level}")
            col2.metric("Commodity",  r.commodity)

            if r.key_facts:
                st.write("**Key facts:**")
                for fact in r.key_facts:
                    st.write(f"• {fact}")

            if r.recommended_action:
                st.info(f"**Recommended action:** {r.recommended_action}")

            st.divider()
            st.subheader("Answer")
            st.write(r.answer_text)

            if r.source_docs:
                with st.expander("📄 Source documents"):
                    for doc in r.source_docs:
                        st.write(f"• {doc}")

        # ── Data Analysis ─────────────────────────────────────────────────────
        elif isinstance(r, DataAnalysisOutput):
            col1, col2 = st.columns(2)
            col1.metric("Analysis focus", r.analysis_focus)
            col2.metric("Period",         r.period)

            if r.key_numbers:
                st.write("**Key numbers:**")
                for n in r.key_numbers:
                    st.write(f"• {n}")

            st.divider()
            st.subheader("Answer")
            st.write(r.answer_text)

            if r.chart_path and Path(r.chart_path).exists():
                st.image(r.chart_path)

        # ── Tariff Analysis ───────────────────────────────────────────────────
        elif isinstance(r, TariffAnalysisOutput):
            col1, col2, col3 = st.columns(3)
            col1.metric("HS Codes",   ", ".join(r.hs_codes) or "N/A")
            col2.metric("Trend",      r.trend.capitalize())
            col3.metric("Period",     r.period)

            if r.tariff_rates:
                st.write("**Rates found:**")
                for rate in r.tariff_rates:
                    st.write(f"• {rate}")

            st.divider()
            st.subheader("Answer")
            st.write(r.answer_text)

            if r.chart_path and Path(r.chart_path).exists():
                st.image(r.chart_path)

    elif run:
        st.warning("Please enter a question.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Export Trends
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    @st.cache_resource(show_spinner="Loading export data…")
    def _load_export():
        from data_agent import load_export_data
        return load_export_data()

    df_exp = _load_export()

    months_avail = sorted(df_exp["report_year"].astype(str) + "-" +
                          df_exp["report_month_num"].astype(str).str.zfill(2))
    date_range = f"{months_avail[0]} → {months_avail[-1]}"
    n_countries = df_exp["country"].nunique()

    st.subheader("Export Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Months of data", df_exp[["report_year","report_month_num"]].drop_duplicates().shape[0])
    c2.metric("Countries tracked", n_countries)
    c3.metric("Period", date_range)
    st.write("")

    subtab_top, subtab_trend, subtab_region, subtab_compare = st.tabs([
        "🏆 Top Destinations",
        "📉 Growing & Shrinking",
        "🌍 Regional Breakdown",
        "🔁 Country Comparison",
    ])

    # ── Top destinations ──────────────────────────────────────────────────────
    with subtab_top:
        n_top = st.slider("Number of countries", 5, 20, 10, key="n_top")
        with st.spinner("Computing…"):
            from data_agent import get_latest_top_destinations
            top_df = get_latest_top_destinations(n=n_top)
        if not top_df.empty:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(top_df["country"][::-1], top_df["usd_million"][::-1],
                           color="#3b82f6")
            ax.set_xlabel("USD Million (latest month)")
            ax.set_title(f"Top {n_top} Export Destinations — Latest Month")
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, top_df["usd_million"][::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"${val:.1f}M", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("📋 Data table"):
                st.dataframe(top_df.rename(columns={"usd_million": "USD Million"}),
                             use_container_width=True)

    # ── Growing & Shrinking markets ───────────────────────────────────────────
    with subtab_trend:
        col_l, col_r = st.columns(2)
        lookback = col_l.slider("Lookback months", 3, 12, 6, key="lookback")
        min_usd  = col_r.slider("Min avg exports (USD M)", 0.5, 20.0, 2.0, step=0.5)
        n_mkts   = st.slider("Markets per group", 5, 15, 8, key="n_mkts")

        with st.spinner("Running trend analysis…"):
            from data_agent import get_market_trends, plot_market_trends
            chart_path = str(ROOT / "charts" / "dash_market_trends.png")
            plot_market_trends(lookback_months=lookback, n=n_mkts, save_as=chart_path)

        if Path(chart_path).exists():
            st.image(chart_path)

        with st.expander("📋 Raw trend data"):
            trends_raw = get_market_trends(lookback_months=lookback,
                                           min_avg_usd=min_usd, n=n_mkts*2)
            # get_market_trends returns a dict with keys: period, growing, shrinking, all_trends
            trends_df = trends_raw.get("all_trends") if isinstance(trends_raw, dict) else trends_raw
            if trends_df is not None and not trends_df.empty:
                st.caption(f"Period: {trends_raw.get('period','') if isinstance(trends_raw, dict) else ''}")
                st.dataframe(trends_df[["country","avg_monthly_usd","trend_slope","growth_pct_latest"]].rename(
                    columns={"avg_monthly_usd":"Avg USD M","trend_slope":"Trend slope",
                             "growth_pct_latest":"Latest YoY %"}
                ), use_container_width=True)

    # ── Regional Breakdown ────────────────────────────────────────────────────
    with subtab_region:
        period = st.radio("Period", ["latest", "ytd"], horizontal=True,
                          key="region_period",
                          format_func=lambda x: "Latest Month" if x == "latest" else "Year-to-Date")
        with st.spinner("Building regional chart…"):
            from data_agent import plot_regional_breakdown, get_regional_summary
            chart_path = str(ROOT / "charts" / "dash_regional.png")
            plot_regional_breakdown(period=period, save_as=chart_path)

        if Path(chart_path).exists():
            st.image(chart_path)

        with st.expander("📋 Summary table"):
            reg_df = get_regional_summary(period=period)
            if not reg_df.empty:
                st.dataframe(reg_df, use_container_width=True)

    # ── Country Comparison ────────────────────────────────────────────────────
    with subtab_compare:
        all_countries = sorted(df_exp["country"].dropna().unique().tolist())
        selected = st.multiselect(
            "Select countries to compare",
            options=all_countries,
            default=["CHINA P RP", "U ARAB EMTS", "VIETNAM SOC REP"],
            key="cmp_countries",
        )
        if selected:
            with st.spinner("Generating comparison…"):
                from data_agent import compare_countries
                cmp = compare_countries(selected)  # returns dict: chart_trend, chart_latest, stats

            # compare_countries generates its own charts internally
            chart_trend  = cmp.get("chart_trend")
            chart_latest = cmp.get("chart_latest")
            if chart_trend and Path(chart_trend).exists():
                st.image(chart_trend)
            if chart_latest and Path(chart_latest).exists():
                st.image(chart_latest)

            with st.expander("📋 Stats table"):
                stats_df = cmp.get("stats")
                if stats_df is not None and not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Select at least one country above.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Tariff Lookup
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    @st.cache_resource(show_spinner="Loading MFN tariff data…")
    def _load_tariff():
        from tariff_agent import load_tariff_data
        return load_tariff_data()

    df_tar = _load_tariff()

    years = sorted(df_tar["year"].unique())
    st.subheader("MFN Tariff Analysis — HS 72 & 73 (2010-2023)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Years covered",   f"{years[0]}–{years[-1]}")
    c2.metric("Unique HS codes", df_tar["hs6"].nunique())
    c3.metric("Chapters",        "HS 72 + HS 73")
    st.write("")

    tar_tab1, tar_tab2, tar_tab3, tar_tab4 = st.tabs([
        "📈 Chapter Trends",
        "🔎 HS Code Lookup",
        "🏆 Highest Tariff Products",
        "💬 Ask the Tariff Agent",
    ])

    # ── Chapter trends ────────────────────────────────────────────────────────
    with tar_tab1:
        with st.spinner("Rendering chapter trend chart…"):
            from tariff_agent import plot_chapter_comparison
            chart_path = str(ROOT / "charts" / "dash_mfn_chapters.png")
            plot_chapter_comparison(save_as=chart_path)
        if Path(chart_path).exists():
            st.image(chart_path)

        with st.expander("📋 Chapter summary table"):
            from tariff_agent import get_chapter_summary
            ch_df = get_chapter_summary()
            if not ch_df.empty:
                st.dataframe(ch_df, use_container_width=True)

    # ── HS Code lookup ────────────────────────────────────────────────────────
    with tar_tab2:
        col_a, col_b = st.columns([2, 1])
        hs_input = col_a.text_input("HS code (4 or 6 digit)", value="7208",
                                    placeholder="e.g. 7208, 730410")
        year_sel = col_b.selectbox("Year", ["All"] + [str(y) for y in years[::-1]],
                                   key="hs_year")

        if hs_input.strip():
            from tariff_agent import get_tariff, get_tariff_trend, plot_tariff_trend, hs_description
            hs = hs_input.strip()
            desc = hs_description(hs)
            if desc:
                st.caption(f"📦 {desc}")

            year_arg = int(year_sel) if year_sel != "All" else None
            result = get_tariff(hs, year=year_arg)
            if not result.empty:
                st.dataframe(result[["year","hs6","avg_rate","min_rate","max_rate","description"]].rename(
                    columns={"avg_rate":"Avg %","min_rate":"Min %","max_rate":"Max %"}
                ), use_container_width=True)

                # Trend chart
                trend = get_tariff_trend(hs)
                if not trend.empty and len(trend) > 1:
                    chart_path = str(ROOT / "charts" / f"dash_trend_{hs}.png")
                    plot_tariff_trend([hs], save_as=chart_path)
                    if Path(chart_path).exists():
                        st.image(chart_path)
            else:
                st.warning(f"No data found for HS {hs}" +
                           (f" in {year_sel}" if year_sel != "All" else ""))

            # Keyword search fallback
            with st.expander("🔍 Search by product name instead"):
                keyword = st.text_input("Product keyword", placeholder="e.g. seamless tube, galvanized",
                                        key="kw_search")
                if keyword:
                    from tariff_agent import search_by_product
                    matches = search_by_product(keyword)
                    if not matches.empty:
                        st.dataframe(matches[["hs6","description","avg_rate","year"]].rename(
                            columns={"avg_rate":"Avg %"}
                        ), use_container_width=True)
                    else:
                        st.info("No matches found.")

    # ── Highest tariff products ───────────────────────────────────────────────
    with tar_tab3:
        col_a, col_b, col_c = st.columns(3)
        year_top = col_a.selectbox("Year", [str(y) for y in years[::-1]], key="top_year")
        chap_top = col_b.selectbox("Chapter", ["Both", "72 (Iron & Steel)", "73 (Articles)"],
                                   key="top_chap")
        n_top2   = col_c.slider("Show top N", 5, 20, 10, key="n_top2")

        chap_map = {"Both": None, "72 (Iron & Steel)": 72, "73 (Articles)": 73}
        from tariff_agent import get_top_tariff_products
        top_t = get_top_tariff_products(int(year_top), n=n_top2,
                                        chapter=chap_map[chap_top])
        if not top_t.empty:
            st.dataframe(top_t[["hs6","heading","avg_rate","max_rate","description"]].rename(
                columns={"avg_rate":"Avg %","max_rate":"Max %"}
            ), use_container_width=True)

            # Bar chart
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            short_labels = [f"HS {r['hs6']}" for _, r in top_t.iterrows()]
            ax.barh(short_labels[::-1], top_t["avg_rate"].values[::-1], color="#8b5cf6")
            ax.set_xlabel("Avg MFN Rate (%)")
            ax.set_title(f"Top {n_top2} Highest MFN Tariff Products ({year_top})")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Ask the Tariff Agent ──────────────────────────────────────────────────
    with tar_tab4:
        tar_q = st.text_area(
            "Ask a tariff question",
            height=80,
            placeholder="e.g. How has India's MFN duty on galvanized steel changed since 2015?",
            key="tar_q",
        )
        tar_run = st.button("Ask Tariff Agent ›", type="primary", key="tar_btn")

        if tar_run and tar_q.strip():
            with st.spinner("Querying tariff agent…"):
                from tariff_agent import query_tariff
                tar_result = query_tariff(tar_q.strip())

            st.write(tar_result["answer"])
            if tar_result.get("chart_path") and Path(tar_result["chart_path"]).exists():
                st.image(tar_result["chart_path"])
            if tar_result.get("error"):
                with st.expander("⚠️ Execution error"):
                    st.code(tar_result["error"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Eval Report
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    EVAL_DIR = ROOT / "eval"
    v1_path  = EVAL_DIR / "baseline_v1.json"
    v1b_path = EVAL_DIR / "baseline_v1b.json"

    def _load_eval(path: Path) -> dict | None:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    v1  = _load_eval(v1_path)
    v1b = _load_eval(v1b_path)

    st.subheader("RAG Evaluation — Baseline Comparison")

    if v1 and v1b:
        s1  = v1["summary"];  r1  = {r["id"]: r for r in v1["results"]}
        s1b = v1b["summary"]; r1b = {r["id"]: r for r in v1b["results"]}

        total = s1b.get("total", 10)

        # ── Summary metrics ───────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Answered (v1)",  f"{s1['answered']}/{total}")
        col2.metric("Answered (v1b)", f"{s1b['answered']}/{total}",
                    delta=f"{s1b['answered'] - s1['answered']:+d}")
        col3.metric("Source hit rate (v1b)",
                    f"{s1b.get('source_hit_rate', 0):.0%}")
        col4.metric("Avg latency (v1b)",
                    f"{s1b.get('avg_latency_ms', 0):,}ms",
                    delta=f"{s1b.get('avg_latency_ms',0) - s1.get('avg_latency_ms',3627):+,}ms",
                    delta_color="inverse")

        st.write("")

        # ── Question table ────────────────────────────────────────────────────
        st.subheader("Question-level results")
        import pandas as pd
        rows = []
        for qid in sorted(r1b.keys()):
            a  = r1.get(qid,  {})
            b  = r1b[qid]
            changed = (a.get("answered") != b["answered"])
            rows.append({
                "ID":       qid,
                "Type":     b["type"],
                "Question": b["question"][:70] + "…",
                "v1 answered":  "✓" if a.get("answered") else "✗",
                "v1b answered": "✓" if b["answered"] else "✗",
                "Changed":  "⬆️ Fixed" if (b["answered"] and not a.get("answered"))
                            else ("⬇️ Broke" if (not b["answered"] and a.get("answered"))
                                  else "—"),
                "Source hit":  "✓" if b.get("source_hit") else "✗",
                "Latency ms":  b.get("latency_ms", "—"),
            })
        df_eval = pd.DataFrame(rows).set_index("ID")
        st.dataframe(df_eval, use_container_width=True)

        # ── Refused questions deep-dive ───────────────────────────────────────
        refused = [r for r in v1b["results"] if not r["answered"]]
        if refused:
            st.write("")
            st.subheader(f"⚠️ Still refused ({len(refused)} questions)")
            for r in refused:
                with st.expander(f"Q{r['id']}: {r['question'][:80]}"):
                    st.write(f"**Type:** {r['type']}")
                    st.write(f"**Top source retrieved:** {r['top_source']}")
                    st.write(f"**Expected source:** {r.get('expected_src','—')}")
                    st.info(
                        "This question requires content not present in the current "
                        "corpus. Consider adding the specific source document."
                    )

    elif v1b:
        st.info("baseline_v1.json not found — showing v1b only.")
        s = v1b["summary"]
        st.metric("Answered", f"{s['answered']}/{s.get('total',10)}")
        st.metric("Source hit rate", f"{s.get('source_hit_rate',0):.0%}")
    else:
        st.warning("No eval results found. Run `python eval/run_eval.py` first.")
