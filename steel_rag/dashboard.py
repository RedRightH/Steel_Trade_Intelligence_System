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

import os
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

# ── Live news pipeline status (sidebar) ───────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Live News Feed")
    _pipeline_log = ROOT / "pipeline_log.json"
    if _pipeline_log.exists():
        try:
            import json as _json
            _log = _json.loads(_pipeline_log.read_text(encoding="utf-8"))
            _runs = _log.get("runs", [])
            _seen = _log.get("seen_ids", [])
            if _runs:
                _last = _runs[-1]
                _ts   = _last["timestamp"][:16].replace("T", " ")
                st.success(f"Last run: **{_ts} UTC**")
                st.metric("Articles indexed", len(_seen))
                st.metric("Latest batch", f"{_last['upserted']} new")
                # Show most recent articles
                _recent = _last.get("articles", [])[:5]
                if _recent:
                    st.markdown("**Recent articles:**")
                    for _a in _recent:
                        st.markdown(
                            f"- [{_a['title'][:55]}…]({_a['url']})",
                            unsafe_allow_html=False,
                        )
            else:
                st.info("No runs recorded yet.")
        except Exception:
            st.warning("Could not read pipeline log.")
    else:
        st.info("Pipeline not yet run.\n\n"
                "```\npython steel_rag/classifier_pipeline.py --once\n```")
    st.divider()
    st.markdown("**Vector backend**")
    try:
        _use_pc = bool(os.getenv("PINECONE_API_KEY", "").strip())
        st.markdown(f"{'🟢 Pinecone' if _use_pc else '🟡 FAISS (local)'}")
    except Exception:
        pass

st.divider()

# ── Eagerly warm up RAG models on first load (prevents 100s cold-start) ────────
@st.cache_resource(show_spinner="Warming up RAG models…")
def _warmup_rag():
    try:
        from rag import warmup
        warmup()
    except Exception as e:
        pass  # Non-fatal — queries still work, just slower on first call

_warmup_rag()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Intelligence Query",
    "🔄 Trade Flows",
    "📊 Tariff Lookup",
    "🧪 Eval Report",
    "📈 Futures & Impact",
    "🌐 Gravity Scenarios",
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

# ── helper: render one assistant turn ─────────────────────────────────────────
def _render_assistant_turn(ro):
    from router import PolicyAnalystOutput, SupplyChainRiskOutput, DataAnalysisOutput, TariffAnalysisOutput
    r = ro["result_obj"]
    badge_cls, badge_label = BADGE_MAP.get(ro["question_type"], ("badge-data", ro["question_type"]))
    st.markdown(
        f'<span class="badge {badge_cls}">{badge_label}</span>'
        f'<span style="color:#94a3b8;font-size:12px;">via {ro["agent_used"]} · {ro["latency_ms"]:,}ms</span>',
        unsafe_allow_html=True,
    )

    if isinstance(r, PolicyAnalystOutput):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duty Type",      r.duty_type)
        col2.metric("Duty Rate",      r.duty_rate)
        col3.metric("Effective Date", r.effective_date)
        col4.metric("Confidence",     f"{r.confidence:.0%}")
        if r.countries:
            st.write("**Countries:**", ", ".join(r.countries))
        st.write(r.answer_text)
        if r.source_docs:
            with st.expander("📄 Sources"):
                for doc in r.source_docs: st.write(f"• {doc}")

    elif isinstance(r, SupplyChainRiskOutput):
        risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.risk_level, "⚪")
        col1, col2 = st.columns(2)
        col1.metric("Risk Level", f"{risk_color} {r.risk_level}")
        col2.metric("Commodity",  r.commodity)
        if r.key_facts:
            for fact in r.key_facts: st.write(f"• {fact}")
        if r.recommended_action:
            st.info(f"**Recommended action:** {r.recommended_action}")
        st.write(r.answer_text)
        if r.source_docs:
            with st.expander("📄 Sources"):
                for doc in r.source_docs: st.write(f"• {doc}")

    elif isinstance(r, DataAnalysisOutput):
        col1, col2 = st.columns(2)
        col1.metric("Focus", r.analysis_focus)
        col2.metric("Period", r.period)
        if r.key_numbers:
            for n in r.key_numbers: st.write(f"• {n}")
        st.write(r.answer_text)
        if r.chart_path and Path(r.chart_path).exists():
            st.image(r.chart_path)

    elif isinstance(r, TariffAnalysisOutput):
        col1, col2, col3 = st.columns(3)
        col1.metric("HS Codes", ", ".join(r.hs_codes) or "N/A")
        col2.metric("Trend",    r.trend.capitalize())
        col3.metric("Period",   r.period)
        if r.tariff_rates:
            for rate in r.tariff_rates: st.write(f"• {rate}")
        st.write(r.answer_text)
        if r.chart_path and Path(r.chart_path).exists():
            st.image(r.chart_path)


with tab1:
    from router import route_query
    from memory import ConversationMemory

    # ── session state init ────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {role, content, ro_dict}
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=5)

    # ── top bar: example picker + clear button ────────────────────────────────
    col_ex, col_clear = st.columns([5, 1])
    with col_ex:
        example = st.selectbox("Example questions", EXAMPLE_QUESTIONS,
                               label_visibility="collapsed", key="chat_example")
    with col_clear:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()

    # ── memory indicator ──────────────────────────────────────────────────────
    mem = st.session_state.memory
    if not mem.is_empty:
        st.caption(f"💬 {mem.turn_count} turn{'s' if mem.turn_count > 1 else ''} in memory · "
                   f"Last topic: **{mem.last_type}**")

    st.divider()

    # ── replay existing chat history ──────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                _render_assistant_turn(msg["ro"])

    # ── chat input ────────────────────────────────────────────────────────────
    # Pre-fill from example picker if user selected one
    prefill = "" if example.startswith("Select") else example
    question = st.chat_input("Ask anything about India's steel trade…")

    # If no chat input but example selected and different from last question
    if not question and prefill:
        last_q = (st.session_state.chat_history[-1].get("content", "")
                  if st.session_state.chat_history else "")
        if prefill != last_q:
            question = prefill

    if question:
        # Show user bubble
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Run router with memory
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ro = route_query(question.strip(), memory=st.session_state.memory)

            # Store serialisable snapshot for replay
            ro_dict = {
                "question":      ro.question,
                "question_type": ro.question_type,
                "agent_used":    ro.agent_used,
                "latency_ms":    ro.latency_ms,
                "result_obj":    ro.result,   # Pydantic object — stays in session RAM
            }
            _render_assistant_turn(ro_dict)

        # Update memory and history
        st.session_state.memory.add(
            question          = ro.question,
            answer            = ro.result.answer_text,
            question_type     = ro.question_type,
            agent_used        = ro.agent_used,
        )
        st.session_state.chat_history.append({"role": "assistant", "ro": ro_dict})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trade Flows (Exports + Imports + Balance)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    @st.cache_resource(show_spinner="Loading export data…")
    def _load_export():
        from data_agent import load_export_data
        return load_export_data()

    @st.cache_resource(show_spinner="Loading import data…")
    def _load_import():
        from data_agent import load_import_data
        return load_import_data()

    df_exp = _load_export()
    df_imp = _load_import()

    # ── Top-line KPIs ─────────────────────────────────────────────────────────
    from data_agent import get_yoy_summary, get_import_yoy_summary
    exp_kpi = get_yoy_summary()
    imp_kpi = get_import_yoy_summary()

    st.subheader("India Steel Trade — Latest Month Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exports",
              f"${exp_kpi['monthly_curr_usd_mn']:.0f}M",
              f"{exp_kpi['monthly_growth_pct']:+.1f}% YoY" if exp_kpi['monthly_growth_pct'] else None)
    c2.metric("Imports",
              f"${imp_kpi['monthly_curr_usd_mn']:.0f}M",
              f"{imp_kpi['monthly_growth_pct']:+.1f}% YoY" if imp_kpi['monthly_growth_pct'] else None)
    trade_balance = exp_kpi['monthly_curr_usd_mn'] - imp_kpi['monthly_curr_usd_mn']
    c3.metric("Trade Balance",
              f"${trade_balance:+.0f}M",
              "Surplus" if trade_balance >= 0 else "Deficit")
    c4.metric("Latest month", exp_kpi['latest_month'])
    st.write("")

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    (subtab_exp, subtab_imp, subtab_balance,
     subtab_trend, subtab_region, subtab_compare) = st.tabs([
        "📤 Exports",
        "📥 Imports",
        "⚖️ Trade Balance",
        "📉 Market Trends",
        "🌍 Regional Breakdown",
        "🔁 Country Comparison",
    ])

    # ── Exports ───────────────────────────────────────────────────────────────
    with subtab_exp:
        n_top = st.slider("Number of countries", 5, 20, 10, key="n_top")
        with st.spinner("Computing…"):
            from data_agent import get_latest_top_destinations
            top_df = get_latest_top_destinations(n=n_top)
        if not top_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(top_df["country"][::-1], top_df["usd_million"][::-1],
                           color="#3b82f6")
            ax.set_xlabel("USD Million (latest month)")
            ax.set_title(f"Top {n_top} Export Destinations — {exp_kpi['latest_month']}")
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

    # ── Imports ───────────────────────────────────────────────────────────────
    with subtab_imp:
        n_imp = st.slider("Number of source countries", 5, 20, 10, key="n_imp")
        with st.spinner("Computing…"):
            from data_agent import get_latest_top_sources
            imp_df = get_latest_top_sources(n=n_imp)
        if not imp_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(imp_df["country"][::-1], imp_df["usd_million"][::-1],
                           color="#ef4444")
            ax.set_xlabel("USD Million (latest month)")
            ax.set_title(f"Top {n_imp} Steel Import Sources — {imp_kpi['latest_month']}")
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, imp_df["usd_million"][::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"${val:.1f}M", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(f"YTD imports: ${imp_kpi['ytd_curr_usd_mn']:.0f}M "
                       f"({imp_kpi['ytd_growth_pct']:+.1f}% vs prev FY)" if imp_kpi['ytd_growth_pct'] else "")
            with st.expander("📋 Data table"):
                st.dataframe(imp_df.rename(columns={"usd_million": "USD Million"}),
                             use_container_width=True)

    # ── Trade Balance ─────────────────────────────────────────────────────────
    with subtab_balance:
        bal_period = st.radio("Period", ["latest_month", "ytd"], horizontal=True,
                              key="bal_period",
                              format_func=lambda x: "Latest Month" if x == "latest_month" else "Year-to-Date")
        n_bal = st.slider("Countries to show", 10, 30, 15, key="n_bal")
        with st.spinner("Computing trade balance…"):
            from data_agent import get_trade_balance
            bal_df = get_trade_balance(period=bal_period)

        if not bal_df.empty:
            # Show top surplus and deficit countries
            top_surplus  = bal_df.head(n_bal // 2)
            top_deficit  = bal_df.tail(n_bal // 2).sort_values("balance_usd")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Surplus
            ax1.barh(top_surplus["country"], top_surplus["balance_usd"], color="#10b981")
            ax1.set_title(f"Top Surplus Countries\n(Exports > Imports)", fontsize=11)
            ax1.set_xlabel("USD Million")
            ax1.axvline(0, color="black", linewidth=0.8)
            ax1.spines[["top","right"]].set_visible(False)

            # Deficit
            ax2.barh(top_deficit["country"], top_deficit["balance_usd"], color="#ef4444")
            ax2.set_title(f"Top Deficit Countries\n(Imports > Exports)", fontsize=11)
            ax2.set_xlabel("USD Million")
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.spines[["top","right"]].set_visible(False)

            fig.suptitle(f"India Steel Trade Balance — "
                         f"{'Latest Month' if bal_period=='latest_month' else 'YTD'}", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Full trade balance table"):
                st.dataframe(
                    bal_df[["country","continent","exports_usd","imports_usd","balance_usd"]]
                    .rename(columns={"exports_usd":"Exports $M","imports_usd":"Imports $M",
                                     "balance_usd":"Balance $M","continent":"Continent"}),
                    use_container_width=True
                )

    # ── Market Trends ─────────────────────────────────────────────────────────
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
            trends_df = trends_raw.get("all_trends") if isinstance(trends_raw, dict) else trends_raw
            if trends_df is not None and not trends_df.empty:
                st.caption(f"Period: {trends_raw.get('period','') if isinstance(trends_raw, dict) else ''}")
                st.dataframe(trends_df[["country","avg_monthly_usd","trend_slope","growth_pct_latest"]].rename(
                    columns={"avg_monthly_usd":"Avg USD M","trend_slope":"Trend slope",
                             "growth_pct_latest":"Latest YoY %"}
                ), use_container_width=True)

    # ── Regional Breakdown ────────────────────────────────────────────────────
    with subtab_region:
        period = st.radio("Period", ["latest_month", "ytd"], horizontal=True,
                          key="region_period",
                          format_func=lambda x: "Latest Month" if x == "latest_month" else "Year-to-Date")
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
                cmp = compare_countries(selected)

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
    import pandas as pd
    EVAL_DIR_ = ROOT / "eval"

    def _load_eval(path: Path) -> dict | None:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    v1b = _load_eval(EVAL_DIR_ / "baseline_v1b.json")
    v2  = _load_eval(EVAL_DIR_ / "baseline_v2.json")

    st.subheader("RAG Evaluation")

    # ── Run selector ─────────────────────────────────────────────────────────
    runs = {}
    if v1b: runs["v1b — 10 questions (no judge)"] = v1b
    if v2:  runs["v2 — 25 questions + LLM judge"]  = v2
    selected_run = st.selectbox("Select eval run", list(runs.keys())) if runs else None

    if not runs:
        st.warning("No eval results found. Run `python eval/run_eval.py --tag v2 --gt v2 --judge`")
    else:
        ev = runs[selected_run]
        s  = ev["summary"]
        results = ev["results"]

        # ── Top-line metrics ──────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Answered",         f"{s['answered']}/{s['total']}")
        col2.metric("Answered rate",    f"{s['answered_rate']:.0%}")
        col3.metric("Source hit rate",  f"{s.get('source_hit_rate',0) or 0:.0%}")
        col4.metric("Routing accuracy", f"{s.get('routing_accuracy') or 0:.0%}"
                    if s.get("routing_accuracy") else "—")
        col5.metric("Avg latency",      f"{s.get('avg_latency_ms',0):,}ms")

        # ── Judge scores (v2 only) ────────────────────────────────────────────
        if s.get("avg_judge_score") is not None:
            st.write("")
            jc1, jc2, jc3, jc4 = st.columns(4)
            jc1.metric("Avg judge score",  f"{s['avg_judge_score']:.2f} / 1.0")
            jc2.metric("Faithfulness",     f"{s['avg_faithfulness']:.2f}")
            jc3.metric("Relevance",        f"{s['avg_relevance']:.2f}")
            jc4.metric("Completeness",     f"{s['avg_completeness']:.2f}")

            # ── Judge score chart by question type ────────────────────────────
            judged = [r for r in results if r.get("judge_score") is not None]
            if judged:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                types = sorted(set(r.get("question_type","?") for r in judged))
                f_scores = [sum(r["faithfulness"]  for r in judged if r.get("question_type")==t)
                            / max(sum(1 for r in judged if r.get("question_type")==t),1)
                            for t in types]
                c_scores = [sum(r["completeness"]  for r in judged if r.get("question_type")==t)
                            / max(sum(1 for r in judged if r.get("question_type")==t),1)
                            for t in types]

                x = range(len(types))
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar([i-0.2 for i in x], f_scores, 0.35, label="Faithfulness", color="#3b82f6")
                ax.bar([i+0.15 for i in x], c_scores, 0.35, label="Completeness", color="#f59e0b")
                ax.set_xticks(list(x)); ax.set_xticklabels(types, rotation=20, ha="right")
                ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
                ax.set_title("Judge scores by question type")
                ax.legend(); ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

        st.write("")
        st.subheader("Question-level results")

        # ── Build table ───────────────────────────────────────────────────────
        rows = []
        for r in sorted(results, key=lambda x: x["id"]):
            row = {
                "ID":       r["id"],
                "Q-Type":   r.get("question_type","")[:16],
                "Type":     r.get("type",""),
                "Answered": "✓" if r["answered"] else "✗",
                "Src hit":  ("✓" if r.get("source_hit") else "✗") if r.get("source_hit") is not None else "—",
                "Route":    ("✓" if r.get("routing_correct") else "✗") if r.get("routing_correct") is not None else "—",
                "Judge":    f"{r['judge_score']:.2f}" if r.get("judge_score") is not None else "—",
                "Faith":    f"{r['faithfulness']:.2f}" if r.get("faithfulness") is not None else "—",
                "Complete": f"{r['completeness']:.2f}" if r.get("completeness") is not None else "—",
                "ms":       r.get("latency_ms","—"),
                "Question": r["question"][:65] + "…",
            }
            rows.append(row)

        df_eval = pd.DataFrame(rows).set_index("ID")
        st.dataframe(df_eval, use_container_width=True)

        # ── Refused questions ─────────────────────────────────────────────────
        refused = [r for r in results if not r["answered"]]
        if refused:
            st.write("")
            st.subheader(f"⚠️ Refused questions ({len(refused)})")
            for r in refused:
                with st.expander(f"Q{r['id']} [{r.get('question_type','')}]: {r['question'][:75]}"):
                    st.write(f"**Expected source:** {r.get('expected_src','—')}")
                    st.write(f"**Top retrieved:** {r.get('top_source','—')}")
                    st.info("Add the source document to the corpus and re-run ingest.py to fix this.")

        # ── Low-scoring answered questions ────────────────────────────────────
        low = [r for r in results if r.get("judge_score") is not None and r["judge_score"] < 0.6]
        if low:
            st.write("")
            st.subheader(f"📉 Low judge score (< 0.60) — {len(low)} questions")
            for r in sorted(low, key=lambda x: x["judge_score"]):
                with st.expander(f"Q{r['id']} score={r['judge_score']:.2f}: {r['question'][:70]}"):
                    st.write(f"**Faithfulness:** {r['faithfulness']:.2f}  "
                             f"**Relevance:** {r['relevance']:.2f}  "
                             f"**Completeness:** {r['completeness']:.2f}")
                    st.write(f"**Judge reasoning:** {r.get('judge_reason','')}")
                    st.write(f"**Expected:** {r.get('expected_ans','')[:300]}")
                    st.write(f"**Actual:** {r.get('actual_answer','')[:300]}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Futures & News Impact
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("📈 Steel Futures & News Impact Analyzer")

    @st.cache_data(ttl=3600, show_spinner="Fetching steel market data…")
    def _load_futures_snapshot():
        from steel_futures import get_futures_snapshot
        return get_futures_snapshot()

    @st.cache_data(ttl=3600, show_spinner="Loading price history…")
    def _load_ticker_history(ticker: str):
        from steel_futures import fetch_ticker, compute_technicals
        df = fetch_ticker(ticker, period="1y")
        if not df.empty:
            return compute_technicals(df)
        return df

    # ── Top KPI row ───────────────────────────────────────────────────────────
    try:
        snap = _load_futures_snapshot()
        prices = snap.get("prices", {})
        fc = snap.get("hrc_forecast", {})

        # KPI cards — HRC futures first, then Indian stocks
        priority = ["HRC=F", "TATASTEEL.NS", "SAIL.NS", "JSWSTEEL.NS", "SLX", "MT", "NUE"]
        ordered  = [t for t in priority if t in prices] + [t for t in prices if t not in priority]
        cols = st.columns(len(ordered))
        for col, ticker in zip(cols, ordered):
            p = prices[ticker]
            arrow = "▲" if p["chg_pct_1d"] >= 0 else "▼"
            color = "green" if p["chg_pct_1d"] >= 0 else "red"
            col.metric(
                label=f"{p['name']}",
                value=f"{p['last']:,.2f} {p['currency']}",
                delta=f"{arrow} {p['chg_pct_1d']:+.2f}% (1d)",
            )

        st.caption(f"Data via Yahoo Finance · Last updated: {snap['last_updated'][:16].replace('T',' ')} UTC")
        st.divider()
    except Exception as e:
        st.error(f"Could not load futures snapshot: {e}")
        st.stop()

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    ft1, ft2, ft3, ft4 = st.tabs([
        "📊 Price Charts & Forecast",
        "📰 News Impact Analyzer",
        "🔮 Scenario Comparison",
        "📋 Articles Scored",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # Sub-tab 1 — Price Charts & Forecast
    # ────────────────────────────────────────────────────────────────────────
    with ft1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import pandas as pd
        from steel_futures import FORECAST_DAYS

        ticker_opts = {v["name"]: k for k, v in
                       __import__("steel_futures", fromlist=["TICKERS"]).TICKERS.items()
                       if k in prices}
        sel_name   = st.selectbox("Select instrument", list(ticker_opts.keys()), key="fut_sel")
        sel_ticker = ticker_opts[sel_name]

        df_t = _load_ticker_history(sel_ticker)

        if not df_t.empty:
            # ── Price + MA chart ──────────────────────────────────────────────
            fig = plt.figure(figsize=(12, 8))
            gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)

            close = df_t["Close"].dropna()
            ax1.plot(close.index, close, color="#1e40af", linewidth=1.4, label="Close")
            if "MA20" in df_t.columns:
                ax1.plot(df_t.index, df_t["MA20"], color="#f59e0b", linewidth=1, linestyle="--", label="MA20")
                ax1.plot(df_t.index, df_t["MA50"], color="#ef4444", linewidth=1, linestyle="--", label="MA50")
                ax1.fill_between(df_t.index, df_t["BB_lower"], df_t["BB_upper"],
                                 alpha=0.08, color="#6366f1", label="BB ±2σ")

            # Add Prophet forecast if this is HRC=F
            if sel_ticker == "HRC=F" and fc:
                fc_df = fc.get("forecast")
                if fc_df is not None:
                    future_mask = fc_df["ds"] > pd.Timestamp(close.index[-1]).tz_localize(None)
                    fc_future = fc_df[future_mask]
                    ax1.plot(fc_future["ds"], fc_future["yhat"],
                             color="#10b981", linewidth=1.5, linestyle="-", label=f"Prophet {FORECAST_DAYS}d")
                    ax1.fill_between(fc_future["ds"], fc_future["yhat_lower"], fc_future["yhat_upper"],
                                     alpha=0.15, color="#10b981", label="80% CI")
                    # Annotate 30/60 day targets
                    if fc.get("target_30"):
                        ax1.axhline(fc["target_30"], color="#10b981", linewidth=0.7, linestyle=":")
                        ax1.text(fc_future["ds"].iloc[-1], fc["target_30"],
                                 f" 60d: {fc['target_60']:.0f}", color="#10b981", fontsize=8, va="center")

            ax1.set_ylabel("Price")
            ax1.legend(fontsize=8, loc="upper left")
            ax1.set_title(f"{sel_name} — 1 Year", fontsize=11)
            ax1.spines[["top","right"]].set_visible(False)
            plt.setp(ax1.get_xticklabels(), visible=False)

            # Volume
            vol_colors = ["#10b981" if c >= o else "#ef4444"
                          for c, o in zip(df_t["Close"], df_t["Open"])]
            ax2.bar(df_t.index, df_t["Volume"], color=vol_colors, width=1, alpha=0.7)
            ax2.set_ylabel("Volume", fontsize=8)
            ax2.spines[["top","right"]].set_visible(False)
            plt.setp(ax2.get_xticklabels(), visible=False)

            # RSI
            if "RSI14" in df_t.columns:
                ax3.plot(df_t.index, df_t["RSI14"], color="#8b5cf6", linewidth=1)
                ax3.axhline(70, color="#ef4444", linewidth=0.6, linestyle="--")
                ax3.axhline(30, color="#10b981", linewidth=0.6, linestyle="--")
                ax3.fill_between(df_t.index, 70, 100, alpha=0.05, color="#ef4444")
                ax3.fill_between(df_t.index, 0, 30, alpha=0.05, color="#10b981")
                ax3.set_ylim(0, 100)
                ax3.set_ylabel("RSI-14", fontsize=8)
                ax3.spines[["top","right"]].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Forecast summary card (HRC only)
            if sel_ticker == "HRC=F" and fc:
                trend_icon = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(fc["trend"], "")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Current HRC",  f"${fc['current']:.0f}",  "USD/short ton")
                fc2.metric("30-day target", f"${fc['target_30']:.0f}", f"{fc['change_pct_30']:+.1f}%")
                fc3.metric("60-day target", f"${fc['target_60']:.0f}", f"{fc['change_pct_60']:+.1f}%")
                fc4.metric("Trend", f"{trend_icon} {fc['trend'].title()}")
                if fc.get("gpr_used"):
                    coef = fc.get("gpr_coef")
                    coef_txt = f" · regressor β = {coef:+.3f} USD/index-pt" if coef is not None else ""
                    st.caption(
                        f"🌐 Forecast conditioned on the Steel-GPR news risk index "
                        f"(future level held at {fc.get('gpr_future_level', 100):.0f}){coef_txt}. "
                        f"The index is built daily from RSS articles scored by the 3-layer AI-GPR pipeline."
                    )

    # ────────────────────────────────────────────────────────────────────────
    # Sub-tab 2 — News Impact Analyzer
    # ────────────────────────────────────────────────────────────────────────
    with ft2:
        st.markdown(
            "Paste any steel trade announcement — tariff change, antidumping ruling, "
            "supply disruption, trade agreement — and the system will quantify the "
            "expected impact on HRC futures prices and India's trade flows."
        )

        EXAMPLE_ANNOUNCEMENTS = [
            "Select an example…",
            "India imposes 25% safeguard duty on HRC imports for 200 days due to surge in Chinese steel",
            "US Section 232 steel tariffs increased to 50% on imports from all countries",
            "China announces 15% cut in steel production capacity in Q3 2026",
            "India and UAE sign comprehensive steel trade agreement, zero duty on bilateral steel trade",
            "BHP reports 20% disruption in Australian iron ore exports due to Cyclone",
            "EU CBAM carbon adjustment mechanism raises effective cost for Indian steel exporters by 12%",
        ]

        col_ann, col_eg = st.columns([3, 2])
        with col_eg:
            eg = st.selectbox("Load example", EXAMPLE_ANNOUNCEMENTS, key="imp_eg")
        with col_ann:
            default_text = "" if eg.startswith("Select") else eg
            announcement = st.text_area(
                "Announcement text",
                value=default_text,
                height=120,
                placeholder="Paste or type a steel trade announcement…",
                key="ann_text",
            )

        use_rag = st.checkbox(
            "🔍 Enrich with RAG context (retrieves historical analogues from document corpus)",
            value=True, key="use_rag_impact"
        )
        run_btn = st.button("⚡ Analyse Impact", type="primary", key="run_impact")

        if run_btn and announcement.strip():
            with st.spinner("Analysing announcement…"):
                try:
                    from steel_futures import analyze_news_impact, analyze_news_impact_with_rag
                    if use_rag:
                        impact = analyze_news_impact_with_rag(announcement.strip())
                    else:
                        impact = analyze_news_impact(announcement.strip())
                except Exception as e:
                    impact = {"error": str(e)}

            if "error" in impact:
                st.error(f"Analysis failed: {impact['error']}")
            else:
                # ── Summary banner ────────────────────────────────────────────
                spill = impact.get("india_spillover", {}) or {}
                role  = spill.get("india_role", "neutral")
                dir_color = {"beneficiary": "🟢", "victim": "🔴", "neutral": "🟡"}.get(role, "⚪")
                st.success(f"{dir_color} **{impact.get('event_type','OTHER').replace('_',' ').title()}** "
                           f"· Risk score {impact.get('risk_score', 0):.2f}/1.0 "
                           f"· Persistence: {impact.get('persistence','—')} "
                           f"· India: {role}")
                if impact.get("summary"):
                    st.info(f"**Summary:** {impact['summary']}")

                # ── KPI row ───────────────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Event Type",      impact.get("event_type", "—").replace("_", " ").title())
                k2.metric("Futures Δ",       f"{impact['futures_impact_pct']:+.1f}%",
                          "Calibrated HRC impact" if impact.get("calibration_factor")
                          else "Base-case HRC impact")
                k3.metric("Trade Flow Δ",    f"{impact['trade_flow_impact_pct']:+.1f}%",
                          "India export flow impact")
                k4.metric("HRC Current",     f"${impact.get('current_hrc_usd') or 0:.0f}")

                if impact.get("calibration_factor"):
                    st.caption(
                        f"⚖️ Event-study calibration applied: ×{impact['calibration_factor']:.2f} "
                        f"(from observed HRC=F abnormal returns around historical "
                        f"{impact.get('event_type','').replace('_',' ').lower()} events — "
                        f"see eval/event_study_results.json)"
                    )

                affected = (impact.get("respondent_countries", [])
                            + impact.get("initiator_countries", []))
                if affected:
                    st.write("**Countries involved:**", ", ".join(affected))
                if impact.get("affected_products"):
                    st.write("**Products:**", ", ".join(impact["affected_products"]))
                if spill.get("india_spillover_summary"):
                    st.write("**India spillover:**", spill["india_spillover_summary"])

                # ── Scenario table ────────────────────────────────────────────
                st.divider()
                st.markdown("#### Bull / Base / Bear Scenarios")
                import pandas as pd
                sc_df = pd.DataFrame(impact["scenarios"])
                sc_df.columns = ["Scenario", "Futures Δ%", "HRC Est. (USD)", "Trade Flow Δ%"]
                st.dataframe(sc_df.set_index("Scenario"), use_container_width=True)

                # ── Visualise scenarios ───────────────────────────────────────
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(10, 4))
                scenarios = [s["scenario"] for s in impact["scenarios"]]
                fut_vals  = [s["futures_chg_pct"] for s in impact["scenarios"]]
                trd_vals  = [s["trade_flow_chg_pct"] for s in impact["scenarios"]]
                colors    = ["#10b981" if v >= 0 else "#ef4444" for v in fut_vals]
                tcolors   = ["#10b981" if v >= 0 else "#ef4444" for v in trd_vals]

                ax_f.barh(scenarios, fut_vals, color=colors)
                ax_f.axvline(0, color="black", linewidth=0.8)
                ax_f.set_title("HRC Futures Impact (%)", fontsize=10)
                ax_f.spines[["top","right"]].set_visible(False)

                ax_t.barh(scenarios, trd_vals, color=tcolors)
                ax_t.axvline(0, color="black", linewidth=0.8)
                ax_t.set_title("India Export Trade Flow Impact (%)", fontsize=10)
                ax_t.spines[["top","right"]].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ── Affected & opportunity markets (gravity + momentum ranker) ─
                st.divider()
                st.markdown("#### 🎯 Market Opportunities Under This Event")
                try:
                    with st.spinner("Ranking export markets (gravity gap + momentum)…"):
                        from market_opportunity import markets_affected_by_event
                        mk = markets_affected_by_event(impact, top_n=10)
                    mk_df = pd.DataFrame(mk["opportunity_markets"])
                    mk_df["event_flag"] = mk_df["event_flag"].map(
                        {"boosted": "🟢 boosted", "at_risk": "🔴 at risk",
                         "affected": "🟡 affected"}).fillna("")
                    mk_df = mk_df.rename(columns={
                        "rank": "#", "country": "Country",
                        "actual_usd_m": "Actual $M/yr", "predicted_usd_m": "Gravity Potential $M/yr",
                        "gravity_gap_pct": "Gap %", "momentum_pct": "6m Momentum %",
                        "rta": "FTA", "opportunity_score": "Score", "event_flag": "Event Effect",
                    })
                    st.dataframe(mk_df.set_index("#"), use_container_width=True)
                    if mk["affected_named"]:
                        st.caption(f"Event directly involves: {', '.join(mk['affected_named'])} "
                                   f"· Trade diversion for India: {mk['trade_diversion']}")
                    st.caption(
                        "Score = 0.40·z(gravity gap) + 0.30·z(6-month momentum) "
                        "+ 0.20·z(market size) + 0.10·FTA. Gravity gap = XGBoost gravity "
                        "model potential vs actual exports (under-served markets score higher)."
                    )
                except Exception as e:
                    st.warning(f"Market ranking unavailable: {e}")

                with st.expander("ℹ️ Model methodology"):
                    st.caption(impact.get("model_note", ""))

    # ────────────────────────────────────────────────────────────────────────
    # Sub-tab 3 — Scenario Comparison
    # ────────────────────────────────────────────────────────────────────────
    with ft3:
        st.markdown("Compare the simultaneous effects of multiple policy scenarios on "
                    "HRC futures and India's bilateral steel exports.")

        scenarios_input = st.text_area(
            "Enter 2–4 scenarios (one per line)",
            height=130,
            value=(
                "India safeguard duty 25% on HRC imports\n"
                "US Section 232 tariffs raised to 50%\n"
                "India-GCC FTA signed: zero duty on bilateral steel\n"
                "China capacity cuts 20% due to environmental regulation"
            ),
            key="multi_scenarios",
        )
        run_multi = st.button("⚡ Compare Scenarios", type="primary", key="run_multi")

        if run_multi and scenarios_input.strip():
            scenario_lines = [s.strip() for s in scenarios_input.strip().split("\n") if s.strip()]
            results_multi = []
            progress = st.progress(0)

            from steel_futures import analyze_news_impact
            for i, sc_text in enumerate(scenario_lines):
                with st.spinner(f"Analysing: {sc_text[:60]}…"):
                    try:
                        res = analyze_news_impact(sc_text)
                        results_multi.append({
                            "Scenario":        sc_text[:70],
                            "Event Type":      res.get("event_type", "—").replace("_", " ").title(),
                            "Magnitude":       res.get("magnitude", "—"),
                            "Futures Δ%":      res.get("futures_impact_pct", 0),
                            "Trade Flow Δ%":   res.get("trade_flow_impact_pct", 0),
                            "India Exports":   res.get("direction_india_exports", "—").title(),
                            "Summary":         res.get("summary", "—"),
                        })
                    except Exception as e:
                        results_multi.append({
                            "Scenario": sc_text[:70], "Event Type": "ERROR",
                            "Magnitude": "—", "Futures Δ%": 0, "Trade Flow Δ%": 0,
                            "India Exports": "—", "Summary": str(e),
                        })
                progress.progress((i + 1) / len(scenario_lines))

            progress.empty()

            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df_multi = pd.DataFrame(results_multi)
            st.dataframe(
                df_multi[["Scenario","Event Type","Magnitude","Futures Δ%","Trade Flow Δ%","India Exports"]],
                use_container_width=True
            )

            # Bubble chart: futures impact vs trade flow impact
            fig, ax = plt.subplots(figsize=(10, 6))
            for _, row in df_multi.iterrows():
                color = "#10b981" if row["India Exports"] == "Positive" else "#ef4444"
                ax.scatter(row["Futures Δ%"], row["Trade Flow Δ%"],
                           s=200, color=color, alpha=0.8, zorder=3)
                ax.annotate(row["Scenario"][:40], (row["Futures Δ%"], row["Trade Flow Δ%"]),
                            fontsize=8, ha="center", va="bottom",
                            xytext=(0, 8), textcoords="offset points")

            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("HRC Futures Impact (%)")
            ax.set_ylabel("India Export Trade Flow Impact (%)")
            ax.set_title("Scenario Map: Futures vs Trade Flow Impact\n🟢 Positive for India exports  🔴 Negative")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Full summaries"):
                for _, row in df_multi.iterrows():
                    st.markdown(f"**{row['Scenario'][:70]}**")
                    st.caption(row["Summary"])
                    st.write("")

    # ── ft4: Articles Scored ──────────────────────────────────────────────────
    with ft4:
        st.subheader("📋 Articles Scored by AI-GPR Pipeline")
        st.caption("Articles fetched by the RSS pipeline, risk-scored by Layer 1 (0–1 scale). Refresh by running the pipeline.")

        import json as _json
        import os as _os
        import pandas as _pd

        log_path = _os.path.join(_os.path.dirname(__file__), "pipeline_log.json")
        gpr_path = _os.path.join(_os.path.dirname(__file__), "futures_cache", "steel_gpr_index.json")

        # ── Load articles ────────────────────────────────────────────────────
        articles_df = None
        if _os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = _json.load(f)
                raw_articles = log.get("articles", [])
                if raw_articles:
                    articles_df = _pd.DataFrame(raw_articles)
                    for col in ["risk_score", "steel_relevant", "title", "feed", "published", "summary", "url"]:
                        if col not in articles_df.columns:
                            articles_df[col] = None
                    articles_df["risk_score"] = _pd.to_numeric(articles_df["risk_score"], errors="coerce").fillna(0.0)
            except Exception as e:
                st.warning(f"Could not load pipeline log: {e}")

        if articles_df is None or articles_df.empty:
            st.info("No articles found. Run the classifier pipeline first: `python steel_rag/classifier_pipeline.py --once`")
        else:
            # ── Filter controls ──────────────────────────────────────────────
            col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
            with col_f1:
                min_risk = st.slider("Min risk score", 0.0, 1.0, 0.0, 0.05, key="art_min_risk")
            with col_f2:
                steel_filter = st.selectbox("Steel relevant", ["All", "Yes only", "No only"], key="art_steel_filter")
            with col_f3:
                feeds_available = ["All"] + sorted(articles_df["feed"].dropna().unique().tolist())
                feed_filter = st.selectbox("Feed", feeds_available, key="art_feed_filter")

            filtered = articles_df[articles_df["risk_score"] >= min_risk].copy()
            if steel_filter == "Yes only":
                filtered = filtered[filtered["steel_relevant"] == True]
            elif steel_filter == "No only":
                filtered = filtered[filtered["steel_relevant"] != True]
            if feed_filter != "All":
                filtered = filtered[filtered["feed"] == feed_filter]

            st.caption(f"Showing **{len(filtered)}** of **{len(articles_df)}** articles")

            # ── Risk score distribution bar ───────────────────────────────────
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            import numpy as _np

            fig_dist, ax_dist = _plt.subplots(figsize=(8, 2))
            scores = articles_df["risk_score"].values
            ax_dist.hist(scores, bins=20, range=(0, 1), color="#6366f1", alpha=0.8, edgecolor="none")
            ax_dist.axvline(0.3, color="#f59e0b", linewidth=1.5, linestyle="--", label="0.3 threshold")
            ax_dist.axvline(0.5, color="#ef4444", linewidth=1.5, linestyle="--", label="0.5 threshold")
            ax_dist.set_xlabel("Risk Score"); ax_dist.set_ylabel("Count")
            ax_dist.set_title("Risk Score Distribution")
            ax_dist.spines[["top","right"]].set_visible(False)
            ax_dist.legend(fontsize=8)
            _plt.tight_layout()
            st.pyplot(fig_dist)
            _plt.close(fig_dist)

            # ── Article table with colour-coded risk ──────────────────────────
            def _risk_badge(score):
                if score >= 0.6:
                    return "🔴"
                elif score >= 0.3:
                    return "🟡"
                else:
                    return "⚪"

            display_df = filtered[["title", "feed", "published", "risk_score", "steel_relevant", "summary"]].copy()
            display_df.insert(0, "Risk", filtered["risk_score"].apply(_risk_badge))
            display_df = display_df.rename(columns={
                "title": "Title", "feed": "Feed", "published": "Published",
                "risk_score": "Score", "steel_relevant": "Steel?", "summary": "Summary"
            })
            display_df["Score"] = display_df["Score"].round(3)
            display_df["Title"] = display_df["Title"].fillna("").str[:80]
            display_df["Summary"] = display_df["Summary"].fillna("").str[:120]

            st.dataframe(
                display_df.sort_values("Score", ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=400,
            )

            # ── Steel-GPR index chart ─────────────────────────────────────────
            if _os.path.exists(gpr_path):
                st.divider()
                st.subheader("Steel-GPR Daily Index")
                st.caption("Normalized geopolitical risk index for steel trade, built from pipeline article scores (baseline=100).")
                try:
                    with open(gpr_path) as f:
                        gpr_data = _json.load(f)
                    gpr_df = _pd.DataFrame(gpr_data)
                    gpr_df["date"] = _pd.to_datetime(gpr_df["date"])
                    gpr_df = gpr_df.sort_values("date")

                    fig_gpr, ax_gpr = _plt.subplots(figsize=(10, 3))
                    ax_gpr.fill_between(gpr_df["date"], gpr_df["steel_gpr"], alpha=0.25, color="#6366f1")
                    ax_gpr.plot(gpr_df["date"], gpr_df["steel_gpr"], color="#6366f1", linewidth=1.5)
                    ax_gpr.axhline(100, color="grey", linewidth=0.8, linestyle="--", label="Baseline 100")
                    ax_gpr.set_ylabel("Steel-GPR Index")
                    ax_gpr.set_title("Steel Trade Geopolitical Risk Index")
                    ax_gpr.spines[["top","right"]].set_visible(False)
                    ax_gpr.legend(fontsize=8)
                    _plt.tight_layout()
                    st.pyplot(fig_gpr)
                    _plt.close(fig_gpr)

                    latest = gpr_df.iloc[-1]
                    delta = latest["steel_gpr"] - 100
                    st.metric("Latest Steel-GPR", f"{latest['steel_gpr']:.1f}",
                              delta=f"{delta:+.1f} vs baseline",
                              delta_color="inverse")
                except Exception as e:
                    st.warning(f"Could not render Steel-GPR chart: {e}")
            else:
                st.info("Steel-GPR index not yet built. Run the pipeline to generate it.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Gravity Model Trade Flow Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("🌐 Gravity Model — Trade Flow Scenario Analysis")
    st.caption(
        "Predicts how India's bilateral steel exports change under GDP and tariff scenarios. "
        "Model: OLS (interpretable) + XGBoost. Honest skill on an unseen market "
        "(leave-country-out CV) R²≈0.27; in-sample R² is much higher but leaks "
        "country identity. Features: ln(GDP), ln(distance), RTA dummy, language, contiguity, FE year."
    )

    @st.cache_resource(show_spinner="Loading gravity model (~5s)…")
    def _load_gravity():
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "steel_rag"))
        from gravity_model import ensure_model_ready
        return ensure_model_ready()

    try:
        grav_mdl = _load_gravity()
        grav_countries = sorted(grav_mdl["df"]["country"].unique().tolist())

        col_g1, col_g2 = st.columns([2, 1])

        with col_g1:
            selected_country = st.selectbox(
                "Partner country", grav_countries,
                index=grav_countries.index("U ARAB EMTS") if "U ARAB EMTS" in grav_countries else 0,
                key="grav_country",
            )

        with col_g2:
            grav_model_type = st.radio("Model", ["xgb", "ols"], index=0, key="grav_model",
                                       help="XGBoost (leave-country-out R²≈0.27) or "
                                            "OLS (in-sample R²=0.43, interpretable elasticities)")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            gdp_growth = st.slider(
                "Partner GDP growth (%)", -10.0, 15.0, 5.0, 0.5, key="grav_gdp",
                help="Simulates change in partner country's GDP vs latest year",
            )
        with col_s2:
            tariff_change = st.slider(
                "Tariff change (pp)", -20.0, 20.0, 0.0, 0.5, key="grav_tariff",
                help="Change in effective tariff rate (negative = tariff cut = more exports)",
            )

        if st.button("Run scenario", key="grav_run"):
            with st.spinner("Computing gravity scenario…"):
                import sys as _sys2
                _sys2.path.insert(0, str(ROOT / "steel_rag"))
                from gravity_model import predict_trade_flow
                res = predict_trade_flow(
                    selected_country,
                    gdp_growth_pct=gdp_growth,
                    tariff_change_pct=tariff_change,
                    model_type=grav_model_type,
                )

            if res.get("status") == "no_data":
                st.warning(res.get("message", "No data for this country."))
            else:
                change_pct = res.get("change_pct", 0)
                baseline   = res.get("baseline_usd", 0)
                scenario   = res.get("scenario_usd", baseline)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Baseline exports",  f"${baseline:,.1f}M / yr")
                mc2.metric("Scenario exports",  f"${scenario:,.1f}M / yr",
                           delta=f"{change_pct:+.1f}%",
                           delta_color="normal")
                mc3.metric("Change",            f"{change_pct:+.1f}%",
                           delta_color="normal")

                st.info(res.get("explanation", ""))

                with st.expander("Model details"):
                    st.json({k: v for k, v in res.items()
                             if k not in ("explanation",)})

        # ── Multi-market Bull / Base / Bear scenario matrix ─────────────────────
        st.divider()
        st.subheader("Bull / Base / Bear — multi-market scenario matrix")
        st.caption(
            "Market-specific assumptions: each market uses its own IMF 2026 GDP "
            "outlook and an FTA-conditioned tariff path, so responses differ by "
            "market (not a flat uniform shock). Bull = GDP +1.5pp, tariff easing; "
            "Bear = GDP −3pp, protection (heaviest for protection-prone markets)."
        )
        n_markets = st.slider("Markets to show", 5, 25, 12, key="grav_matrix_n")
        if st.button("Run scenario matrix", key="grav_matrix_run"):
            with st.spinner("Computing market-specific scenarios…"):
                from gravity_model import run_scenario_matrix, GDP_OUTLOOK_SOURCE
                mat = run_scenario_matrix(top_n=n_markets, model_type="ols")
            if mat.empty:
                st.warning("No scenario data available.")
            else:
                disp = mat.rename(columns={
                    "country": "Market", "gdp_outlook_pct": "GDP '26 %",
                    "fta": "FTA", "baseline_usd_m": "Baseline $M",
                    "bull_usd_m": "Bull $M", "bull_change_pct": "Bull Δ%",
                    "bear_usd_m": "Bear $M", "bear_change_pct": "Bear Δ%"})
                st.dataframe(disp.set_index("Market"), use_container_width=True)
                st.caption(f"GDP outlook source: {GDP_OUTLOOK_SOURCE}")

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np
                m = mat.iloc[::-1]
                y = np.arange(len(m))
                fig, ax = plt.subplots(figsize=(9, max(3, len(m) * 0.4)))
                ax.barh(y + 0.2, m["bull_change_pct"], height=0.38,
                        color="#2A9D8F", label="Bull")
                ax.barh(y - 0.2, m["bear_change_pct"], height=0.38,
                        color="#E76F51", label="Bear")
                ax.set_yticks(y); ax.set_yticklabels(m["country"], fontsize=8)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Change in export value vs baseline (%)")
                ax.set_title("Scenario sensitivity by market", fontsize=11)
                ax.legend(fontsize=8, loc="lower right")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ── Baseline top-10 export partners ─────────────────────────────────────
        st.divider()
        st.subheader("Baseline export volumes — top countries")
        df_grav = grav_mdl["df"]
        latest_fy = df_grav["fy_start"].max()
        top_partners = (
            df_grav[df_grav["fy_start"] == latest_fy]
            .nlargest(12, "exports_usd_mn")
            [["country", "exports_usd_mn", "ln_gdp_partner", "ln_distance"]]
            .rename(columns={
                "country": "Country",
                "exports_usd_mn": "Exports (USD Mn)",
                "ln_gdp_partner": "ln(GDP partner)",
                "ln_distance": "ln(Distance km)",
            })
            .reset_index(drop=True)
        )
        top_partners["Exports (USD Mn)"] = top_partners["Exports (USD Mn)"].round(1)
        top_partners["ln(GDP partner)"]  = top_partners["ln(GDP partner)"].round(2)
        top_partners["ln(Distance km)"]  = top_partners["ln(Distance km)"].round(2)
        st.dataframe(top_partners, use_container_width=True)

        # ── Model performance summary ────────────────────────────────────────────
        st.divider()
        st.markdown("**Model performance — honestly evaluated**")
        _m = grav_mdl.get("metrics", {})
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        perf_col1.metric("OLS R² (in-sample)", f"{_m.get('ols_r2', 0.431):.3f}")
        perf_col2.metric("XGB R² (in-sample)", f"{_m.get('xgb_r2_insample', 0.0):.3f}")
        perf_col3.metric("XGB R² (held-out)",  f"{_m.get('xgb_r2_holdout', 0.0):.3f}")
        perf_col4.metric("XGB R² (leave-country-out)", f"{_m.get('xgb_r2_loco', 0.0):.3f}")
        st.caption(
            "The in-sample R² overstates skill: distance, contiguity, language and FTA are "
            "time-invariant per country, so a random split lets XGBoost memorise known "
            "markets. Leave-country-out CV (≈0.27) is the honest skill at predicting an "
            "**unseen** market — modest. OLS is retained for interpretable elasticities "
            "(ln(GDP)≈0.85, ln(distance)≈−0.64). Treat gravity-gap rankings as indicative, "
            "not precise."
        )

    except Exception as _ge:
        st.error(f"Gravity model unavailable: {_ge}")
