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
        last_q = (st.session_state.chat_history[-1]["content"]
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
