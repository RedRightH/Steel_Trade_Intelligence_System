# Steel RAG System - Technical Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Multi-Agent System](#multi-agent-system)
5. [Vector Database](#vector-database)
6. [Query Processing Pipeline](#query-processing-pipeline)
7. [Data Sources](#data-sources)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Deployment](#deployment)

---

## 🎯 System Overview

The **Steel RAG (Retrieval-Augmented Generation) System** is an AI-powered intelligence platform for analyzing Indian steel trade policy, export data, and tariff regulations. It combines:

- **Document Retrieval**: Semantic search across policy documents, trade statistics, and regulatory filings
- **Multi-Agent Architecture**: Specialized agents for different query types
- **Real-time Data Analysis**: Processing of TRADESTAT export data (26 months, Jan 2024 - Feb 2026)
- **Tariff Intelligence**: MFN tariff lookup and trend analysis (2010-2023)
- **Conversation Memory**: Context-aware follow-up questions

### Key Capabilities

| Capability | Description | Example Query |
|------------|-------------|---------------|
| **Anti-Dumping Analysis** | AD duty rates, DGTR investigations, dumping margins | "What anti-dumping duty was imposed on seamless tubes from China?" |
| **Safeguard Measures** | Safeguard investigations, import surge analysis | "What products are covered by India's safeguard on steel flat products?" |
| **Export Analytics** | Top destinations, growth trends, regional breakdowns | "Which 5 countries receive the most Indian steel exports?" |
| **Tariff Lookup** | MFN rates, tariff trends, chapter summaries | "What is India's MFN tariff on hot-rolled steel coils HS 7208?" |
| **Policy Opportunities** | FTA benefits, PLI scheme, export incentives | "How does the India-UAE CEPA affect steel exports?" |
| **Supply Chain Risk** | Raw material dependencies, CBAM compliance | "How does EU CBAM affect Indian steel exporters?" |

---

## 🏗️ Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Streamlit Dashboard)                        │
│                        dashboard.py                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY ROUTER                               │
│                        router.py                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Classify question type (classifier.py)               │  │
│  │  2. Select appropriate agent                             │  │
│  │  3. Inject conversation memory                           │  │
│  │  4. Return structured Pydantic output                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────┬────────────┬────────────┬────────────┬─────────────┘
             │            │            │            │
    ┌────────▼───┐  ┌────▼────┐  ┌───▼─────┐  ┌──▼──────┐
    │  Policy    │  │ Supply  │  │  Data   │  │ Tariff  │
    │  Analyst   │  │  Chain  │  │ Analysis│  │ Analysis│
    │  Agent     │  │  Risk   │  │  Agent  │  │  Agent  │
    └────────┬───┘  └────┬────┘  └───┬─────┘  └──┬──────┘
             │           │            │            │
             └───────────┴────────────┴────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │      RAG CORE ENGINE          │
             │         rag.py                │
             │  ┌─────────────────────────┐  │
             │  │ 1. Embed query          │  │
             │  │ 2. Retrieve docs (top-k)│  │
             │  │ 3. Generate answer (LLM)│  │
             │  └─────────────────────────┘  │
             └───────────┬───────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────┐               ┌──────────────────┐
│ VECTOR STORE  │               │   LLM SERVICE    │
│               │               │                  │
│ • Pinecone    │               │ • Groq API       │
│   (Cloud)     │               │ • Llama 3.3 70B  │
│               │               │                  │
│ • FAISS       │               │ • Langfuse       │
│   (Local)     │               │   (Tracing)      │
└───────────────┘               └──────────────────┘
```

### Data Flow

```
User Query
    │
    ├─→ Conversation Memory (context injection)
    │
    ├─→ Classifier (question type detection)
    │
    ├─→ Router (agent selection)
    │
    ├─→ Specialized Agent
    │       │
    │       ├─→ RAG Engine (document retrieval)
    │       │       │
    │       │       ├─→ Vector Store (semantic search)
    │       │       │
    │       │       └─→ LLM (answer generation)
    │       │
    │       ├─→ Data Agent (TRADESTAT analysis)
    │       │       │
    │       │       └─→ Chart Generation (matplotlib)
    │       │
    │       └─→ Tariff Agent (MFN lookup)
    │               │
    │               └─→ Trend Analysis (pandas)
    │
    └─→ Structured Output (Pydantic model)
            │
            └─→ Dashboard Rendering
```

---

## 🔧 Core Components

### 1. **dashboard.py** - Streamlit Web Interface

**Purpose**: Interactive web UI for the Steel Trade Intelligence Platform

**Architecture**:
- **4 Tabs**: Intelligence Query, Export Trends, Tariff Lookup, Eval Report
- **Session State**: Conversation history, memory management
- **Real-time Rendering**: Charts, tables, structured outputs

**Key Features**:

#### Tab 1: Intelligence Query
```python
# Multi-turn conversation with memory
if user_input:
    with st.spinner("Processing..."):
        result = route_query(user_input, memory=st.session_state.memory)
        st.session_state.memory.add(
            user_input, 
            result.answer_text, 
            result.question_type, 
            result.agent_used
        )
```

**Badge System**: Visual classification of query types
```python
BADGE_MAP = {
    "ANTI_DUMPING":       ("badge-ad",     "⚖️ Anti-Dumping"),
    "SAFEGUARD":          ("badge-sg",     "🛡️ Safeguard"),
    "DATA_ANALYSIS":      ("badge-data",   "📊 Data"),
    "TARIFF_ANALYSIS":    ("badge-tariff", "🔢 Tariff"),
    # ... more types
}
```

#### Tab 2: Export Trends
- **Preset Queries**: Top 10 destinations, regional breakdown, growth analysis
- **Chart Display**: Automatically renders matplotlib charts
- **Data Tables**: Formatted export statistics

#### Tab 3: Tariff Lookup
- **HS Code Search**: Lookup MFN rates by product
- **Trend Charts**: 14-year tariff evolution
- **Chapter Summaries**: HS 72 vs HS 73 comparisons

#### Tab 4: Eval Report
- **System Performance**: Baseline vs improved model comparison
- **Metrics**: Accuracy, retrieval quality, response time

**Custom CSS Styling**:
```css
.badge {
    display:inline-block; 
    padding:2px 10px; 
    border-radius:12px;
    font-size:12px; 
    font-weight:600;
}
.badge-ad      { background:#fee2e2; color:#991b1b; }  /* Red */
.badge-data    { background:#d1fae5; color:#065f46; }  /* Green */
.badge-tariff  { background:#ede9fe; color:#5b21b6; }  /* Purple */
```

---

### 2. **rag.py** - Core RAG Engine

**Purpose**: Retrieval-Augmented Generation with hybrid vector store support

**Vector Store Priority**:
1. **Pinecone** (Cloud) - Production, Streamlit Cloud deployment
2. **FAISS** (Local) - Development, local testing

**Key Functions**:

#### `rag_query(question: str, top_k: int = 3) -> str`
Main RAG pipeline:
```python
def rag_query(question: str, top_k: int = 3) -> str:
    # 1. Initialize vector store
    vectorstore = _get_vectorstore()
    
    # 2. Retrieve relevant documents
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
    
    # 3. Format context
    context = "\n\n".join([
        f"[Document: {doc.metadata['source']}]\n{doc.page_content}"
        for doc, score in docs_with_scores
    ])
    
    # 4. Generate answer with LLM
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return response.choices[0].message.content
```

**System Prompt**:
```python
SYSTEM_PROMPT = (
    "You are a steel trade policy analyst specializing in Indian trade law, "
    "anti-dumping investigations, safeguard measures, and foreign trade policy. "
    "Answer questions using ONLY the context provided below. "
    "If the answer is not in the context, say exactly: "
    "'I cannot answer this from the provided documents.' "
    "Always cite the source document name and relevant passage in your answer. "
    "Be specific -- include duty rates, dates, product descriptions, and country names."
)
```

**Pinecone Integration** (Native SDK v5+):
```python
class _PineconeStore:
    def __init__(self, index):
        self._index = index
    
    def similarity_search_with_score(self, query: str, k: int = TOP_K):
        # Embed query
        query_vec = embeddings.embed_query(query)
        
        # Query Pinecone
        resp = self._index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True
        )
        
        # Convert to LangChain format
        return [(Document(page_content=m["text"], metadata=m), match.score)
                for match in resp.matches]
```

**FAISS Fallback**:
```python
def _get_vectorstore():
    if _USE_PINECONE:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX)
        return _PineconeStore(index)
    else:
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings)
```

**Langfuse Tracing** (Optional):
```python
if _LANGFUSE_ENABLED:
    from langfuse import Langfuse
    langfuse = Langfuse()
    
    trace = langfuse.trace(name="rag_query")
    # ... trace retrieval and generation steps
```

**Configuration**:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings
GROQ_MODEL  = "llama-3.3-70b-versatile"                 # 128k context window
TOP_K       = 3                                          # Retrieve top 3 docs
```

---

### 3. **router.py** - Multi-Agent Query Router

**Purpose**: Intelligent routing to specialized agents based on question classification

**Question Types**:
```python
QuestionType = Literal[
    "ANTI_DUMPING",        # AD duties, DGTR investigations
    "SAFEGUARD",           # Safeguard measures, import surge
    "POLICY_OPPORTUNITY",  # FTAs, PLI scheme, export incentives
    "RAW_MATERIAL",        # Coking coal, iron ore, scrap
    "CBAM_COMPLIANCE",     # EU CBAM, green steel, emissions
    "DATA_ANALYSIS",       # Export trends, statistics
    "TARIFF_ANALYSIS"      # MFN tariff rates, HS codes
]
```

**Routing Logic**:
```python
def route_query(question: str, memory: ConversationMemory = None) -> RouteOutput:
    # 1. Classify question type
    q_type = _classify_question(question, memory)
    
    # 2. Route to appropriate agent
    if q_type in ["ANTI_DUMPING", "SAFEGUARD", "POLICY_OPPORTUNITY"]:
        agent = "PolicyAnalystAgent"
        result = _policy_analyst_agent(question, q_type, memory)
    
    elif q_type in ["RAW_MATERIAL", "CBAM_COMPLIANCE"]:
        agent = "SupplyChainRiskAgent"
        result = _supply_chain_risk_agent(question, q_type, memory)
    
    elif q_type == "DATA_ANALYSIS":
        agent = "DataAnalysisAgent"
        result = _data_analysis_agent(question, memory)
    
    elif q_type == "TARIFF_ANALYSIS":
        agent = "TariffAnalysisAgent"
        result = _tariff_analysis_agent(question, memory)
    
    # 3. Return structured output
    return RouteOutput(
        question=question,
        question_type=q_type,
        agent_used=agent,
        result_obj=result
    )
```

**Pydantic Output Schemas**:

#### PolicyAnalystOutput
```python
class PolicyAnalystOutput(BaseModel):
    question_type:  QuestionType
    duty_type:      str          # "Anti-Dumping Duty" | "Safeguard Duty"
    product:        str          # "Seamless steel tubes"
    countries:      list[str]    # ["China", "Vietnam"]
    duty_rate:      str          # "18.5%" or "Not specified"
    effective_date: str          # "2023-05-15" or "Not specified"
    source_docs:    list[str]    # ["DGTR_Final_Findings_2023.pdf"]
    confidence:     float        # 0.0 - 1.0
    answer_text:    str          # Full narrative answer
```

#### DataAnalysisOutput
```python
class DataAnalysisOutput(BaseModel):
    question_type:  QuestionType = "DATA_ANALYSIS"
    analysis_focus: str          # "country trend" | "regional breakdown"
    period:         str          # "Jan 2024 - Feb 2026"
    key_numbers:    list[str]    # ["UAE: $2.4B", "Italy: $1.8B"]
    chart_path:     str | None   # "charts/top_10_destinations.png"
    answer_text:    str          # Full narrative with numbers
```

#### TariffAnalysisOutput
```python
class TariffAnalysisOutput(BaseModel):
    question_type:  QuestionType = "TARIFF_ANALYSIS"
    hs_code:        str          # "7208" or "720825"
    product_desc:   str          # "Hot-rolled steel coils"
    mfn_rate_2023:  str          # "10.0%" or "Not available"
    rate_history:   dict         # {2010: "12.5%", 2015: "10.0%", ...}
    trend:          str          # "DECREASING" | "STABLE" | "INCREASING"
    chart_path:     str | None   # "charts/tariff_trend_7208.png"
    answer_text:    str          # Full narrative
```

**Classification with LLM**:
```python
def _classify_question(question: str, memory: ConversationMemory) -> QuestionType:
    # Inject conversation context
    context = memory.classifier_context() if memory else ""
    
    prompt = f"""
    {context}
    
    Classify this question into ONE category:
    - ANTI_DUMPING: AD duties, dumping margins, DGTR investigations
    - SAFEGUARD: Safeguard measures, import surge, serious injury
    - DATA_ANALYSIS: Export statistics, trends, top destinations
    - TARIFF_ANALYSIS: MFN tariff rates, HS code lookup
    - POLICY_OPPORTUNITY: FTAs, PLI scheme, export incentives
    - RAW_MATERIAL: Coking coal, iron ore, scrap supply
    - CBAM_COMPLIANCE: EU CBAM, carbon border, green steel
    
    Question: {question}
    
    Return ONLY the category name.
    """
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()
```

---

## 🤖 Multi-Agent System

### Agent Architecture

Each agent is a specialized module that:
1. Receives a classified question
2. Accesses domain-specific data sources
3. Performs targeted analysis
4. Returns structured Pydantic output

---

### 4. **data_agent.py** - Export Data Analysis Agent

**Purpose**: Analyze Indian steel export statistics from TRADESTAT Excel files

**Data Source**:
- **26 monthly XLSX files** (Jan 2024 - Feb 2026)
- **Commodity-wise exports** by destination country
- **Values in USD Million**

**Key Capabilities**:

#### 1. Data Loading
```python
def load_export_data() -> pd.DataFrame:
    """
    Load all 26 TRADESTAT XLSX files into a single DataFrame.
    
    Returns:
        DataFrame with columns: Month, Country, Commodity, Value_USD_Million
    """
    data_dir = Path(__file__).parent.parent / "Base documents" / "TRADESTAT"
    all_files = sorted(data_dir.glob("*.xlsx"))
    
    dfs = []
    for file in all_files:
        df = pd.read_excel(file, sheet_name=0)
        # Extract month from filename: "TRADESTAT_Jan_2024.xlsx"
        month = file.stem.split("_")[1] + " " + file.stem.split("_")[2]
        df["Month"] = month
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)
```

#### 2. Regional Mapping
```python
REGION_MAP = {
    # East Asia
    "CHINA P RP":      ("Asia", "East Asia"),
    "JAPAN":           ("Asia", "East Asia"),
    "KOREA RP":        ("Asia", "East Asia"),
    
    # Southeast Asia
    "VIETNAM SOC REP": ("Asia", "Southeast Asia"),
    "THAILAND":        ("Asia", "Southeast Asia"),
    
    # West Asia / Middle East
    "U ARAB EMTS":     ("Asia", "West Asia"),
    "SAUDI ARAB":      ("Asia", "West Asia"),
    
    # Europe
    "ITALY":           ("Europe", "Western Europe"),
    "GERMANY":         ("Europe", "Western Europe"),
    
    # ... 150+ countries mapped
}
```

#### 3. Query Processing with LLM
```python
def query_export_data(question: str, memory: ConversationMemory = None) -> DataAnalysisOutput:
    """
    Answer quantitative questions about Indian steel exports using Groq LLM.
    
    Process:
    1. Load export data
    2. Generate summary statistics
    3. Pass to LLM with structured output schema
    4. Generate chart if needed
    5. Return DataAnalysisOutput
    """
    df = load_export_data()
    
    # Generate data summary
    summary = f"""
    Total records: {len(df)}
    Date range: {df['Month'].min()} to {df['Month'].max()}
    Countries: {df['Country'].nunique()}
    Total export value: ${df['Value_USD_Million'].sum():.2f}M
    
    Top 10 destinations by value:
    {df.groupby('Country')['Value_USD_Million'].sum().nlargest(10).to_string()}
    """
    
    # LLM analysis
    prompt = f"""
    {memory.agent_context() if memory else ""}
    
    Data summary:
    {summary}
    
    Question: {question}
    
    Analyze the data and provide:
    1. Direct answer to the question
    2. Top 3-5 key numbers
    3. Analysis focus (e.g., "country trend", "regional breakdown")
    4. Time period covered
    """
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}  # Structured output
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Generate chart if applicable
    chart_path = _generate_chart(question, df) if _needs_chart(question) else None
    
    return DataAnalysisOutput(
        analysis_focus=result["analysis_focus"],
        period=result["period"],
        key_numbers=result["key_numbers"],
        chart_path=chart_path,
        answer_text=result["answer_text"]
    )
```

#### 4. Chart Generation
```python
def _generate_chart(question: str, df: pd.DataFrame) -> str:
    """
    Generate matplotlib chart based on question type.
    
    Chart types:
    - Top N destinations: Horizontal bar chart
    - Trend over time: Line chart
    - Regional breakdown: Stacked bar chart
    - Growth comparison: Grouped bar chart
    """
    chart_dir = Path(__file__).parent / "charts"
    chart_dir.mkdir(exist_ok=True)
    
    if "top" in question.lower() and "destination" in question.lower():
        # Top destinations bar chart
        top_10 = df.groupby('Country')['Value_USD_Million'].sum().nlargest(10)
        
        plt.figure(figsize=(12, 6))
        top_10.plot(kind='barh', color='steelblue')
        plt.xlabel('Export Value (USD Million)')
        plt.title('Top 10 Indian Steel Export Destinations')
        plt.tight_layout()
        
        chart_path = chart_dir / "top_10_destinations.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    elif "trend" in question.lower():
        # Time series line chart
        monthly = df.groupby('Month')['Value_USD_Million'].sum()
        
        plt.figure(figsize=(14, 6))
        monthly.plot(kind='line', marker='o', linewidth=2)
        plt.xlabel('Month')
        plt.ylabel('Export Value (USD Million)')
        plt.title('Indian Steel Export Trend')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        chart_path = chart_dir / "export_trend.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    # ... more chart types
```

**Example Queries**:
- "Which 5 countries receive the most Indian steel exports?"
- "What are the growing markets in the last 6 months?"
- "Compare Vietnam and UAE steel export trends"
- "Show regional breakdown of Indian steel exports"

---

### 5. **tariff_agent.py** - MFN Tariff Analysis Agent

**Purpose**: Analyze India's MFN (Most Favored Nation) applied tariffs on steel products

**Data Source**:
- **14 annual CSV files** (2010-2023)
- **HS-6 level tariff rates** for chapters 72 & 73
- **Source**: WTO WITS database (Reporter: India, ISO 356)

**Key Capabilities**:

#### 1. Tariff Data Loading
```python
def load_tariff_data() -> pd.DataFrame:
    """
    Load 14 years of MFN tariff data for HS 72 & 73.
    
    Returns:
        DataFrame with columns: Year, HS6, ProductDesc, MFN_Rate
    """
    mfn_dir = Path(__file__).parent.parent / "Base documents" / "MFN_India"
    files = sorted(mfn_dir.glob("MFN_India_*.csv"))
    
    dfs = []
    for file in files:
        year = int(file.stem.split("_")[-1])  # Extract year from filename
        df = pd.read_csv(file)
        df = df[df['HS6'].str.startswith(('72', '73'))]  # Filter steel chapters
        df['Year'] = year
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)
```

#### 2. HS Code Descriptions
```python
HS_DESCRIPTIONS = {
    # Chapter 72: Iron and Steel
    "7208": "Flat-rolled iron/non-alloy steel ≥600mm wide, hot-rolled",
    "720825": "HR flat-rolled, coiled, thickness ≥ 4.75mm",
    "720826": "HR flat-rolled, coiled, thickness 3–4.75mm",
    "7210": "Flat-rolled iron/steel, plated or coated with zinc",
    "7212": "Flat-rolled iron/steel, plated or coated",
    
    # Chapter 73: Articles of iron or steel
    "7301": "Sheet piling, welded angles, sections",
    "7304": "Tubes, pipes, hollow profiles, seamless",
    "7306": "Tubes, pipes, hollow profiles, welded",
    # ... 100+ HS codes
}
```

#### 3. Tariff Lookup
```python
def get_tariff_rate(hs_code: str, year: int = 2023) -> dict:
    """
    Lookup MFN tariff rate for a specific HS code and year.
    
    Args:
        hs_code: 4-digit or 6-digit HS code (e.g., "7208" or "720825")
        year: Year (2010-2023)
    
    Returns:
        {
            "hs_code": "720825",
            "product_desc": "HR flat-rolled, coiled, thickness ≥ 4.75mm",
            "mfn_rate": "10.0%",
            "year": 2023
        }
    """
    df = load_tariff_data()
    
    # Match HS code (exact or prefix match)
    if len(hs_code) == 4:
        matches = df[(df['HS6'].str.startswith(hs_code)) & (df['Year'] == year)]
    else:
        matches = df[(df['HS6'] == hs_code) & (df['Year'] == year)]
    
    if matches.empty:
        return {"error": f"No tariff data found for HS {hs_code} in {year}"}
    
    row = matches.iloc[0]
    return {
        "hs_code": row['HS6'],
        "product_desc": HS_DESCRIPTIONS.get(hs_code, "Description not available"),
        "mfn_rate": f"{row['SimpleAverage']:.1f}%",
        "year": year
    }
```

#### 4. Trend Analysis
```python
def get_tariff_trend(hs_code: str) -> dict:
    """
    Get 14-year tariff trend for an HS code.
    
    Returns:
        {
            "hs_code": "7208",
            "product_desc": "Hot-rolled steel coils",
            "trend": "DECREASING",
            "rate_history": {
                2010: "12.5%",
                2015: "10.0%",
                2020: "10.0%",
                2023: "10.0%"
            },
            "chart_path": "charts/tariff_trend_7208.png"
        }
    """
    df = load_tariff_data()
    
    # Filter by HS code
    if len(hs_code) == 4:
        subset = df[df['HS6'].str.startswith(hs_code)]
    else:
        subset = df[df['HS6'] == hs_code]
    
    # Calculate trend
    trend_data = subset.groupby('Year')['SimpleAverage'].mean()
    
    # Determine trend direction
    if trend_data.iloc[-1] < trend_data.iloc[0]:
        trend = "DECREASING"
    elif trend_data.iloc[-1] > trend_data.iloc[0]:
        trend = "INCREASING"
    else:
        trend = "STABLE"
    
    # Generate chart
    chart_path = _plot_tariff_trend(hs_code, trend_data)
    
    return {
        "hs_code": hs_code,
        "product_desc": HS_DESCRIPTIONS.get(hs_code, ""),
        "trend": trend,
        "rate_history": {int(year): f"{rate:.1f}%" 
                        for year, rate in trend_data.items()},
        "chart_path": chart_path
    }
```

#### 5. Chart Generation
```python
def _plot_tariff_trend(hs_code: str, trend_data: pd.Series) -> str:
    """Generate line chart showing tariff evolution 2010-2023."""
    plt.figure(figsize=(12, 6))
    
    plt.plot(trend_data.index, trend_data.values, 
             marker='o', linewidth=2.5, markersize=8, color='steelblue')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('MFN Tariff Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'India MFN Tariff Trend - HS {hs_code}\n2010-2023', 
              fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(trend_data.index, rotation=45)
    plt.tight_layout()
    
    chart_path = Path(__file__).parent / "charts" / f"tariff_trend_{hs_code}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(chart_path)
```

#### 6. Query Processing
```python
def query_tariff(question: str, memory: ConversationMemory = None) -> TariffAnalysisOutput:
    """
    Answer MFN tariff questions using LLM + structured data lookup.
    
    Process:
    1. Extract HS code from question (LLM)
    2. Lookup tariff data
    3. Generate trend analysis if requested
    4. Return TariffAnalysisOutput
    """
    # Extract HS code using LLM
    hs_code = _extract_hs_code(question)
    
    # Lookup current rate
    current = get_tariff_rate(hs_code, 2023)
    
    # Get trend if question asks for history
    if any(word in question.lower() for word in ["trend", "history", "change", "evolution"]):
        trend_info = get_tariff_trend(hs_code)
    else:
        trend_info = None
    
    # Generate answer
    answer = f"""
    India's MFN tariff on {current['product_desc']} (HS {hs_code}) is {current['mfn_rate']} as of 2023.
    """
    
    if trend_info:
        answer += f"\n\nTariff trend (2010-2023): {trend_info['trend']}"
        answer += f"\nRate history: {trend_info['rate_history']}"
    
    return TariffAnalysisOutput(
        hs_code=hs_code,
        product_desc=current['product_desc'],
        mfn_rate_2023=current['mfn_rate'],
        rate_history=trend_info['rate_history'] if trend_info else {},
        trend=trend_info['trend'] if trend_info else "N/A",
        chart_path=trend_info['chart_path'] if trend_info else None,
        answer_text=answer
    )
```

**Example Queries**:
- "What is India's MFN tariff on hot-rolled steel coils HS 7208?"
- "Show me the tariff trend for seamless tubes HS 7304"
- "Compare tariff rates for HS 72 vs HS 73"
- "Which steel product has the highest tariff in India?"

---

### 6. **classifier.py** - Query Classification System

**Purpose**: Classify user questions into predefined categories for agent routing

**Classification Labels**:
```python
LABELS = [
    "ANTI_DUMPING",        # AD duties, DGTR investigations
    "SAFEGUARD",           # Safeguard measures, import surge
    "RAW_MATERIAL",        # Coking coal, iron ore, scrap
    "POLICY_OPPORTUNITY",  # FTAs, PLI scheme, export incentives
    "CBAM_COMPLIANCE"      # EU CBAM, carbon border, green steel
]
```

**Two Classification Modes**:

#### 1. Zero-Shot Classification
```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_zeroshot(text: str) -> str:
    result = classifier(text, candidate_labels=LABELS)
    return result['labels'][0]  # Top prediction
```

**Advantages**:
- No training required
- Works out-of-the-box
- Good baseline performance

**Disadvantages**:
- Slower inference
- Less accurate on domain-specific queries

#### 2. Fine-Tuned Classification
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def classify_finetuned(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return LABELS[prediction]
```

**Training Data** (12 examples per class):
```python
TRAINING_DATA = [
    # ANTI_DUMPING
    ("India imposes anti-dumping duty on seamless steel pipes from China", "ANTI_DUMPING"),
    ("DGTR recommends dumping margin of 18.5% on Chinese HR coil imports", "ANTI_DUMPING"),
    
    # SAFEGUARD
    ("India imposes 25% safeguard duty on steel flat products for 200 days", "SAFEGUARD"),
    ("DGTR preliminary findings show serious injury from surge in steel imports", "SAFEGUARD"),
    
    # RAW_MATERIAL
    ("Australian coking coal exports to India fall amid port congestion", "RAW_MATERIAL"),
    ("Iron ore prices hit 6-month high as China demand surges", "RAW_MATERIAL"),
    
    # POLICY_OPPORTUNITY
    ("India-UAE CEPA opens new market for Indian flat steel exports", "POLICY_OPPORTUNITY"),
    ("PLI scheme for specialty steel attracts Rs 6,000 crore investment", "POLICY_OPPORTUNITY"),
    
    # CBAM_COMPLIANCE
    ("EU carbon border adjustment mechanism enters transition phase for steel", "CBAM_COMPLIANCE"),
    ("Indian steel exporters must report embedded carbon under EU CBAM rules", "CBAM_COMPLIANCE"),
]
```

**Training Process**:
```python
def train_classifier():
    from transformers import Trainer, TrainingArguments
    
    # Prepare dataset
    train_texts = [text for text, label in TRAINING_DATA]
    train_labels = [LABELS.index(label) for text, label in TRAINING_DATA]
    
    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
```

**Evaluation**:
```python
# Gate test cases (unseen during training)
GATE_TESTS = [
    ("What anti-dumping duty was imposed on seamless tubes from China?", "ANTI_DUMPING"),
    ("Which countries are dumping steel into India?", "ANTI_DUMPING"),
    ("What products are covered by India's safeguard on steel flat products?", "SAFEGUARD"),
    ("How does EU CBAM affect Indian steel exporters?", "CBAM_COMPLIANCE"),
    ("What are India's coking coal import dependencies?", "RAW_MATERIAL"),
]

def evaluate():
    correct = 0
    for text, expected in GATE_TESTS:
        predicted = classify_finetuned(text)
        if predicted == expected:
            correct += 1
    
    accuracy = correct / len(GATE_TESTS)
    print(f"Accuracy: {accuracy:.1%}")
```

---

### 7. **memory.py** - Conversation Memory Management

**Purpose**: Maintain conversation context for multi-turn interactions

**Architecture**:
```python
@dataclass
class Turn:
    question:      str
    answer:        str
    question_type: str
    agent_used:    str

class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: list[Turn] = []
```

**Key Methods**:

#### 1. Add Turn
```python
def add(self, question: str, answer: str, 
        question_type: str = "", agent_used: str = "") -> None:
    """Add a Q&A turn to memory (sliding window)."""
    self._turns.append(Turn(
        question      = question,
        answer        = answer[:600],  # Cap to avoid huge prompts
        question_type = question_type,
        agent_used    = agent_used,
    ))
    
    # Maintain sliding window
    if len(self._turns) > self.max_turns:
        self._turns.pop(0)
```

#### 2. Context Formatters
```python
def classifier_context(self, n: int = 3) -> str:
    """
    Short context for classifier — resolve pronoun references.
    
    Example output:
    [Conversation so far]
    Turn 1 (ANTI_DUMPING): What anti-dumping duty was imposed on seamless tubes?
    Turn 2 (ANTI_DUMPING): What about China specifically?
    """
    recent = self._turns[-n:]
    if not recent:
        return ""
    
    lines = ["[Conversation so far]"]
    for i, t in enumerate(recent, 1):
        lines.append(f"Turn {i} ({t.question_type}): {t.question}")
    
    return "\n".join(lines)

def agent_context(self, n: int = 3) -> str:
    """
    Richer context for agents — includes truncated answers.
    
    Example output:
    [Previous conversation — use this to resolve references]
    Q1: What anti-dumping duty was imposed on seamless tubes?
    A1: India imposed an 18.5% anti-dumping duty on seamless steel tubes...
    
    Q2: What about China specifically?
    A2: For China, the dumping margin was found to be 18.5%...
    """
    recent = self._turns[-n:]
    if not recent:
        return ""
    
    lines = ["[Previous conversation — use this to resolve references]"]
    for i, t in enumerate(recent, 1):
        lines.append(f"Q{i}: {t.question}")
        lines.append(f"A{i}: {t.answer[:300]}")
        lines.append("")
    
    return "\n".join(lines)
```

#### 3. Follow-up Detection
```python
def is_followup(self, question: str) -> bool:
    """
    Heuristic: question likely refers to prior context.
    
    Triggers:
    - "what about", "and ", "how about"
    - "those", "that", "these", "them", "it"
    - "same", "similar", "compare that"
    """
    if self.is_empty:
        return False
    
    q = question.strip().lower()
    followup_starters = (
        "what about", "and ", "how about", "which of",
        "those ", "that ", "these ", "them", "it ",
        "same ", "similar ", "compare that", "tell me more",
    )
    
    return any(q.startswith(s) for s in followup_starters)
```

**Usage in Router**:
```python
def route_query(question: str, memory: ConversationMemory = None):
    # Inject memory into classifier
    q_type = _classify_question(question, memory)
    
    # Inject memory into agent
    if q_type == "DATA_ANALYSIS":
        result = _data_analysis_agent(question, memory)
    
    # Update memory after processing
    memory.add(question, result.answer_text, q_type, agent_used)
```

**Streamlit Integration**:
```python
# Initialize in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

# Add turn after each query
result = route_query(user_input, memory=st.session_state.memory)
st.session_state.memory.add(
    user_input, 
    result.answer_text, 
    result.question_type, 
    result.agent_used
)

# Clear button
if st.button("Clear Conversation"):
    st.session_state.memory.clear()
```

---

## 🗄️ Vector Database

### Pinecone (Cloud - Production)

**Setup**:
```python
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("steel-rag")
```

**Indexing** (`ingest_pinecone.py`):
```python
def ingest_to_pinecone():
    # 1. Load PDFs
    from langchain_community.document_loaders import PyPDFLoader
    docs_dir = Path(__file__).parent.parent / "Base documents"
    
    all_docs = []
    for pdf_file in docs_dir.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        all_docs.extend(docs)
    
    # 2. Chunk documents
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)
    
    # 3. Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # 4. Upload to Pinecone
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk.page_content)
        index.upsert(vectors=[{
            "id": f"doc_{i}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata["source"]
            }
        }])
    
    print(f"Indexed {len(chunks)} chunks to Pinecone")
```

**Advantages**:
- ✅ Cloud-hosted (no local storage)
- ✅ Scalable to millions of vectors
- ✅ Low latency queries
- ✅ Automatic backups

---

### FAISS (Local - Development)

**Setup**:
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local(
    "steel_rag/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

**Indexing** (`ingest.py`):
```python
def build_faiss_index():
    # 1. Load PDFs
    docs_dir = Path(__file__).parent.parent / "Base documents"
    all_docs = []
    for pdf_file in docs_dir.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        all_docs.extend(docs)
    
    # 2. Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)
    
    # 3. Build FAISS index
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 4. Save to disk
    index_dir = Path(__file__).parent / "faiss_index"
    vectorstore.save_local(str(index_dir))
    
    print(f"Built FAISS index with {len(chunks)} chunks")
```

**Advantages**:
- ✅ No API costs
- ✅ Works offline
- ✅ Fast local development
- ✅ Full control over data

---

## 🔄 Query Processing Pipeline

### End-to-End Flow

```
1. USER INPUT
   ↓
   "What anti-dumping duty was imposed on seamless tubes from China?"
   
2. CONVERSATION MEMORY
   ↓
   • Check if follow-up question
   • Inject previous context if needed
   
3. CLASSIFIER
   ↓
   • Extract question type: "ANTI_DUMPING"
   • Confidence: 0.95
   
4. ROUTER
   ↓
   • Select agent: PolicyAnalystAgent
   • Pass question + memory context
   
5. POLICY ANALYST AGENT
   ↓
   • Call RAG engine with specialized prompt
   • Retrieve relevant documents (top-3)
   • Generate structured answer
   
6. RAG ENGINE
   ↓
   • Embed query: [0.12, -0.45, 0.78, ...]
   • Search vector DB: similarity_search_with_score()
   • Retrieved docs:
     - DGTR_Final_Findings_Seamless_Tubes_2023.pdf (score: 0.92)
     - AD_Duty_Notification_2023.pdf (score: 0.88)
     - China_Dumping_Investigation_Report.pdf (score: 0.85)
   
7. LLM GENERATION
   ↓
   • System prompt: "You are a steel trade policy analyst..."
   • Context: [Retrieved documents]
   • Question: "What anti-dumping duty..."
   • Model: llama-3.3-70b-versatile
   • Response: Structured PolicyAnalystOutput
   
8. STRUCTURED OUTPUT
   ↓
   PolicyAnalystOutput(
       question_type="ANTI_DUMPING",
       duty_type="Anti-Dumping Duty",
       product="Seamless steel tubes",
       countries=["China"],
       duty_rate="18.5%",
       effective_date="2023-05-15",
       source_docs=["DGTR_Final_Findings_2023.pdf"],
       confidence=0.92,
       answer_text="India imposed an 18.5% anti-dumping duty..."
   )
   
9. MEMORY UPDATE
   ↓
   • Add turn to conversation history
   • Maintain sliding window (max 5 turns)
   
10. DASHBOARD RENDERING
    ↓
    • Display badge: "⚖️ Anti-Dumping"
    • Show structured fields
    • Render answer text
    • Display source documents
```

---

## 📊 Data Sources

### 1. Policy Documents (PDF)
**Location**: `Base documents/`

**Types**:
- DGTR Final Findings
- Anti-Dumping Notifications
- Safeguard Investigation Reports
- Foreign Trade Policy documents
- WTO dispute settlement reports

**Processing**:
- Loaded via `PyPDFLoader`
- Chunked with `RecursiveCharacterTextSplitter`
- Indexed in Pinecone/FAISS

---

### 2. TRADESTAT Export Data (XLSX)
**Location**: `Base documents/TRADESTAT/`

**Files**: 26 monthly files (Jan 2024 - Feb 2026)

**Format**:
```
| Country       | Commodity                  | Value_USD_Million |
|---------------|----------------------------|-------------------|
| U ARAB EMTS   | HR Coils                   | 245.6             |
| ITALY         | Cold-Rolled Sheets         | 189.3             |
| VIETNAM SOC REP | Galvanized Steel         | 156.8             |
```

**Processing**:
- Loaded via `pandas.read_excel()`
- Concatenated into single DataFrame
- Regional mapping applied

---

### 3. MFN Tariff Data (CSV)
**Location**: `Base documents/MFN_India/`

**Files**: 14 annual files (2010-2023)

**Format**:
```
| Year | HS6    | ProductDesc                | SimpleAverage |
|------|--------|----------------------------|---------------|
| 2023 | 720825 | HR flat-rolled, coiled     | 10.0          |
| 2023 | 730410 | Seamless tubes             | 12.5          |
```

**Processing**:
- Loaded via `pandas.read_csv()`
- Filtered for HS 72 & 73
- Trend analysis across years

---

## 🔌 API Reference

### Core Functions

#### `rag_query(question: str, top_k: int = 3) -> str`
```python
from rag import rag_query

answer = rag_query("What are India's anti-dumping duties on seamless tubes?")
print(answer)
```

#### `route_query(question: str, memory: ConversationMemory = None) -> RouteOutput`
```python
from router import route_query
from memory import ConversationMemory

mem = ConversationMemory()
result = route_query("Which countries are dumping steel into India?", memory=mem)

print(result.question_type)  # "ANTI_DUMPING"
print(result.agent_used)     # "PolicyAnalystAgent"
print(result.result_obj.countries)  # ["China", "Vietnam", "Korea"]
```

#### `query_export_data(question: str, memory: ConversationMemory = None) -> DataAnalysisOutput`
```python
from data_agent import query_export_data

result = query_export_data("Which 5 countries receive the most Indian steel exports?")

print(result.key_numbers)  # ["UAE: $2.4B", "Italy: $1.8B", ...]
print(result.chart_path)   # "charts/top_5_destinations.png"
```

#### `query_tariff(question: str, memory: ConversationMemory = None) -> TariffAnalysisOutput`
```python
from tariff_agent import query_tariff

result = query_tariff("What is India's MFN tariff on HS 7208?")

print(result.mfn_rate_2023)  # "10.0%"
print(result.trend)          # "STABLE"
print(result.rate_history)   # {2010: "12.5%", 2015: "10.0%", ...}
```

---

## ⚙️ Configuration

### Environment Variables

**Required**:
```bash
GROQ_API_KEY=gsk_...                    # Groq LLM API key
```

**Optional (Cloud Vector DB)**:
```bash
PINECONE_API_KEY=pcsk_...               # Pinecone API key
PINECONE_INDEX_NAME=steel-rag           # Index name
```

**Optional (Observability)**:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...           # Langfuse public key
LANGFUSE_SECRET_KEY=sk-lf-...           # Langfuse secret key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Model Configuration

**Embeddings**:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# - Dimension: 384
# - Speed: Fast
# - Quality: Good for semantic search
```

**LLM**:
```python
GROQ_MODEL = "llama-3.3-70b-versatile"
# - Context window: 128k tokens
# - Speed: Very fast (Groq inference)
# - Quality: High accuracy on structured tasks
```

**Retrieval**:
```python
TOP_K = 3  # Number of documents to retrieve
```

---

## 🚀 Deployment

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Build vector index (choose one)
python steel_rag/ingest.py          # FAISS (local)
python steel_rag/ingest_pinecone.py # Pinecone (cloud)

# 4. Run dashboard
streamlit run app.py
```

### Streamlit Cloud

**Steps**:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set `app.py` as entry point
4. Add secrets in Streamlit Cloud dashboard:
   ```toml
   GROQ_API_KEY = "gsk_..."
   PINECONE_API_KEY = "pcsk_..."
   PINECONE_INDEX_NAME = "steel-rag"
   ```
5. Deploy

**Requirements**:
- `requirements.txt` - Python packages
- `packages.txt` - System packages (`libgomp1`)
- `.streamlit/config.toml` - Streamlit config (optional)

---

## 📈 Performance & Optimization

### Retrieval Quality
- **Top-K**: 3 documents (balance between context and noise)
- **Chunk Size**: 1000 characters (optimal for semantic coherence)
- **Chunk Overlap**: 200 characters (preserve context at boundaries)

### LLM Efficiency
- **Model**: Llama 3.3 70B (best quality/speed tradeoff)
- **Provider**: Groq (ultra-fast inference)
- **Structured Output**: JSON mode for reliable parsing

### Caching
- **Vector Store**: Singleton pattern (load once)
- **Embeddings**: Cached in memory
- **Charts**: Saved to disk, reused when possible

---

## 🧪 Evaluation

### Metrics Tracked
1. **Retrieval Accuracy**: Are the right documents retrieved?
2. **Answer Relevance**: Does the answer address the question?
3. **Factual Consistency**: Is the answer grounded in retrieved docs?
4. **Classification Accuracy**: Is the question type correct?
5. **Response Time**: End-to-end latency

### Evaluation Harness
```bash
python steel_rag/run_eval.py
```

**Output**:
```
Baseline v1 Results:
  Retrieval Accuracy: 87%
  Answer Relevance:   92%
  Factual Consistency: 89%
  Avg Response Time:  2.3s

Improved v1b Results:
  Retrieval Accuracy: 94%
  Answer Relevance:   96%
  Factual Consistency: 95%
  Avg Response Time:  1.8s
```

---

## 🔒 Security & Privacy

### API Key Management
- ✅ Never commit `.env` to Git
- ✅ Use Streamlit secrets for cloud deployment
- ✅ Rotate keys regularly

### Data Privacy
- ✅ All documents stored in private vector DB
- ✅ No user queries logged by default
- ✅ Langfuse tracing optional (can be disabled)

---

## 📚 Additional Resources

- **LangChain Docs**: https://python.langchain.com
- **Groq API**: https://console.groq.com
- **Pinecone Docs**: https://docs.pinecone.io
- **Streamlit Docs**: https://docs.streamlit.io
- **Langfuse Docs**: https://langfuse.com/docs

---

**Last Updated**: May 19, 2026  
**Version**: 2.0  
**Maintained by**: Steel Trade Intelligence Team
