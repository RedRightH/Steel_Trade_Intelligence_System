# Steel Trade Intelligence System - Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Scripts & Files](#core-scripts--files)
4. [Steel RAG System](#steel-rag-system)
5. [Configuration Files](#configuration-files)
6. [Data Files](#data-files)
7. [Setup & Installation](#setup--installation)
8. [Usage Guide](#usage-guide)

---

## 🎯 Project Overview

The **Steel Trade Intelligence System** is a comprehensive AI-powered platform for analyzing international steel trade data. It combines:
- **UN Comtrade API integration** for real-time trade data
- **RAG (Retrieval-Augmented Generation)** system for document-based intelligence
- **Interactive dashboard** for visualization and querying
- **Multi-agent architecture** for specialized trade analysis

---

## 📁 Project Structure

```
AI_Trade_Capstone/
├── 📄 Core Scripts (UN Comtrade API)
│   ├── test_api_keys.py
│   ├── get_steel_exports.py
│   ├── get_steel_exports_specific_countries.py
│   └── plot_steel_exports.py
│
├── 🤖 Steel RAG System
│   ├── steel_rag/
│   │   ├── dashboard.py          # Streamlit UI
│   │   ├── rag.py                # Core RAG engine
│   │   ├── router.py             # Query routing logic
│   │   ├── data_agent.py         # Data analysis agent
│   │   ├── tariff_agent.py       # Tariff analysis agent
│   │   ├── classifier.py         # Query classification
│   │   ├── memory.py             # Conversation memory
│   │   ├── ingest.py             # FAISS indexing
│   │   ├── ingest_pinecone.py    # Pinecone indexing
│   │   ├── eval_harness.py       # Evaluation framework
│   │   └── run_eval.py           # Run evaluations
│
├── ⚙️ Configuration
│   ├── .env                      # API keys (not committed)
│   ├── .env.example              # Template for API keys
│   ├── requirements.txt          # Python dependencies
│   ├── packages.txt              # System packages
│   └── .gitignore                # Git exclusions
│
├── 📊 Data & Outputs
│   ├── Base documents/           # Source PDFs & data files
│   ├── chinese_steel_exports_*.csv
│   ├── chinese_steel_exports_plot.png
│   └── steel_rag/
│       ├── faiss_index/          # Local vector DB
│       ├── charts/               # Generated visualizations
│       └── classifier_model/     # ML model weights
│
├── 📖 Documentation
│   ├── README.md                 # Main project README
│   └── PROJECT_DOCUMENTATION.md  # This file
│
└── 🚀 Deployment
    └── app.py                    # Streamlit Cloud entry point
```

---

## 📄 Core Scripts & Files

### 1. **test_api_keys.py**
**Purpose:** Validate UN Comtrade API credentials

**What it does:**
- Tests both primary and secondary API keys
- Makes sample API calls to verify authentication
- Provides clear success/failure feedback

**Usage:**
```bash
python test_api_keys.py
```

**Key Features:**
- Tests India (code 356) total exports for 2023
- Validates API response format
- Error handling with detailed messages

**Dependencies:**
- `comtradeapicall` - UN Comtrade API wrapper
- `python-dotenv` - Environment variable management

---

### 2. **get_steel_exports.py**
**Purpose:** Retrieve top 10 export destinations for Indian steel (HS 7208)

**What it does:**
- Fetches Indian steel export data for 2023
- Identifies top 10 destination countries by export value
- Generates formatted console output and CSV export

**Usage:**
```bash
python get_steel_exports.py
```

**API Parameters:**
- **Reporter:** India (code 356)
- **HS Code:** 7208 (Flat-rolled products of iron/non-alloy steel)
- **Flow:** Exports (X)
- **Period:** 2023
- **Partners:** All countries

**Output:**
- Console: Ranked table with country, value, quantity
- File: `indian_steel_exports_top10.csv`

**Key Functions:**
- `get_top_10_steel_export_destinations()` - Main data retrieval function

---

### 3. **get_steel_exports_specific_countries.py**
**Purpose:** Analyze Chinese steel exports to specific countries over 10 years

**What it does:**
- Fetches Chinese steel exports (HS 72) to Belgium, Italy, UAE
- Retrieves data for 2014-2023 (10 years)
- Performs year-by-year API calls to handle rate limits
- Generates detailed reports and summaries

**Usage:**
```bash
python get_steel_exports_specific_countries.py
```

**API Parameters:**
- **Reporter:** China (code 156)
- **HS Code:** 72 (All iron and steel products)
- **Flow:** Exports (X)
- **Period:** 2014-2023
- **Partners:** Belgium (56), Italy (380), UAE (784)

**Output Files:**
- `chinese_steel_exports_specific_countries.csv` - Detailed year-by-year data
- `chinese_steel_exports_summary.csv` - Aggregated summary by country

**Key Features:**
- Iterative fetching (one year + country at a time)
- Progress tracking for each API call
- Comprehensive error handling
- Summary statistics by country

**Country Codes:**
```python
COUNTRY_CODES = {
    'Belgium': '56',
    'Italy': '380',
    'United Arab Emirates': '784'
}
```

---

### 4. **plot_steel_exports.py**
**Purpose:** Visualize Chinese steel export trends

**What it does:**
- Reads CSV data from `get_steel_exports_specific_countries.py`
- Creates multi-line time series plot
- Generates summary statistics

**Usage:**
```bash
python plot_steel_exports.py
```

**Prerequisites:**
- Must run `get_steel_exports_specific_countries.py` first

**Output:**
- `chinese_steel_exports_plot.png` - High-resolution chart (300 DPI)
- Console: Summary statistics (total, average, peak year)

**Visualization Features:**
- 14x8 inch figure size
- Color-coded lines (Belgium: blue, Italy: orange, UAE: green)
- Formatted Y-axis (billions/millions USD)
- Grid lines for readability
- Legend with country names

**Statistics Displayed:**
- Total export value (2014-2023)
- Average per year
- Peak year and value

---

## 🤖 Steel RAG System

The `steel_rag/` directory contains a sophisticated RAG-based intelligence system.

### **dashboard.py**
**Purpose:** Streamlit web interface for the Steel Trade Intelligence System

**What it does:**
- Provides interactive chat interface
- Displays generated charts and tables
- Manages conversation history
- Routes queries to appropriate agents

**Key Features:**
- Multi-turn conversations with memory
- Real-time chart generation
- Query classification display
- Session state management
- Responsive UI with custom styling

**Components:**
- Chat interface with message history
- Sidebar with system information
- Chart display area
- Query routing visualization

---

### **rag.py**
**Purpose:** Core RAG (Retrieval-Augmented Generation) engine

**What it does:**
- Retrieves relevant documents from vector database
- Generates context-aware responses using LLM
- Manages document embeddings and similarity search

**Key Components:**
- **Vector Store:** FAISS or Pinecone for document retrieval
- **Embeddings:** Sentence transformers for semantic search
- **LLM:** Groq API for response generation

**Functions:**
- `retrieve_documents()` - Semantic search in vector DB
- `generate_response()` - LLM-based answer generation
- `format_context()` - Prepare retrieved docs for LLM

---

### **router.py**
**Purpose:** Intelligent query routing system

**What it does:**
- Classifies incoming queries by type
- Routes to appropriate specialized agent
- Manages multi-agent workflow

**Query Types:**
- **Data queries:** Route to `data_agent.py`
- **Tariff queries:** Route to `tariff_agent.py`
- **General queries:** Use base RAG system

**Routing Logic:**
```python
if is_data_query(query):
    return data_agent.handle(query)
elif is_tariff_query(query):
    return tariff_agent.handle(query)
else:
    return rag.generate_response(query)
```

---

### **data_agent.py**
**Purpose:** Specialized agent for data analysis and visualization

**What it does:**
- Processes numerical/statistical queries
- Generates charts and graphs
- Performs data aggregation and analysis
- Handles TRADESTAT Excel files

**Capabilities:**
- Time series analysis
- Comparative analysis across countries
- Export/import trend visualization
- Statistical summaries

**Chart Types:**
- Line charts (trends over time)
- Bar charts (comparisons)
- Stacked charts (composition)

---

### **tariff_agent.py**
**Purpose:** Specialized agent for tariff and trade policy analysis

**What it does:**
- Analyzes tariff rates and structures
- Compares tariff schedules across countries
- Provides policy recommendations
- Handles HS code lookups

**Key Features:**
- HS code classification
- Tariff rate comparisons
- Trade agreement analysis
- Duty calculation support

---

### **classifier.py**
**Purpose:** ML-based query classification

**What it does:**
- Classifies user queries into categories
- Determines query intent
- Supports router decision-making

**Classification Categories:**
- Data/statistical queries
- Tariff/policy queries
- General information queries
- Document retrieval queries

**Model:**
- Fine-tuned transformer model
- Stored in `classifier_model/`

---

### **memory.py**
**Purpose:** Conversation context management

**What it does:**
- Maintains conversation history
- Provides context for follow-up questions
- Manages session state

**Features:**
- Multi-turn conversation support
- Context window management
- Memory summarization for long conversations

---

### **ingest.py**
**Purpose:** Build local FAISS vector index

**What it does:**
- Processes PDF documents from `Base documents/`
- Chunks documents into semantic units
- Generates embeddings
- Creates FAISS index

**Usage:**
```bash
python steel_rag/ingest.py
```

**Output:**
- `steel_rag/faiss_index/` - Local vector database

**Processing Steps:**
1. Load PDFs from source directory
2. Split into chunks (configurable size)
3. Generate embeddings using sentence-transformers
4. Build and save FAISS index

---

### **ingest_pinecone.py**
**Purpose:** Build cloud-based Pinecone vector index

**What it does:**
- Uploads document embeddings to Pinecone
- Enables cloud-based vector search
- Supports distributed deployment

**Usage:**
```bash
python steel_rag/ingest_pinecone.py
```

**Advantages over FAISS:**
- Cloud-hosted (no local storage)
- Scalable to millions of vectors
- Supports distributed queries

---

### **eval_harness.py**
**Purpose:** Evaluation framework for RAG system

**What it does:**
- Defines evaluation metrics
- Provides test harness for system validation
- Measures retrieval and generation quality

**Metrics:**
- Retrieval accuracy
- Answer relevance
- Factual consistency
- Response latency

---

### **run_eval.py**
**Purpose:** Execute evaluation suite

**What it does:**
- Runs predefined test cases
- Generates evaluation reports
- Compares system versions

**Usage:**
```bash
python steel_rag/run_eval.py
```

---

## ⚙️ Configuration Files

### **.env**
**Purpose:** Store sensitive API keys and configuration

**Contents:**
```bash
GROQ_API_KEY=<your_key>
LANGFUSE_PUBLIC_KEY=<your_key>
LANGFUSE_SECRET_KEY=<your_key>
COMTRADE_PRIMARY_KEY=<your_key>
COMTRADE_SECONDARY_KEY=<your_key>
```

**Security:** 
- ❌ Never commit to Git (in `.gitignore`)
- ✅ Use `.env.example` as template

---

### **.env.example**
**Purpose:** Template for environment variables

**What it does:**
- Documents required API keys
- Provides setup instructions
- Safe to commit to version control

**Setup:**
```bash
cp .env.example .env
# Edit .env with your actual keys
```

---

### **requirements.txt**
**Purpose:** Python package dependencies

**Categories:**

**Core RAG Pipeline:**
- `langchain-community` - LangChain framework
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Embeddings
- `groq` - LLM API client

**Vector Database:**
- `pinecone` - Cloud vector DB

**Data & Visualization:**
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `openpyxl` - Excel file parsing

**Dashboard:**
- `streamlit` - Web UI framework

**Observability:**
- `langfuse` - LLM tracing (optional)

**Legacy Scripts:**
- `comtradeapicall` - UN Comtrade API
- `requests` - HTTP client

**Installation:**
```bash
pip install -r requirements.txt
```

---

### **packages.txt**
**Purpose:** System-level dependencies for Streamlit Cloud

**Contents:**
```
libgomp1
```

**What it does:**
- Installs OpenMP library for parallel processing
- Required for FAISS and sentence-transformers

---

### **.gitignore**
**Purpose:** Exclude files from version control

**Key Exclusions:**

**Security:**
- `.env` - API keys and secrets

**Generated Files:**
- `__pycache__/` - Python bytecode
- `*.pyc`, `*.pyo` - Compiled Python
- `steel_rag/faiss_index/` - Vector DB (rebuild with ingest.py)
- `steel_rag/charts/` - Generated visualizations
- `steel_rag/classifier_model/` - ML model weights

**Large Files:**
- `Base documents/**/*.pdf` - Source PDFs (indexed in Pinecone)

**Development:**
- `.vscode/`, `.idea/` - IDE settings
- `*.ipynb` - Jupyter notebooks
- `.ipynb_checkpoints/` - Notebook checkpoints

---

### **app.py**
**Purpose:** Streamlit Cloud deployment entry point

**What it does:**
- Adds project directories to Python path
- Imports and runs `steel_rag/dashboard.py`
- Enables deployment to Streamlit Community Cloud

**Why needed:**
- Streamlit Cloud expects `app.py` at repo root
- Actual dashboard code is in `steel_rag/dashboard.py`
- This shim bridges the two

**Deployment:**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Point to `app.py`
4. Add secrets in Streamlit Cloud dashboard

---

## 📊 Data Files

### **Generated CSV Files**

#### **chinese_steel_exports_specific_countries.csv**
- Detailed year-by-year export data
- Columns: period, reporterDesc, partnerDesc, primaryValue, qty, netWgt, etc.
- 30 records (10 years × 3 countries)

#### **chinese_steel_exports_summary.csv**
- Aggregated totals by country
- Columns: Country, Total Export Value, Total Quantity, Total Net Weight
- 3 records (one per country)

### **Generated Visualizations**

#### **chinese_steel_exports_plot.png**
- Line chart showing export trends 2014-2023
- 300 DPI high-resolution image
- Suitable for reports and presentations

### **Base Documents/**
- Source PDFs and Excel files
- Trade statistics, tariff schedules, policy documents
- Indexed by `ingest.py` or `ingest_pinecone.py`

---

## 🚀 Setup & Installation

### **1. Clone Repository**
```bash
git clone <repository_url>
cd AI_Trade_Capstone
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Configure API Keys**
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

**Required Keys:**
- **Groq API:** https://console.groq.com
- **UN Comtrade:** https://comtradeplus.un.org
- **Langfuse (optional):** https://cloud.langfuse.com

### **5. Build Vector Index**

**Option A: Local FAISS**
```bash
python steel_rag/ingest.py
```

**Option B: Cloud Pinecone**
```bash
python steel_rag/ingest_pinecone.py
```

### **6. Run Dashboard**
```bash
streamlit run app.py
```

---

## 📖 Usage Guide

### **Testing API Keys**
```bash
python test_api_keys.py
```
Expected output: ✅ validation for both keys

### **Fetching Trade Data**

**Indian Steel Exports (Top 10):**
```bash
python get_steel_exports.py
```

**Chinese Steel Exports (Specific Countries):**
```bash
python get_steel_exports_specific_countries.py
```

### **Generating Visualizations**
```bash
python plot_steel_exports.py
```
Requires: `chinese_steel_exports_specific_countries.csv`

### **Running the Dashboard**
```bash
streamlit run app.py
```
Access at: http://localhost:8501

### **Evaluating System Performance**
```bash
python steel_rag/run_eval.py
```

---

## 🔧 Customization

### **Modify Target Countries**
Edit `get_steel_exports_specific_countries.py`:
```python
COUNTRY_CODES = {
    'Belgium': '56',
    'Italy': '380',
    'United Arab Emirates': '784',
    'Germany': '276',  # Add new country
}
```

### **Change Time Period**
Modify the `years` parameter:
```python
result = get_steel_exports_to_countries(years=5)  # Last 5 years
```

### **Adjust HS Code**
Change commodity classification:
```python
cmdCode='7208'  # Specific: Flat-rolled products
cmdCode='72'    # Broad: All iron and steel
```

---

## 📚 Additional Resources

### **UN Comtrade API Documentation**
- https://comtradeplus.un.org/
- API reference and country/commodity codes

### **HS Code Reference**
- Chapter 72: Iron and steel
- 7208: Flat-rolled products of iron or non-alloy steel

### **Streamlit Documentation**
- https://docs.streamlit.io
- Dashboard customization guide

### **LangChain Documentation**
- https://python.langchain.com
- RAG implementation patterns

---

## 🐛 Troubleshooting

### **API Key Errors**
- Verify keys in `.env` file
- Check API quota/rate limits
- Ensure no extra spaces in key values

### **No Data Retrieved**
- Verify country codes (use UN Comtrade reference)
- Check HS code validity
- Confirm year availability (some data may be delayed)

### **Import Errors**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Activate virtual environment
- Check Python version (3.8+)

### **Vector Index Issues**
- Rebuild index: `python steel_rag/ingest.py`
- Check `Base documents/` directory has PDFs
- Verify sufficient disk space

---

## 📝 Version History

**v1.0** - Initial release
- UN Comtrade API integration
- Basic data retrieval scripts
- Visualization capabilities

**v2.0** - RAG System
- Multi-agent architecture
- Streamlit dashboard
- Vector database integration
- Conversation memory

---

## 👥 Contributing

For questions or contributions, please refer to the main README.md file.

---

**Last Updated:** May 19, 2026
**Maintained by:** Steel Trade Intelligence Team
