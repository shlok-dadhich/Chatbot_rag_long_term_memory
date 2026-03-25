# 🧠 LangGraph Long-Term Memory Chatbot

A production-ready AI chatbot built with **LangGraph**, **Streamlit**, and **PostgreSQL** that remembers users across conversations, supports PDF document Q&A, and comes with a suite of real-world tools.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Long-Term Memory** | Automatically extracts & persists user facts (name, preferences, projects, goals) across all sessions |
| **Smart Deduplication** | Fuzzy-matching prevents the same fact from being stored twice |
| **Memory CRUD** | View, delete individual, or clear all memories from the UI |
| **Persistent Conversations** | Full conversation history stored in PostgreSQL via LangGraph checkpointer |
| **PDF RAG** | Upload PDFs per-thread or globally; ask questions grounded in document content |
| **Tool Use** | Web search (DuckDuckGo), calculator, stock prices (Alpha Vantage), weather (OpenWeatherMap) |
| **Auto Thread Titles** | Conversation titles generated automatically after the first exchange |
| **Dark UI** | Polished dark-theme Streamlit interface |

---

## 🗂️ Project Structure

```
langgraph-ltm-chatbot/
│
├── app.py                  # Streamlit entry point
│
├── backend/
│   ├── __init__.py
│   ├── config.py           # Env vars, constants, prompt templates
│   ├── database.py         # PostgreSQL pool, checkpointer, store
│   ├── llm.py              # HuggingFace chat model + embeddings
│   ├── rag.py              # PDF ingestion & FAISS retriever state
│   ├── tools.py            # All LangChain tools + llm_with_tools
│   ├── memory.py           # Memory schemas, helpers, CRUD API
│   ├── threads.py          # Thread metadata table management
│   └── graph.py            # LangGraph nodes + compiled chatbot
│
├── frontend/
│   ├── __init__.py
│   ├── styles.py           # Global CSS
│   └── utils.py            # Pure helper functions
│
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option A — Local (recommended for development)

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/langgraph-ltm-chatbot.git
cd langgraph-ltm-chatbot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and fill in your API keys (see Configuration section below)
```

**5. Start PostgreSQL**

You need a running Postgres instance. The easiest way:
```bash
docker run -d --name chatbot-pg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=chatbot -p 5432:5432 postgres:16-alpine
```
Or use any managed Postgres service (Neon, Supabase, Railway, etc.) and set `DATABASE_URL` in `.env`.

**6. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. 🎉

---

### Option B — Docker Compose (one-command setup)

```bash
cp .env.example .env
# Fill in your API keys in .env

docker compose up --build
```

This starts both the Postgres container and the Streamlit app. Open [http://localhost:8501](http://localhost:8501).

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in the values:

```env
# Required
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chatbot?sslmode=disable
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

# Optional (tools will return an error message if keys are missing)
ALPHA_VINTAGE_KEY=your_alphavantage_key_here
WEATHER_API_KEY=your_openweathermap_key_here

# App defaults
LTM_USER_ID=u1
COOKIE_SECRET=change-me-cookie-secret
```

| Variable | Where to get it | Required? |
|---|---|---|
| `DATABASE_URL` | Your Postgres connection string | ✅ Yes |
| `HUGGINGFACEHUB_API_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Yes |
| `ALPHA_VINTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Optional |
| `WEATHER_API_KEY` | [openweathermap.org](https://openweathermap.org/api) | Optional |
| `COOKIE_SECRET` | Any strong random string used to encrypt browser cookie data | Recommended |

---

## ☁️ Deployment

### Streamlit Community Cloud (free, recommended for sharing)

1. Push your code to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and main file `app.py`
4. Add your secrets under **Advanced settings → Secrets**:
   ```toml
   DATABASE_URL = "postgresql://user:pass@host/chatbot?sslmode=require"
   HUGGINGFACEHUB_API_TOKEN = "hf_..."
   ALPHA_VINTAGE_KEY = "..."
   WEATHER_API_KEY = "..."
   ```
5. Click **Deploy** — done!

> **Note:** Streamlit Cloud needs a cloud-hosted Postgres. Use [Neon](https://neon.tech) (free tier) or [Supabase](https://supabase.com) (free tier).
>
> If `DATABASE_URL` is missing/unreachable, the app now starts in **in-memory fallback mode**
> (no import crash), but conversation and long-term memory persistence will reset on restart.

---

### Railway (full Docker deployment)

1. Create a new project on [railway.app](https://railway.app)
2. Add a **PostgreSQL** plugin — Railway auto-sets `DATABASE_URL`
3. Connect your GitHub repo
4. Set environment variables in the Railway dashboard
5. Railway auto-detects `Dockerfile` and deploys

---

### Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repo
3. Set **Docker** as environment, port `8501`
4. Add a **PostgreSQL** database and link the connection string
5. Add all env vars and deploy

---

## 🧠 How Long-Term Memory Works

```
User message
     │
     ▼
┌─────────────┐    Extract facts with LLM    ┌──────────────────┐
│ remember_   │ ──────────────────────────► │  PostgresStore   │
│    node     │    Deduplicate & store       │  (long-term      │
└─────────────┘                              │   memories)      │
     │                                       └──────────────────┘
     ▼
┌─────────────┐    Load relevant memories    ┌──────────────────┐
│  chat_node  │ ◄──────────────────────────  │  Build system    │
│   (LLM)     │    Inject into system msg    │   prompt         │
└─────────────┘                              └──────────────────┘
     │
     ▼ (tool calls?)
┌─────────────┐
│  tool_node  │  → web search / calculator / stock / weather / RAG
└─────────────┘
     │
     ▼
  Response
```

Memories are categorised into: **profile**, **preferences**, **projects**, **goals**.

---

## 🛠️ Available Tools

| Tool | Description |
|---|---|
| `DuckDuckGoSearch` | Real-time web search |
| `calculator` | Basic arithmetic (add, sub, mul, div) |
| `get_stock_price` | Live stock quotes via Alpha Vantage |
| `get_weather` | Current weather via OpenWeatherMap |
| `rag_tool` | Semantic search over uploaded PDFs |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
