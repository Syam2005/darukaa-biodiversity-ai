# Darukaa.Earth — Biodiversity Intelligence Chatbot

An AI environmental scientist, not a chatbot: a knowledge-grounded, multi-metric
reasoning system that turns soil/climate/land-use inputs into evidence-backed,
non-obvious biodiversity recommendations.

> **Live demo:** `<ADD_URL_HERE>`
> **API base (self-hosted):** `<ADD_URL_HERE>`

---

## 1. Architecture

```
┌─────────────┐      text / JSON       ┌─────────────────────┐
│  Frontend   │ ─────────────────────▶ │   FastAPI backend    │
│ (chat UI)   │ ◀───────────────────── │   /chat, /chat/...    │
└─────────────┘      structured JSON    └─────────┬────────────┘
                                                     │
                     ┌───────────────────────────────┼───────────────────────────────┐
                     ▼                               ▼                               ▼
            ┌─────────────────┐           ┌────────────────────┐          ┌────────────────────┐
            │ Conversation     │           │ Retrieval layer     │          │ Reasoning layer      │
            │ layer            │           │ (TF-IDF + cosine    │          │ (trigger-condition   │
            │ - session memory │──────────▶│  similarity over    │─────────▶│  matching, multi-    │
            │ - free-text field│           │  knowledge_base.json)│          │  metric scoring)     │
            │   extraction     │           └──────────┬───────────┘          └──────────┬───────────┘
            │ - clarifying-Q   │                      │                                 │
            │   logic          │                      ▼                                 ▼
            └─────────────────┘           ┌────────────────────┐          ┌────────────────────┐
                                            │ Structured knowledge│          │ Ranked, evidence-    │
                                            │ base (10 entries,   │          │ backed recommendations│
                                            │ soil/climate/land-  │          │ with mechanism,       │
                                            │ use/biodiversity/   │          │ source, confidence,   │
                                            │ human-impact)       │          │ time horizon           │
                                            └────────────────────┘          └────────────────────┘
```

**Pipeline per turn:**
1. The conversation layer merges any newly supplied fields (extracted from free
   text via regex, or sent explicitly as structured JSON) into the session's
   accumulated state — this is how multi-turn memory works.
2. If any of the three mandatory variables (soil organic carbon, rainfall,
   land use) is still missing, the system asks a targeted clarifying question
   instead of guessing.
3. Once the minimum is known, a natural-language query is built from all known
   fields and run through the **retrieval layer**: TF-IDF vectorization +
   cosine similarity over the structured knowledge base — a genuine
   retrieve-then-reason step, not a knowledge dump stuffed into a prompt.
4. The **reasoning layer** re-scores retrieved candidates by (a) whether their
   structured `trigger_condition` is actually satisfied by the user's numbers
   (e.g. `soil_organic_carbon < 0.8`), and (b) how many distinct environmental
   categories (soil / climate / land use / biodiversity / human impact) the
   entry connects — this is what rewards multi-variable, non-obvious answers
   over single-variable ones, and is what satisfies the "at least 3 variables
   together" requirement.
5. The top recommendations are returned with: what to do, why it works
   (mechanism), which metrics improve, a quantified expected impact, a time
   horizon, a confidence level, and a source citation (FAO / IPCC / CBD /
   IPBES).

**Design choices:**
- The retriever is swappable — `KnowledgeRetriever.query()` exposes the same
  interface a vector-DB-backed retriever (e.g. Chroma/FAISS +
  sentence-transformers) would use, so TF-IDF can be replaced with embeddings
  without touching the reasoning or conversation layers.
- Everything runs locally with no paid external API calls (no LLM API, no
  hosted vector DB), so the whole stack is free to self-host.

## 2. Database / Schema

There is no relational database — state is intentionally lightweight:

**Knowledge base** (`backend/data/knowledge.json`) — the retrievable knowledge
layer. Each entry:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique knowledge-base ID |
| `category` | string | One of: soil_health, climate_factors, land_use, biodiversity_indicators, human_impact |
| `trigger_metrics` | string[] | Which input variables this entry reasons about |
| `trigger_condition` | string | Boolean expression evaluated against user input (sandboxed eval, no builtins) |
| `title` / `intervention` | string | The recommendation itself |
| `mechanism` | string | Scientific reasoning for *why* it works |
| `impacted_metrics` | string[] | Which metrics improve |
| `expected_impact` | string | Quantified, sourced estimate |
| `time_horizon` | enum | short / medium / long |
| `confidence` | enum | high / medium / low |
| `source` | string | Citation (FAO / IPCC / CBD / IPBES) |
| `linked_variables` | string[] | Other environmental categories this connects to |

**Session memory** — an in-process dict (`SESSIONS: Dict[session_id, fields]`)
in `conversation.py`. Swappable for Redis/SQLite in production by replacing
that one module; the API surface (`get_session` / `update_session` /
`reset_session`) stays the same.

## 3. Local Setup

```bash
git clone <REPO_URL>
cd darukaa-biodiversity-ai

# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# -> API docs at http://localhost:8000/docs

# Frontend (separate terminal, from repo root)
cd frontend
python -m http.server 5173
# -> open http://localhost:5173  (chat UI talks to http://localhost:8000)
```

Or with Docker Compose (one command, both services):

```bash
docker compose up --build
# API:      http://localhost:8000
# Frontend: http://localhost:5173
```

### Try it

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Biodiversity is declining on my land"}'
# -> clarifying question

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid region"}'
# -> ranked, evidence-backed recommendations
```

Structured (JSON) input is also supported directly:

```bash
curl -X POST "http://localhost:8000/chat/structured?session_id=demo2" \
  -H "Content-Type: application/json" \
  -d '{"soil_organic_carbon":0.3,"rainfall":"low","land_use":"monoculture","region":"semi-arid"}'
```

Full knowledge base: `GET /knowledge`. Interactive API docs: `/docs`.

### Run tests

```bash
PYTHONPATH=backend python -m pytest tests/ -v
```

## 4. CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. Installs backend dependencies
2. Runs the pytest suite (multi-turn clarifying questions, multi-metric
   recommendation generation, structured-input endpoint)
3. Boots the API and hits `/health` as a smoke test

Deployment is container-based (`Dockerfile` + `docker-compose.yml`), so the
same image used in CI is what ships — deploy it to any host that can run a
container (Render/Fly.io/a VPS/etc. all work for a free-tier self-hosted
deployment).

## 5. Repository layout

```
backend/
  app/
    main.py          FastAPI endpoints
    conversation.py  session memory + free-text field extraction
    retrieval.py      TF-IDF retriever over the knowledge base
    reasoning.py      multi-metric scoring + recommendation generation
    knowledge_base.py loads data/knowledge.json
    schemas.py        Pydantic request/response models
  data/knowledge.json structured, sourced knowledge layer
frontend/index.html   vanilla-JS chat UI (no build step)
tests/test_api.py     pytest suite
.github/workflows/ci.yml
Dockerfile / docker-compose.yml
```

## 6. Meeting the brief

- **Knowledge system**: structured JSON knowledge layer (not prompt-stuffed),
  indexed and retrieved via TF-IDF/cosine similarity — see `retrieval.py`.
- **Conversational intelligence**: multi-turn session memory + targeted
  clarifying questions when inputs are incomplete — see `conversation.py`.
- **Evidence-backed recommendations**: every recommendation carries what/why/
  impacted-metric/source — see `knowledge.json` + `reasoning.py`.
- **Multi-metric reasoning**: reasoning layer explicitly scores by how many
  distinct environmental categories (soil/water/land-use/biodiversity/human
  impact) a recommendation connects, and requires 3 variables minimum before
  reasoning — see `reasoning.py::_category_span` and
  `missing_required_fields`.
- **Input handling**: text (`/chat`) and structured JSON
  (`/chat/structured`, or a `structured` field on `/chat`); `region`/`lat`/
  `lon` supported as spatial context.
- **Output quality**: every response includes recommendation, impacted
  metrics, time horizon, and confidence level.
