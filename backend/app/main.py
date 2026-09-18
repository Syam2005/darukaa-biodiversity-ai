from fastapi.staticfiles import StaticFiles
from pathlib import Path
"""Darukaa.Earth Biodiversity Intelligence Chatbot - API entrypoint.

Endpoints:
  POST /chat              - text input, multi-turn, session-memory aware
  POST /chat/structured   - structured (JSON) input, same reasoning path
  GET  /session/{id}      - inspect accumulated known fields for a session
  DELETE /session/{id}    - reset a session
  GET  /knowledge         - list the structured knowledge base (transparency
                             into what the retrieval layer indexes)
  GET  /health            - liveness check
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatMessage, ChatResponse, RecommendationOut, StructuredInput
from .conversation import extract_fields_from_text, get_session, update_session, reset_session
from .reasoning import generate_recommendations, missing_required_fields, ELICITABLE_FIELDS
from .knowledge_base import knowledge_base

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence Chatbot",
    description="Knowledge-grounded, multi-metric environmental reasoning API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _question_for_field(field: str) -> str:
    for f, q in ELICITABLE_FIELDS:
        if f == field:
            return q
    return f"Could you share {field.replace('_', ' ')}?"


def _handle_turn(session_id: str, free_text: str, structured: StructuredInput | None) -> ChatResponse:
    text_fields = extract_fields_from_text(free_text) if free_text else {}
    structured_fields = structured.known_fields() if structured else {}
    # structured (explicit) input takes precedence over text extraction
    merged_new = {**text_fields, **structured_fields}
    known = update_session(session_id, merged_new)

    missing = missing_required_fields(known)
    if missing:
        question = _question_for_field(missing[0])
        return ChatResponse(
            session_id=session_id,
            reply_type="clarifying_question",
            message=(
                "I can help, but I need a bit more to reason across soil, "
                f"water and land-use variables together. {question}"
            ),
            missing_fields=missing,
            known_fields=known,
            recommendations=[],
        )

    recs = generate_recommendations(known, free_text=free_text, top_k=4)
    rec_objs = [RecommendationOut(**r) for r in recs]
    summary = (
        f"Based on {', '.join(f'{k}={v}' for k, v in known.items())}, "
        f"here are {len(rec_objs)} evidence-backed recommendations connecting "
        "soil, water/climate and biodiversity together."
    )
    return ChatResponse(
        session_id=session_id,
        reply_type="recommendations",
        message=summary,
        missing_fields=[],
        known_fields=known,
        recommendations=rec_objs,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatMessage):
    return _handle_turn(payload.session_id, payload.message, payload.structured)


@app.post("/chat/structured", response_model=ChatResponse)
def chat_structured(session_id: str, structured: StructuredInput):
    return _handle_turn(session_id, "", structured)


@app.get("/session/{session_id}")
def read_session(session_id: str):
    return get_session(session_id)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/knowledge")
def list_knowledge():
    return knowledge_base.all()


@app.get("/health")
def health():
    return {"status": "ok"}

# Serve the frontend from the same FastAPI service.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
