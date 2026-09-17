import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_clarifying_question_on_vague_input():
    r = client.post("/chat", json={"session_id": "t1", "message": "Biodiversity is declining on my land"})
    body = r.json()
    assert body["reply_type"] == "clarifying_question"
    assert "soil_organic_carbon" in body["missing_fields"]


def test_multi_turn_reaches_recommendations():
    client.delete("/session/t2")
    client.post("/chat", json={"session_id": "t2", "message": "soil organic carbon is 0.3%"})
    client.post("/chat", json={"session_id": "t2", "message": "rainfall is low"})
    r = client.post("/chat", json={"session_id": "t2", "message": "monoculture wheat, semi-arid region"})
    body = r.json()
    assert body["reply_type"] == "recommendations"
    assert len(body["recommendations"]) > 0
    # every recommendation must be evidence-backed
    for rec in body["recommendations"]:
        assert rec["source"]
        assert rec["expected_impact"]
        assert rec["time_horizon"] in ("short", "medium", "long")


def test_multi_metric_reasoning_spans_multiple_categories():
    client.delete("/session/t3")
    client.post("/chat", json={"session_id": "t3", "message": "soil organic carbon is 0.3%, rainfall is low, monoculture wheat, semi-arid region"})
    r = client.post("/chat", json={"session_id": "t3", "message": "any more detail?"})
    recs = r.json()["recommendations"]
    all_linked = set()
    for rec in recs:
        all_linked.update(rec["linked_variables"])
    # must connect more than one environmental category (soil/water/land-use/biodiversity)
    assert len(all_linked) >= 3


def test_structured_endpoint():
    r = client.post(
        "/chat/structured?session_id=t4",
        json={
            "soil_organic_carbon": 0.3,
            "rainfall": "low",
            "land_use": "monoculture",
            "region": "semi-arid",
        },
    )
    body = r.json()
    assert body["reply_type"] == "recommendations"
    assert len(body["recommendations"]) > 0
