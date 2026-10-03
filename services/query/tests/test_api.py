import json
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from query_service.pipeline.orchestrator import FALLBACK_ANSWER
from query_service.providers import LLMError

from conftest import ScriptedLLM

ANSWERABLE = "Do I need a credit card to start the free trial?"
UNRELATED = "zebra xylophone quokka"


def _sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ------------------------------------------------------------------ auth


def test_missing_token_is_401_problem_json(client: TestClient) -> None:
    resp = client.post("/v1/query", json={"query": ANSWERABLE})
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json()["title"] == "Unauthorized"


def test_bad_signature_is_401(client: TestClient) -> None:
    resp = client.post("/v1/query", json={"query": ANSWERABLE}, headers={"Authorization": "Bearer abc.def.ghi"})
    assert resp.status_code == 401


# ------------------------------------------------------------ validation


def test_unknown_field_is_422(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.post("/v1/query", json={"query": ANSWERABLE, "tenant_id": "other"}, headers=auth)
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["loc"] == ["body", "tenant_id"]


def test_query_too_long_is_413(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.post("/v1/query", json={"query": "x" * 4001}, headers=auth)
    assert resp.status_code == 413


# --------------------------------------------------------------- answers


def test_json_answer_with_citations(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.reply = "No credit card is needed for the 30-day trial [S1]."
    resp = client.post("/v1/query", json={"query": ANSWERABLE}, headers=auth)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_type"] == "answer"
    assert body["answer"] == "No credit card is needed for the 30-day trial [S1]."
    assert [c["id"] for c in body["citations"]] == ["S1"]
    assert body["citations"][0]["url"] == "pricing.md"
    assert body["prompt_version"] == "answer-v1"
    assert body["index_version"] == "corpus_v1"
    assert len(body["trace_id"]) == 32
    assert body["usage"]["input_tokens"] == 100
    assert body["timings_ms"]["retrieval"] is not None


def test_prompt_wraps_sources_and_question(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    client.post("/v1/query", json={"query": ANSWERABLE}, headers=auth)
    call = llm.calls[0]
    user_message = call["messages"][-1]["content"]  # type: ignore[index]
    assert '<source id="S1"' in user_message
    assert user_message.rstrip().endswith(f"Question: {ANSWERABLE}")
    assert "INSUFFICIENT_CONTEXT" in str(call["system"])


def test_invalid_citations_are_removed(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.reply = "No card is needed [S1][S99]."
    body = client.post("/v1/query", json={"query": ANSWERABLE}, headers=auth).json()
    assert body["answer"] == "No card is needed [S1]."
    assert [c["id"] for c in body["citations"]] == ["S1"]


def test_max_tokens_stop_is_partial(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.stop = "max_tokens"
    assert client.post("/v1/query", json={"query": ANSWERABLE}, headers=auth).json()["answer_type"] == "partial"


def test_refusal(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.stop = "refusal"
    assert client.post("/v1/query", json={"query": ANSWERABLE}, headers=auth).json()["answer_type"] == "refused"


# -------------------------------------------------------------- fallback


def test_no_relevant_chunks_skips_llm(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    body = client.post("/v1/query", json={"query": UNRELATED}, headers=auth).json()
    assert body["answer_type"] == "insufficient_context"
    assert body["answer"] == FALLBACK_ANSWER
    assert body["citations"] == []
    assert llm.calls == []


def test_model_sentinel_becomes_insufficient_context(
    client: TestClient, auth: dict[str, str], llm: ScriptedLLM
) -> None:
    llm.reply = "INSUFFICIENT_CONTEXT"
    resp = client.post("/v1/query", json={"query": ANSWERABLE}, headers={**auth, "Accept": "text/event-stream"})
    events = _sse(resp.text)
    assert "token" not in [name for name, _ in events]  # the sentinel never reaches the client
    final = events[-1][1]
    assert final["answer_type"] == "insufficient_context"
    assert final["answer"] == FALLBACK_ANSWER
    assert final["related_sources"]


# ------------------------------------------------------------------- ACL


def test_acl_hides_restricted_documents(client: TestClient, make_token: Callable[..., str]) -> None:
    question = "How do I fail over the metadata database with nimbusctl?"
    public = client.post(
        "/v1/query", json={"query": question}, headers={"Authorization": f"Bearer {make_token(['public'])}"}
    ).json()
    assert all(c["url"] != "oncall.md" for c in public["citations"] + public["related_sources"])

    engineer = client.post(
        "/v1/query",
        json={"query": question},
        headers={"Authorization": f"Bearer {make_token(['public', 'engineering'])}"},
    ).json()
    assert engineer["citations"][0]["url"] == "oncall.md"


def test_other_tenant_sees_nothing(client: TestClient, make_token: Callable[..., str]) -> None:
    token = make_token(["public"], tenant="someone-else")
    body = client.post("/v1/query", json={"query": ANSWERABLE}, headers={"Authorization": f"Bearer {token}"}).json()
    assert body["answer_type"] == "insufficient_context"
    assert body["related_sources"] == []


# ------------------------------------------------------------- streaming


def test_sse_event_order(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.reply = "No credit card is needed [S1]."
    resp = client.post("/v1/query", json={"query": ANSWERABLE}, headers={**auth, "Accept": "text/event-stream"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse(resp.text)
    names = [name for name, _ in events]
    assert names[:2] == ["meta", "retrieval"]
    assert names[-1] == "final"
    assert set(names[2:-1]) == {"token"}
    assert "".join(d["delta"] for n, d in events if n == "token") == "No credit card is needed [S1]."
    assert events[-1][1]["answer"] == "No credit card is needed [S1]."
    assert events[0][1]["trace_id"] == events[-1][1]["trace_id"]


# ---------------------------------------------------------------- errors


def test_llm_outage_is_503_with_retry_after(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.error = LLMError("upstream timeout", retryable=True)
    resp = client.post("/v1/query", json={"query": ANSWERABLE}, headers=auth)
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "5"
    assert resp.json()["retryable"] is True


def test_llm_outage_in_stream_ends_with_error_event(client: TestClient, auth: dict[str, str], llm: ScriptedLLM) -> None:
    llm.error = LLMError("bad request", retryable=False)
    resp = client.post("/v1/query", json={"query": ANSWERABLE}, headers={**auth, "Accept": "text/event-stream"})
    name, data = _sse(resp.text)[-1]
    assert name == "error"
    assert data == {"code": "llm_error", "message": "bad request", "retryable": False}


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
