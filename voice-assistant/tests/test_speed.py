"""Tests for the speed work: sentence streaming and the manual-search fast path."""
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import pytest  # noqa: E402
import api  # noqa: E402


def events(gen):
    return [json.loads(e[len("data: "):]) for e in gen]


@pytest.fixture(autouse=True)
def _no_kb(monkeypatch):
    monkeypatch.setattr(api, "retrieve_manual_context", lambda q: None)


# ── Sentence splitting ───────────────────────────────────────────────────────
def test_split_keeps_decimals_and_list_markers_together():
    pieces = ["1. ఇంజిన్ ఆయిల్ సామ", "ర్థ్యం 3.5 లీటర్లు ఉంటుంది. రెండవ వాక్యం ఇక్కడ ఉంది, ఇది పొడవుగా ఉంటుంది.\nGrade: SAE 5W-30"]
    segs = list(api.split_sentences(pieces))
    assert len(segs) == 3
    assert "3.5 లీటర్లు" in segs[0] and segs[0].startswith("1. ")
    assert "".join(segs) == "".join(pieces)          # nothing lost or duplicated


def test_split_emits_first_sentence_before_stream_ends():
    def pieces():
        yield "ఇది మొదటి వాక్యం, ఇది తగినంత పొడవుగా ఉంది. "
        yield "రెండవ"
        raise AssertionError("consumer should have received the first sentence before this point")
    assert next(api.split_sentences(pieces())).startswith("ఇది మొదటి")


# ── Streaming pipeline ───────────────────────────────────────────────────────
def test_stream_event_order_and_content(monkeypatch):
    monkeypatch.setattr(api, "telugu_llm_stream", lambda msgs: iter(
        ["**ఇంజిన్ ఆయిల్** సామర్థ్యం 3.5 లీటర్లు ఉంటుంది. ", "ప్రతి నెలా ఆయిల్ స్థాయిని తప్పకుండా చూడండి.\nGrade: SAE 5W-30"]))
    ev = events(api._conversation_events(lambda: "ఆయిల్ ఎంత?", [], "/t"))
    assert [e["type"] for e in ev] == ["transcript", "chunk", "chunk", "chunk", "response", "done"]
    assert "**" not in ev[1]["text"] and "3.5" in ev[1]["text"]
    assert ev[4]["text"].count("\n") == 1 and "SAE 5W-30" in ev[4]["text"]
    assert ev[5]["history"][-1] == {"role": "assistant", "content": ev[4]["text"]}


def test_stream_instant_reply_skips_llm(monkeypatch):
    monkeypatch.setattr(api, "telugu_llm_stream", lambda msgs: pytest.fail("LLM must not be called"))
    ev = events(api._conversation_events(lambda: "హలో", [], "/t"))
    assert [e["type"] for e in ev] == ["transcript", "chunk", "response", "done"]


@pytest.mark.parametrize("stream", [lambda msgs: iter([]), lambda msgs: (_ for _ in ()).throw(RuntimeError("boom"))])
def test_stream_falls_back_to_apology_when_llm_gives_nothing(monkeypatch, stream):
    monkeypatch.setattr(api, "telugu_llm_stream", stream)
    ev = events(api._conversation_events(lambda: "ఏదైనా చెప్పు", [], "/t"))
    assert [e["type"] for e in ev] == ["transcript", "chunk", "response", "done"]
    assert ev[1]["text"] == ev[2]["text"] == api.LLM_FALLBACK_TEXT


def test_stream_empty_transcript_is_an_error():
    assert events(api._conversation_events(lambda: "  ", [], "/t")) == [{"type": "error", "msg": "Empty transcript"}]


class FakeModels:
    def __init__(self, behaviour):
        self.behaviour, self.called = behaviour, []

    def generate_content_stream(self, model, contents, config):
        self.called.append(model)
        return self.behaviour[model]()


def _chunks(*texts, fail=False):
    def gen():
        for t in texts:
            yield SimpleNamespace(text=t, candidates=[])
        if fail:
            raise RuntimeError("stream dropped")
    return gen


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


def test_llm_stream_falls_back_to_next_model_before_first_text(monkeypatch):
    fake = FakeModels({"a": _chunks(fail=True), "b": _chunks("సరే ", "జవాబు")})
    monkeypatch.setattr(api, "LLM_MODELS", ["a", "b"])
    monkeypatch.setattr(api, "gemini_client", SimpleNamespace(models=fake))
    assert list(api.telugu_llm_stream(MSGS)) == ["సరే ", "జవాబు"] and fake.called == ["a", "b"]


def test_llm_stream_does_not_restart_after_partial_answer(monkeypatch):
    fake = FakeModels({"a": _chunks("సగం జవాబు", fail=True), "b": _chunks("వేరే జవాబు")})
    monkeypatch.setattr(api, "LLM_MODELS", ["a", "b"])
    monkeypatch.setattr(api, "gemini_client", SimpleNamespace(models=fake))
    assert list(api.telugu_llm_stream(MSGS)) == ["సగం జవాబు"] and fake.called == ["a"]


# ── Manual search fast path ──────────────────────────────────────────────────
class FakeCollection:
    def __init__(self, dist_by_query):
        self.dist_by_query, self.queries = dist_by_query, []

    def query(self, query_embeddings, n_results):
        q = query_embeddings[0][0]
        self.queries.append(q)
        return {"documents": [["Oil capacity 3.5 L"]], "distances": [[self.dist_by_query[q]]],
                "metadatas": [[{"source": "manual.pdf"}]]}


class FakeEmbedder:
    def encode(self, texts):
        return SimpleNamespace(tolist=lambda: [[texts[0]]])   # the "embedding" is the query text


@pytest.fixture
def kb(monkeypatch):
    monkeypatch.undo()   # use the real retrieve_manual_context
    def setup(dists, skip_above=None):
        col = FakeCollection({f"query: {k}": v for k, v in dists.items()})
        calls = []
        monkeypatch.setattr(api, "_manuals_collection", col)
        monkeypatch.setattr(api, "_embed_model", FakeEmbedder())
        monkeypatch.setattr(api, "RAG_SKIP_TRANSLATE_ABOVE", skip_above)
        monkeypatch.setattr(api, "gemini_translate", lambda t, a, b: calls.append(t) or "oil capacity?")
        return col, calls
    return setup


def test_direct_hit_skips_translation(kb):
    col, calls = kb({"ఆయిల్ ఎంత?": 0.20})
    assert api.retrieve_manual_context("ఆయిల్ ఎంత?") == {"passage": "Oil capacity 3.5 L", "source": "manual.pdf"}
    assert calls == [] and len(col.queries) == 1


def test_direct_miss_retries_with_translation(kb):
    col, calls = kb({"ఆయిల్ ఎంత?": 0.45, "oil capacity?": 0.25})
    assert api.retrieve_manual_context("ఆయిల్ ఎంత?")["source"] == "manual.pdf"
    assert calls == ["ఆయిల్ ఎంత?"] and len(col.queries) == 2


def test_miss_on_both_returns_none(kb):
    _, calls = kb({"క్రికెట్ స్కోర్?": 0.70, "oil capacity?": 0.60})
    assert api.retrieve_manual_context("క్రికెట్ స్కోర్?") is None and len(calls) == 1


def test_clearly_unrelated_question_skips_translation_when_configured(kb):
    col, calls = kb({"క్రికెట్ స్కోర్?": 0.70}, skip_above=0.55)
    assert api.retrieve_manual_context("క్రికెట్ స్కోర్?") is None
    assert calls == [] and len(col.queries) == 1
