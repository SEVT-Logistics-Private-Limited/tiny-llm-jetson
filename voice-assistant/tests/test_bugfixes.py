"""Regression tests for the v10.1 bug fixes (text cleaning, TTS, audio types, feedback, models)."""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import pytest  # noqa: E402
import api  # noqa: E402

RAW = """**ఇంజిన్ ఆయిల్ వివరాలు:**
1. ఇంజిన్ ఆయిల్ సామర్థ్యం 3.5 లీటర్లు.
2. Grade: SAE 5W-30
* టైర్ ప్రెజర్ 32 psi (Pounds per Square Inch) ఉండాలి.
This means you should check the oil level every month carefully.
"""


# ── Text cleaning ────────────────────────────────────────────────────────────
def test_decimals_survive_tts_cleaning():
    assert "3.5 లీటర్లు" in api.clean_for_tts(api.clean_for_display(RAW))
    assert "2.5" in api.clean_for_tts("సామర్థ్యం 2.5 లీటర్లు, ఒత్తిడి 1.2 bar.")


def test_display_keeps_english_specs_and_explanations():
    d = api.clean_for_display(RAW)
    for kept in ("SAE 5W-30", "(Pounds per Square Inch)", "This means you should check", "3.5"):
        assert kept in d
    assert "**" not in d and "* " not in d


def test_tts_keeps_short_specs_drops_long_english_lines():
    t = api.clean_for_tts(api.clean_for_display(RAW))
    assert "SAE 5W-30" in t
    assert "This means" not in t and "Pounds" not in t
    assert "\n" not in t


def test_numbered_list_markers_removed_only_at_line_start():
    t = api.clean_for_tts("1. మొదటిది\n2. రెండోది 4.5 kg")
    assert t.startswith("మొదటిది") and "4.5 kg" in t


def test_english_only_answer_is_spoken_not_replaced():
    t = api.clean_for_tts("Check the ECU fuse box under the dashboard now.")
    assert "ECU" in t and "మళ్ళీ" not in t


def test_llm_returns_display_text(monkeypatch):
    monkeypatch.setattr(api, "_llm_call_with_fallback", lambda **kw: (RAW, "m"))
    out = api.telugu_llm([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
    assert "SAE 5W-30" in out and "This means" in out


# ── TTS ──────────────────────────────────────────────────────────────────────
class FakeGTTS:
    calls = []

    def __init__(self, text, tld, timeout, **kw):
        FakeGTTS.calls.append({"text": text, "tld": tld, "timeout": timeout})
        self.tld = tld

    def write_to_fp(self, fp):
        if self.tld == "co.in":
            raise TimeoutError("read timed out")
        fp.write(b"mp3")


def test_tts_uses_real_timeout_and_falls_back(monkeypatch):
    FakeGTTS.calls = []
    monkeypatch.setattr(api, "gTTS", FakeGTTS)
    assert api.do_tts("నమస్కారం") == (b"mp3", "audio/mpeg")
    assert [c["tld"] for c in FakeGTTS.calls] == ["co.in", "com"]
    assert all(c["timeout"] == api.GTTS_TIMEOUT for c in FakeGTTS.calls)


def test_tts_no_longer_truncates_at_500_chars(monkeypatch):
    FakeGTTS.calls = []
    monkeypatch.setattr(api, "gTTS", FakeGTTS)
    long_text = "ఇది ఒక పొడవైన వాక్యం. " * 60   # ~1300 chars
    api.do_tts(long_text)
    assert len(FakeGTTS.calls[-1]["text"]) > 1000


def test_truncation_cuts_at_sentence_boundary():
    out = api._truncate_at_sentence("మొదటి వాక్యం. రెండవ వాక్యం. మూడవ", 30)
    assert out.endswith(".") and len(out) <= 30


# ── Audio type ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("given,expected", [
    ("audio/webm;codecs=opus", "audio/webm"), ("audio/mp4", "audio/mp4"), ("AUDIO/OGG", "audio/ogg")])
def test_audio_mime_normalised(given, expected):
    assert api.AudioReq(audio_base64="AAAA", mime_type=given).mime_type == expected


def test_audio_mime_rejects_unknown():
    with pytest.raises(Exception):
        api.AudioReq(audio_base64="AAAA", mime_type="text/html")


def test_stt_receives_recording_mime(monkeypatch):
    seen = {}
    def fake_call(contents, config, context_label):
        seen["mime"] = contents[0].parts[0].inline_data.mime_type
        return "హలో", "m"
    monkeypatch.setattr(api, "_llm_call_with_fallback", fake_call)
    api.gemini_stt(b"x", "audio/mp4")
    assert seen["mime"] == "audio/mp4"


# ── Models ───────────────────────────────────────────────────────────────────
def test_retired_models_not_in_fallback_chain():
    assert api.LLM_MODELS and not {"gemini-2.0-flash", "gemini-1.5-flash"} & set(api.LLM_MODELS)
