"""Tests: voice mode must fall back to HERMES SERVER-SIDE STT when the browser
SpeechRecognition engine is unavailable or cannot reach its cloud backend.

Context (#voice-mode-server-stt): voice mode (boot.js ``_startListening``) was
hard-wired to ``window.SpeechRecognition`` (webkit SpeechRecognition), a
browser-vendor cloud service. In Chromium forks (Brave), behind VPNs, or on
browsers without the Web Speech API, the recognizer silently produces no
results; the old ``onend`` handler just re-armed the recognition, leaving the
UI stuck on "Listening…" forever with no error shown.

Fix: voice mode now prefers the same capture path the 🎤 dictation button
already uses — MediaRecorder + POST /api/transcribe — when the server
confirms an STT provider via /api/transcribe/capability. Speech onsets are
detected with a small RMS VAD (same family as the existing barge-in VAD) and
the clip is transcribed + auto-sent after a silence gap. Browser
SpeechRecognition remains the fallback when server STT is unavailable.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BOOT = REPO / "static" / "boot.js"
VAD_JS = REPO / "tests" / "voice_mode_vad.test.js"


def _boot_src() -> str:
    return BOOT.read_text(encoding="utf-8")


class TestVoiceModeServerSttEngine:
    def test_engine_function_present(self):
        src = _boot_src()
        assert "function _startListeningServerStt()" in src, (
            "Voice mode must define _startListeningServerStt() — the "
            "MediaRecorder + /api/transcribe capture engine."
        )

    def test_capability_probe_with_local_storage_latch(self):
        src = _boot_src()
        assert re.search(
            r"fetch\('api/transcribe/capability'", src
        ), "Voice mode must probe /api/transcribe/capability."
        assert "hermes-voice-mode-server-stt" in src, (
            "The probe result must be latched in localStorage "
            "(hermes-voice-mode-server-stt) so a one-time capability is "
            "honored across restarts, mirroring mic mic_force_mediarecorder."
        )

    def test_engine_selection_in_activate(self):
        src = _boot_src()
        assert re.search(
            r"_startListeningServerStt\(\)", src
        ), "_activate() must route to _startListeningServerStt() when server STT is available."

    def test_server_transcribe_call(self):
        src = _boot_src()
        # The engine must POST the recorded blob to the same endpoint the
        # mic dictation uses (one authoritative transcription path).
        assert re.search(
            r"fetch\('api/transcribe',\s*\{method:'POST'", src
        ), "Voice-mode server engine must transcribe via POST /api/transcribe."

    def test_speech_recognition_still_fallback(self):
        src = _boot_src()
        assert "_recognition=new SpeechRecognition()" in src, (
            "Browser SpeechRecognition must remain as the fallback engine "
            "when server STT is unavailable (existing #4761 behavior)."
        )
        assert "function _startListeningBrowser()" in src, (
            "The old browser-only loop must be preserved as "
            "_startListeningBrowser() so the fallback path is intact."
        )

    def test_silent_failure_no_longer_silent_on_server_path(self):
        """The server-STT engine must surface errors (toast) instead of
        silently re-arming forever — the original user-visible bug was an
        error-free eternal 'Listening…'."""
        src = _boot_src()
        # The whole server-STT section: capability probe + transcribe handler
        # + listener, up to (not including) the browser fallback.
        engine = src[src.index("function _probeVoiceModeServerStt()"):]
        engine = engine[: engine.index("function _startListeningBrowser()")]
        assert "showToast" in engine, (
            "The server-STT engine must toast on failures (mic denied, "
            "transcription error) instead of restarting silently."
        )
        assert "voice_mode_transcribe_error" in engine, (
            "Repeated transcription failures must produce a visible toast."
        )
        # On repeated transcription failures it must stop, not loop forever.
        assert re.search(r"_vmSttFails", engine), (
            "Repeated server-STT failures must be counted and lead to "
            "deactivation/toast, not an endless re-arm loop."
        )


class TestVoiceModeVadConstants:
    """The VAD in boot.js must use the same constants as its decision-logic
    test (tests/voice_mode_vad.test.js) so the mirrored logic can't drift."""

    def test_onset_constant(self):
        boot = _boot_src()
        vad = VAD_JS.read_text(encoding="utf-8")
        m_boot = re.search(r"_vmOnsetFrames\s*=\s*(\d+)", boot)
        m_vad = re.search(r"const ONSET=(\d+);", vad)
        assert m_boot and m_vad, (
            "Both boot.js (_vmOnsetFrames) and voice_mode_vad.test.js "
            "(ONSET) must define the onset frame count."
        )
        assert int(m_boot.group(1)) == int(m_vad.group(1)), (
            "Onset frame count drifted between boot.js and its test."
        )

    def test_silence_gap_constant(self):
        boot = _boot_src()
        vad = VAD_JS.read_text(encoding="utf-8")
        m_boot = re.search(r"_vmSilenceFrames\s*=\s*(\d+)", boot)
        m_vad = re.search(r"const SILENCE=(\d+);", vad)
        assert m_boot and m_vad, (
            "Both boot.js (_vmSilenceFrames) and voice_mode_vad.test.js "
            "(SILENCE) must define the post-speech silence gap."
        )
        assert int(m_boot.group(1)) == int(m_vad.group(1)), (
            "Silence-gap frame count drifted between boot.js and its test."
        )

    def test_adaptive_noise_floor(self):
        src = _boot_src()
        assert re.search(r"\*0\.95\+rms\*0\.05", src), (
            "The VAD must use the same adaptive noise-floor update as the "
            "barge-in VAD (noise*0.95 + rms*0.05)."
        )


class TestDeactivateReleasesServerSttState:
    def test_deactivate_stops_recorder_and_stream(self):
        src = _boot_src()
        deact = src[src.index("function _deactivate()"):]
        deact = deact[: deact.index("modeBtn.onclick")]
        assert "_vmStopCapture()" in deact, (
            "_deactivate() must release the voice-mode server-STT capture "
            "(recorder, analyser, stream) via _vmStopCapture(), otherwise the "
            "mic is left hot after exiting voice mode."
        )
        # And _vmStopCapture must actually stop the recorder and tracks.
        body = src[src.index("function _vmStopCapture()"):]
        body = body[: body.index("function _vmTranscribeStop(")]
        assert "_vmRecorder" in body and "_vmStream" in body and "getTracks" in body

    def test_barage_in_still_supported(self):
        """Server-STT voice mode must still barge in (interrupt TTS) on
        speech onset — barge-in is the core reason voice mode exists."""
        src = _boot_src()
        assert "function _bargeTick" in src
        # The dedicated barge-in machinery (own analyser, armed by
        # _setState('speaking')) must be untouched by the engine change.
        assert "_bargeStart" in src and "_bargeAnalyser" in src
        # _setState must still arm barge-in on the 'speaking' state.
        setstate = src[src.index("function _setState("):]
        setstate = setstate[: setstate.index("function _probeVoiceModeServerStt(")]
        assert "_bargeStart()" in setstate
