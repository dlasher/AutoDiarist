import pytest
from pipeline.transcribe import decode_audio, transcribe_audio, TranscodeError

def test_transcribe_mock():
    segments = transcribe_audio("dummy.wav", mock=True)
    assert len(segments) == 2
    assert segments[0]["speaker"] == "Speaker 1"
    assert segments[1]["t0"] > segments[0]["t1"]

def test_decode_audio_bad_file():
    with pytest.raises(TranscodeError):
        decode_audio("nonexistent.wav", "/tmp/out.wav")
