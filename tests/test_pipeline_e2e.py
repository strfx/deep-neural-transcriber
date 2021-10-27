"""
Simple end-to-end tests for the subtitle generation pipeline.

These tests do not aim to be comprehensive, but rather provide a simple way to
tell if the pipeline has defects. Ultimatly, it just shows that the app can
transcribe some video using the pre-trained models and return the subtitles.

NOTE: These tests only work with the tflite packages (i.e., deepspeech-tflite).
The tests will break if you use any other format.
"""
from pathlib import Path

import pytest

from dnt.core import Pipeline
from dnt.preprocessing import IntervalSegmenter, extract_audio
from dnt.transcription import DeepSpeechTranscriber
from dnt.translation import NopTranslator
from dnt.subtitles import SubtitleFormat, Subtitles


TEST_VIDEO = Path('tests/data/fmsd-sample-video.mp4')
MODELS_PATH = Path("models/pretrained-v0.9.3")
TEST_MODEL_PATH = MODELS_PATH / "deepspeech-0.9.3-models.tflite"
TEST_SCORER_PATH = MODELS_PATH / "deepspeech-0.9.3-models.scorer"


if not TEST_MODEL_PATH.exists():
    raise RuntimeError(
        f"No acoustic model found at specified path: '{TEST_MODEL_PATH}'")


if not TEST_SCORER_PATH.exists():
    raise RuntimeError(
        f"No language model (scorer) found at specified path: '{TEST_SCORER_PATH}'")


@pytest.mark.integration
def test_pipeline_generates_subtitles(tmp_path):
    example_pipeline = Pipeline(
        IntervalSegmenter(),
        DeepSpeechTranscriber(TEST_MODEL_PATH, TEST_SCORER_PATH),
        NopTranslator(),
        [SubtitleFormat.from_suffix('vtt'), SubtitleFormat.from_suffix('srt')]
    )

    wavfile = tmp_path / "sample.wav"

    extract_audio(TEST_VIDEO, wavfile)

    subtitles = example_pipeline.process(wavfile, keep_original=True)
    # We expected four types of subtitles:
    # - English (original) subtitles in SRT format
    # - English (original) subtitles in VTT format
    # - German (translated) subtitles in SRT format
    # - German (translated) subtitles in VTT format
    assert len(subtitles) == 4
