"""
Module contains transcribers.
"""
from pathlib import Path

import wave
import numpy as np
from deepspeech import Model


class DeepSpeechTranscriber:
    """
    Transcribes an audio file using Mozilla DeepSpeech

    Resources:
        Find more information on Mozilla DeepSpeech on their GitHub Repo:
        https://github.com/mozilla/DeepSpeech

    """

    def __init__(self, model_file: Path, scorer_file: Path):
        self.ds = Model(str(model_file))
        self.ds.enableExternalScorer(str(scorer_file))

    def transcribe(self, segment) -> str:
        """
        Transcribe a segment of audio.

        TODO:
            Wrap pydub's AudioSegment into a custom type, so that we
            don't have to rely on pydub here. Best place is to change
            it in the Pipeline class.

        """
        segment_as_wav = segment.export(format="wav")
        with wave.open(segment_as_wav, 'r') as w:
            frames = w.getnframes()
            buffer = w.readframes(frames)
            data = np.frombuffer(buffer, dtype=np.int16)

        return self.ds.stt(data)
