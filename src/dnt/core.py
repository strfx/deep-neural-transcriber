"""
This module contains the core pipeline to transcribe audio files.

Currently, the module only implements a single, sequential pipeline. Feel free
to create your own pipeline implementations depending on your needs, like a
pipeline that supports multi-threading/processing.
"""
from pathlib import Path
from typing import List

from dnt.subtitles import Subtitles
from dnt.translation import Translator
from dnt.utils import listify


class Pipeline:
    """
    Simple, sequential transcription pipeline.

    TODO: The pipeline's configuration could be solved more elegantely. There
    are some hardcoded parts in here, or in each step, which could be easily be
    extracted into a central config type that holds all this information. 
    """

    def __init__(self, segmenter, transcriber, translator: Translator, subtitle_formats):
        """
        Initialize the pipeline.

        Args:
            segmenter: How to split the audio file into smaller pieces.

            transcriber: How to produce a textual transcript of the spoken words
                in the audio.

            translator: How to translate the transcript into a
                target language.

            subtitle_format: How to format the subtitles.

        """
        self.segmenter = segmenter
        self.transcriber = transcriber
        self.translator = translator
        self.subtitle_formats = listify(subtitle_formats)

    def process(self, audiofile: Path, keep_original: bool = True) -> List[Subtitles]:
        """
        Run the transcription pipeline on given audio file.

        TODO: Atm. process() only generates english & german subtitles.
        Add some parameters for that.

        The pipeline works in four steps:
        1. Segment the input audio into smaller segments
        2. Transcribe each segment (i.e., convert speech to text)
        3. Translate each transcript into a target language (text to text)
        4. Finally, generate subtitle files in configured formats.

        Args:
            audiofile: Location of the audio file to transcribe.
            keep_original: When set to True, the pipeline also generates
                subtitles in the audio's source language.

        Example:
            >>> some_audio = Path("some-audio-file.wav")
            >>> pipeline.process(some_audio, keep_original=True)
                [
                    Subtitle(format='vtt', language='de', ...), 
                    Subtitle(format='vtt', language='en', ...), 
                    Subtitle(format='srt', language='de', ...), 
                    Subtitle(format='srt', language='en', ...)
                ]

        Returns:
            A list of generated subtitles.

        """

        # 1. Segment the input audio into segments
        segments = self.segmenter.segment(audiofile)

        # 2. Transcribe each segment (i.e., convert speech to text)
        transcripts = [
            self.transcriber.transcribe(segment) for segment in segments
        ]

        # 3. Translate each transcript into a target language (text to text)
        translations = [self.translator.translate(t) for t in transcripts]

        # 4. Finally, generate subtitle files in configured formats.
        subtitles_to_create = [('de', translations)]
        if keep_original:
            subtitles_to_create.append(('en', transcripts))

        subtitles = [
            subtitle_format.compile(subtitle, language)
            for language, subtitle in subtitles_to_create
            for subtitle_format in self.subtitle_formats

        ]

        return subtitles
