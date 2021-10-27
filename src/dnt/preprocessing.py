"""
Module pre-processes video input into audio segments.
"""
import re
import subprocess
from math import ceil
from pathlib import Path

import pydub
from num2words import num2words


# All lowercase, English letters and apostrophe:
# See https://github.com/mozilla/DeepSpeech/blob/master/data/alphabet.txt
DEEPSPEECH_ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def normalize(transcript: str, alphabet: str = DEEPSPEECH_ALPHABET) -> str:
    """
    Normalize input transcript.

    DeepSpeech uses a simple alphabet consisting of the 26 English letters and some
    special characters (see DEEPSPEECH_ALPHABET). This function normalizes the
    input transcript to comply with this alphabet.

    Args:
        transcript: Transcript to normalize / clean.
        alphabet (optional): Alphabet to use.

    Note:
        This function has been mostly created to normalize the Europarl-ST input
        data. Therefore, it contains some specific transformations. However, feel
        free to adapt it to your needs.

    Returns:
        A normalized transcript. The returned transcript is guaranteed to only
        contain characters that are defined in the provided alphabet (+ whitespace).

    """
    transcript = transcript.lower()

    # Remove apostrophes used as quotes
    transcript = re.sub(r"(‘.(?P<inner>\w+).’)",
                        lambda m: m.group('inner'), transcript)

    # Normalize weird apostrophs (in the data, it has whitespaces around too)
    transcript = transcript.replace(" ’ ", "'")

    # Sometimes, bindestrichs have space arounds
    transcript = transcript.replace(" - ", " ")
    transcript = transcript.replace(" – ", " ")

    transcript = transcript.replace("&", "and")

    # Replace umlauts with their spoken counterparts
    transcript = transcript.replace("ä", "a")
    transcript = transcript.replace("ö", "o")
    transcript = transcript.replace("ü", "u")

    # Replace bindestrich with a space
    transcript = transcript.replace("-", " ")

    # Remove double spaces
    transcript = transcript.replace("  ", " ")

    # Replace 4 digits with year
    # quick manual analysis showed that all 4pairs are years (mostly)
    transcript = re.sub(
        '([0-9]{4})', lambda m: num2words(m.group(), to='year'), transcript)

    # Replace numbers with spoken words (cardinal)
    transcript = re.sub(r'(\d+)', lambda m: num2words(m.group()), transcript)

    include = set(alphabet)

    if " " not in include:
        # Ensure whitespace is included, altough not in the alphabet usually.
        include.add(" ")

    return ''.join(ch for ch in transcript if ch in include)


def extract_audio(video: Path, outfile: Path, channels: int = 1, sample_rate: int = 16_000) -> Path:
    """
    Extract audio track from an mp4 video as wav.

    Args:
        video: Path to the source video file
        channels: Number of channels to extract (1 = mono)
        sample_rate: Sample rate of the resulting audio.

    Returns:
       Path of the resulting audio file.

    """
    args = [
        'ffmpeg',
        '-y',
        '-i',
        str(video.absolute()),
        '-ac',
        str(channels),
        '-ar',
        str(sample_rate),
        str(outfile.absolute())
    ]

    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return outfile


def segment_audio(
        audiofile: Path, outfile: Path, start: float, end: float, sample_rate=16000
):
    """
    Extract a segment from an audio file.

    A segment is a snippet of the provided audio file with a start time
    (offset from beginning) and an end. The destination file's length
    is end - start.

    Args:
        audiofile: Audio file's path
        outfile: Desired path to the output file
        start: Beginning of the segment (in seconds)
        end: End of the segment (in seconds)
        sample_rate: Optionally change the sample rate of the output file.

    """
    # Tip: When providing -ss (segment start) before -i, the segmentation process
    # is faster because ffmpeg skips directly to the segment's start position.
    ffmpeg_commands = [
        'ffmpeg',
        '-y',                               # say yes to all prompts
        '-ss', str(start),                  # start of segment
        '-i', str(audiofile.absolute()),    # input file path
        '-to', str(end),                    # end of segment
        '-copyts',                          # make timestamps correct
        '-ac', '1',                         # force mono Channel
        '-ar', str(sample_rate),            # resample to given sample rate
        str(outfile.absolute())             # write to desired output path.
    ]

    subprocess.run(ffmpeg_commands, shell=True, check=True,
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)  # type: ignore


class IntervalSegmenter:

    def __init__(self, interval=10_000):
        self.interval = interval

    def segment(self, audiofile: Path):
        """
        Segment an audio at a regular interval.

        Args:
            audio: Audio to segment (pydub.AudioSegment)
            interval: Interval to split audio at (in ms)

        Returns:
            A list of audio segments.
        """
        audio = pydub.AudioSegment.from_wav(str(audiofile))
        number_of_segments = ceil(len(audio) / self.interval)

        return [
            audio[i * self.interval: i * self.interval + self.interval]
            for i in range(0, number_of_segments)
        ]
