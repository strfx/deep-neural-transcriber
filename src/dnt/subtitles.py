"""
Module generates subtitles in multiple file formats: SRT, VTT.
"""
from dataclasses import dataclass
from textwrap import dedent
from datetime import timedelta
from typing import List, Literal, Optional, Tuple


@dataclass
class Subtitles:
    format: str
    language_code: str
    content: str


class SubtitleFormat:
    """
    Subtitle format configuration.

    Attributes:
        header: Optional file header
        name: Subtitle format name (will be used as suffix)
        timecode_format: Format string (using %-syntax) to represent
            the format's timecodes.

    """
    header: str
    name: str
    timecode_format: str

    @staticmethod
    def from_suffix(suffix):
        if suffix == "vtt":
            return VTT()
        elif suffix == "srt":
            return SRT()
        else:
            return None

    def compile(self, texts: List[str], language_code: str) -> Subtitles:
        """
        Compile a list of strings into subtitles of specified format.
        """
        subtitles = []

        if self.header:
            subtitles.append(self.header)

        for index, text in enumerate(texts):
            start, end = timecodes(index, self.timecode_format)
            line = dedent(f"""\
                {index + 1}
                {start} --> {end}
                {text}
            """)

            subtitles.append(line)

        return Subtitles(format=self.name, content="\n".join(subtitles), language_code=language_code)


class VTT(SubtitleFormat):
    # WebVTT requires a special header at the beginning of the file.
    # Note: The space after WEBVTT is intentional, it requires at least
    # one space.
    header = "WEBVTT \n"
    name = "vtt"
    timecode_format = "%02d:%02d:%02d.%03d"


class SRT(SubtitleFormat):
    header = ""
    name = "srt"
    timecode_format = "%02d:%02d:%02d,%03d"


def timecodes(offset: int, formatstr: str, interval: int = 10) -> List[str]:
    """
    Generate timecodes based on an interval.

    Args:
        offset: Denotes the intervals position (i.e., the n-th interval)
        formatstr: Format string (%-syntax) for time codes
        interval: Interval's length in seconds

    Returns:
        Formatted timecodes for the interval's start and end.

    """
    codes = []

    start = timedelta(seconds=offset * interval)
    end = timedelta(seconds=offset * interval + interval)

    for t in [start, end]:
        hrs, secs_remainder = divmod(t.seconds, 60 * 60)
        hrs += t.days * 24
        mins, secs = divmod(secs_remainder, 60)
        msecs = t.microseconds // 1000
        timecode = formatstr % (hrs, mins, secs, msecs)
        codes.append(timecode)

    return codes
