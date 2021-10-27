"""
Unit tests for the subtitle generation.
"""
from textwrap import dedent

import pytest

from dnt.subtitles import SRT, VTT, SubtitleFormat, Subtitles


def test_from_suffix_constructor():
    assert isinstance(SubtitleFormat.from_suffix('vtt'), VTT)
    assert isinstance(SubtitleFormat.from_suffix('srt'), SRT)

    assert not SubtitleFormat.from_suffix('i-dont-exist')


@pytest.mark.parametrize('subtitle_format, expected', [
    (
        'vtt',
        Subtitles(
            format='vtt',
            language_code='en',
            content=dedent("""\
            WEBVTT 

            1
            00:00:00.000 --> 00:00:10.000
            I am the first subtitle line

            2
            00:00:10.000 --> 00:00:20.000
            I am the second subtitle line.
            """)
        )
    ),
    (
        'srt',
        Subtitles(
            format='srt',
            language_code='en',
            content=dedent("""\
            1
            00:00:00,000 --> 00:00:10,000
            I am the first subtitle line

            2
            00:00:10,000 --> 00:00:20,000
            I am the second subtitle line.
            """)
        )
    )
])
def test_compile_subtitles(subtitle_format, expected):
    """
    compile() should produce valid subtitles in specified format.
    """
    texts = [
        "I am the first subtitle line",
        "I am the second subtitle line."
    ]

    fmt = SubtitleFormat.from_suffix(subtitle_format)
    subtitles = fmt.compile(texts, 'en')

    assert subtitles == expected
