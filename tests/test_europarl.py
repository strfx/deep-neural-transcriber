"""
Test the handling of the Europarl-ST dataset.

The dataset has a specific format which needs to be unpacked before we can
perform any processing on it. See the module docstring of dnt.datasets.europarl
for a detailed description.

We are using a small fraction of the original dataset, stored under
`tests/data`.
 """
import wave
from pathlib import Path

from dnt.datasets.europarl import EuroparlST, parse_segments_listing
from dnt.preprocessing import segment_audio

import pytest


@pytest.fixture(scope='function')
def europarl():
    """
    Fixture providing a small fraction (dev set) of Europarl-ST data.
    """
    return EuroparlST(Path("tests/data/europarlST-v1.1/"), 'en', 'de', 'dev')


def test_parse_segment_listing():
    """
    Splits a line in segments.lst file into a tuple with appropriate to allow
    calculating the segment's size.
    """
    segment = "en.20080924.31.3-213 0.0 10.77"
    expected = ("en.20080924.31.3-213", 0.0, 10.77)

    assert parse_segments_listing(segment) == expected


def test_initialize_dataset(europarl):
    """
    EuroparlST should load dataset from disk upon initialization.
    """
    assert europarl.number_of_segments == 19


def test_get_training_segment(europarl):
    """
    Should join a segment with the corresponding transcript to create (audio,
    transcript) pairs for training.
    """
    transcript = (
        "Madam President, the President of the European Central Bank,"
        " Jean-Claude Trichet, recently said that, when the market"
        " stabilises, we will not return to business as usual, but"
        " instead we will experience a new normality."
    )

    expected = [(
        Path("tests/data/europarlST-v1.1/en/audios/en.20080924.31.3-243.m4a"),
        0.0, 10.77,
        transcript
    )]

    segments = list(europarl.get_segments(n=1))

    assert segments == expected


def test_segments_are_ordered(europarl):
    """
    Segments must be in order, i.e., successive calls to get_segments with the
    same parameter should always yield the same segments.
    """
    assert list(europarl.get_segments(n=10)) == list(
        europarl.get_segments(n=10))


@pytest.mark.parametrize('to_resolve, expected_path', [
    (
        dict(segments="segments.en"),
        Path("tests/data/europarlST-v1.1/en/de/dev/segments.en")
    ),
    (
        dict(sample="en.20080924.31.3-243"),
        Path("tests/data/europarlST-v1.1/en/audios/en.20080924.31.3-243.m4a")
    )
])
def test_resolve_samples_and_segments(europarl, to_resolve, expected_path):
    """
    resolve() should return the path of a segment or sample on the filesystem,
    so that the Europarl module hides the dataset's directory layout.
    """
    assert europarl.resolve(**to_resolve) == expected_path


def test_get_pairs(europarl):
    """
    Dataset offers triplets of (audio, text, text). For translation tasks, we
    only want the (text, text) samples.
    """
    expected = [
        (
            "Madam President, the President of the European Central Bank, Jean-Claude Trichet, recently said that, when the market stabilises, we will not return to business as usual, but instead we will experience a new normality.",
            "Frau Präsidentin! Der Präsident der Europäischen Zentralbank Jean-Claude Trichet erklärte kürzlich, dass wir, wenn sich der Markt stabilisiert, nicht wie gewohnt weitermachen, sondern eine neue Normalität erleben werden."
        ),
        (
            "Given the failings and weaknesses in the market and institutions that have been brought to light in devastating fashion over the last year, a move away from the abuses and faults of the past is only to be welcomed.",
            "Angesichts des Versagens und der Schwachstellen des Marktes und der Institutionen, die im letzten Jahr auf zerstörerische Art und Weise ans Tageslicht gelangt sind, kann eine Verlagerung weg von den Missbräuchen und Fehlern der Vergangenheit nur begrüßt werden."
        ),
        (
            "The financial crisis has caused terrible panic, but it has also served to emphasise the need to eliminate obscurities and to introduce transparency, and for we legislators to regulate.",
            "Die Finanzkrise hat zu einer schrecklichen Panik geführt, aber sie hat auch die Notwendigkeit betont, Unklarheiten zu beseitigen und Transparenz einzuführen, und für uns als Gesetzgeber, zu regulieren."
        )
    ]

    some_pairs = europarl.pairs(n=3)

    assert list(some_pairs) == expected


@pytest.mark.integration
def test_split_segment(europarl, tmp_path):
    """
    Simple integration tests that ensures audio tracks are splitted correctly
    according to the parsed segments.

    NOTE: These tests require you to have the system dependencies stated in the
    README installed. Provide --skip-integration to pytest to skip this test.
    """
    sample = "en.20080924.31.3-243"
    filename = europarl.resolve(sample=sample)

    segments = europarl.get_segments_of_sample("en.20080924.31.3-243")

    for index, segment in enumerate(segments):
        _, time_start, time_end = segment
        outfile = tmp_path / f'segment-{index}.wav'

        segment_audio(filename, outfile, time_start, time_end)

    wav_files = [f for f in tmp_path.iterdir() if f.is_file()]

    assert len(wav_files) == len(segments)

    for index, segment in enumerate(segments):
        _, time_start, time_end = segment
        expected_duration = time_end - time_start

        segment_file = tmp_path / f'segment-{index}.wav'

        with wave.open(str(segment_file), 'r') as fd:
            frames = fd.getnframes()
            rate = fd.getframerate()
            channels = fd.getnchannels()
            duration = frames / float(rate)

        # Audio's duration should match the segments interval, allowing +/- 0.1
        # seconds.
        assert expected_duration == pytest.approx(duration, 0.1)

        # Audio must be 16 kHz to comply with DeepSpeechs requirements.
        assert rate == 16000

        # Audio must be mono - also a DeepSpeech requirement.
        assert channels == 1
