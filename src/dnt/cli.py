"""Deep Neural Transcriber MVP v1.0

Usage:
    deep-neural-transcriber --version
    deep-neural-transcriber prepare <dataset> <partition> <output_directory>
    deep-neural-transcriber process <video_file> --model=<model_path> --scorer=<scorer_path> [--output=<output_path>]
    deep-neural-transcriber web [--host=<listen_addr>] [--port=<port>]


Options:
    -h --help     Show this screen.
    --version     Show version.

"""
import os
import csv
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import deepspeech

from docopt import docopt
from tqdm import tqdm

from dnt.core import Pipeline
from dnt.datasets.europarl import EuroparlST
from dnt.preprocessing import (IntervalSegmenter, extract_audio, normalize,
                               segment_audio)
from dnt.subtitles import SRT, VTT, Subtitles
from dnt.transcription import DeepSpeechTranscriber
from dnt.translation import DeepL, NopTranslator


def process(arguments) -> List[Tuple[Subtitles, Path]]:
    """
    Generate subtitles for a video.

    process can also be used programmatically. After processing the input
    video, process returns a list containing information about the generated
    file paths. This feature is used by the Web UI for offering subtitles
    files for download.
    """
    videofile = Path(arguments['<video_file>'])

    if arguments['--output']:
        outputdir = Path(arguments['--output'])
    else:
        outputdir = videofile.parent

    model_path = Path(arguments['--model'])
    scorer_path = Path(arguments['--scorer'])

    deepl_api_key = os.environ.get('DEEPL_API_KEY', None)
    if not deepl_api_key:
        # Abort if no DeepL API key could be retrieved
        # Note: If you don't have a DeepL API key, you can also skip translation
        # by using the NopTranslator.
        pass
        # raise RuntimeError("No API key found in DEEPL_API_KEY env variable!")

    pipeline = Pipeline(
        IntervalSegmenter(),
        DeepSpeechTranscriber(model_path, scorer_path),
        NopTranslator(),
        [VTT(), SRT()]
    )
    # DeepL(deepl_api_key),

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Use a temporary file to store the wav file content
        # during transcription.

        # Note: TemporaryFile creates and *opens* a temporary file.
        # Under windows, when opening a temporary file will cause
        # a permission denied error, since the file is already open.
        # Therefore, we have to work around that problem by manually
        # creating a tempfile.
        wavfile = Path(tmpdirname) / 'temporary.wav'
        extract_audio(videofile, wavfile)
        subtitles = pipeline.process(wavfile)

    end = time.time()

    # Keep track of generated subtitle files to return when used
    # programmatically.
    subtitle_files = []

    for subtitle in subtitles:
        subtitle_file = outputdir / \
            f"{videofile.name}.{subtitle.language_code}.{subtitle.format}"

        subtitle_file.write_text(subtitle.content, encoding="utf-8")

        subtitle_files.append((subtitle, subtitle_file))

        print(
            "* Created subtitle file:",
            f"language={subtitle.language_code}, format={subtitle.format}",
            f"filename={str(subtitle_file)}"
        )

    duration = (end - start)
    print("Duration:", duration)

    return subtitle_files


def prepare(arguments):
    """
    Prepare dataset for training.
    """
    dataset = Path(arguments['<dataset>'])
    partition = arguments['<partition>']
    destination = Path(arguments['<output_directory>']) / partition

    # Create partition directory in the output directory, e.g. "some-directory/train"
    destination.mkdir(parents=True, exist_ok=True)

    # Set up metadata csv
    labels = ['wav_filename', 'wav_filesize', 'transcript']
    indexfile = (destination / partition).with_suffix('.csv')

    # Initialize the dataset partition from disk
    europarl = EuroparlST(dataset, "en", "de", partition)

    # Keep track of some stats
    stats = defaultdict(int)

    with open(indexfile, "w", encoding="utf-8") as fd:
        writer = csv.writer(fd, quoting=csv.QUOTE_NONE, escapechar='')
        writer.writerow(labels)

        # Loop over all segments and split the audio file on the fly using
        # ffmpeg.
        for index, segment in enumerate(tqdm(
            europarl.get_segments(),
            total=europarl.number_of_segments
        )):
            sample, time_start, time_end, transcript = segment

            # We only want samples between 10s and 20s
            duration = time_end - time_start

            if duration > 20.0 or duration < 10.0:
                # Discard segments that are too long or too short, according
                # to the deepspeech docs.
                stats['discarded'] += 1
                continue

            outfile = destination / f"{sample.name}-segment{index}.wav"
            segment_audio(sample, outfile, time_start, time_end)

            # Get size of the freshly produced segment file.
            size = outfile.stat().st_size

            # Normalize the transcript acc. to DeepSpeech alphabet.
            transcript = normalize(transcript)

            writer.writerow([outfile.name, size, transcript])

    print(stats)


def main():
    arguments = docopt(__doc__, version="Deep Neural Transcriber MVP v1.0")

    if arguments['prepare']:
        prepare(arguments)

    if arguments['process']:
        process(arguments)


if __name__ == "__main__":
    main()
