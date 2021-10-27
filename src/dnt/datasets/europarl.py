"""
Streamlines working with the Europarl-ST dataset.

Obtain the original dataset from: https://www.mllp.upv.es/europarl-st/. The
website also links to the corresponding paper. Please consult the paper and the
website for more information about the dataset.

Dataset layout
==============

The dataset contains one folder per spoken language (i.e., the language of the
recorded speeches). Each language folder holds the audio recordings in a
`audios/` folder. Besides the audio, these folders also contain the transcript
in the different languages. Each transcript folder is split up into dev / test /
train.

We are mostly interested in the transcripts folder, as they contain the most
important information to work with the dataset. 

The main challenge is to extract the segments from the audio files, as they are
not yet segmented. That means, they contain the whole speech, instead of the
per-sentence segments.

speeches.lst
------------

Contains the list of speeches that contains to this set (i.e., partition). This
is a list of the audio snippets, for example:

```
en.20080924.31.3-243
en.20080924.31.3-246
en.20080924.4.3-018
```

speeches.${source-language}
---------------------------

Transcripts of the speech in its original language. The file places each
transcript on its own line. The line numbers correspond with the line number in
the speeches.lst file. That means, the first filename in speeches.lst
corresponds to the first transcript in this file.

For example:
```
$ head -n3 speeches.en
Madam President, European [...] countries and globally.
Madam President, I do not [...] be congratulated for that.
Mr President, the whole [...] since I retire after this session.
```

speeches.${target-language}
---------------------------

Contains the translated transcript into a target language. Has the same
structure as the speeches.${source-language} file.

For example:
```
$ head -n3 speeches.de
Frau Präsidentin! Der Präsident der [...] europäischen Länder und weltweit sichern.
Frau Präsidentin! Ich teile nicht die EZB [...] dazu beglückwünschen.
Herr Präsident! Das gesamte [...] weil ich nach dieser Sitzung in den Ruhestand gehe.
```

segments.lst
------------

The dataset has been automatically segmented on sentence level. These segments
can be used to train models. A line in the file consists of the three fields
`audio_file`, `start` and `end`, meaning the start and end timestamp of the
segment in a given audio file.

segments.${source-language} / segments.${target-language}
---------------------------------------------------------

Contains the original and translated transcript for each segment. The
transcripts are placed on per line to correspond with the line numbers of
segments.lst

"""
from itertools import islice
from pathlib import Path
from typing import Iterator, Tuple

from dnt.utils import lines


def parse_segments_listing(line):
    parts = line.split()

    if len(parts) != 3:
        return None

    audio_file, start, end = parts[0], float(parts[1]), float(parts[2])
    return (audio_file, start, end)


class EuroparlST:

    def __init__(self, dataset_location: Path, source_language, target_language, partition):
        self.partition = partition
        self.path = dataset_location
        self.src = source_language
        self.tgt = target_language
        self.load()

        self.audio_path = self.path / self.src / "audios"

    def load(self):
        """
        Load dataset partition from filesystem.
        """
        self.partition_path = self.path / self.src / self.tgt / self.partition
        segments_lst = self.partition_path / "segments.lst"
        segments = [
            parse_segments_listing(line)
            for line in segments_lst.read_text(encoding="utf-8").splitlines()
            if line
        ]

        self.segments = segments

    @property
    def number_of_segments(self):
        """
        Returns the total number of segments in this partition.
        """
        return len(self.segments)

    @property
    def number_of_speeches(self):
        """
        Returns the total number of speeches in this partition.
        """
        return len(self.speeches)

    def resolve(self, segments=None, sample=None) -> Path:
        """
        Resolve paths for specific files.
        """
        if segments:
            return self.partition_path / segments

        if sample:
            return self.audio_path / (sample + ".m4a")

        raise ValueError(
            "Nothing provided to resolve.",
            "Either segments or sample must be set!"
        )

    def get_segments_of_sample(self, sample):
        """
        Return all segments by a specific audio sample.
        """
        return sorted([
            segment
            for segment in self.segments
            if segment[0] == sample
        ], key=lambda x: x[1])

    def get_segments(self, n: int = 0):
        """
        An audio sample can have multiple segments. Each segment is a short
        audio clip. During iteration, we want to process this number of samples
        for later training.
        """
        transcripts = lines(self.resolve(segments=f"segments.{self.src}"))

        if n == 0 or n > self.number_of_segments:
            n = self.number_of_segments

        for index, segment in enumerate(self.segments[:n]):
            audio_filename, time_start, time_end = segment
            filename = self.resolve(sample=audio_filename)
            yield (filename, time_start, time_end, transcripts[index])

    def pairs(self, *, n: int = 0) -> Iterator[Tuple[str, str]]:
        """
        Returns an iterator over the first n language pairs (text, text) tuples.

        Args:
            n: A positive integer counting the number of pairs.
               0 by default, returns an iterator over all text pairs.

        Returns:
            An iterator over (sentence in source language, translated sentence)
            tuples.

        """
        if n < 0:
            raise ValueError("n must be a positive integer (>= 0).")

        src = lines(self.resolve(segments="segments." + self.src))
        dst = lines(self.resolve(segments="segments." + self.tgt))

        return islice(zip(src, dst), n)
