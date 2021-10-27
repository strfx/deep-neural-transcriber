<h1 align="center">
  Deep Neural Transcriber 🧠🖊
  <br>
</h1>

<h3 align="center">Automatically generate subtitles for recorded university lectures.</h3>

<p align="center">  
  <a href="https://github.com/strfx/deep-neural-transcriber/actions" target="_blank">
    <img src="https://img.shields.io/github/workflow/status/strfx/deep-neural-transcriber/build" />
  </a>
  <a href="https://github.com/strfx/deep-neural-transcriber/blob/main/LICENSE" target="_blank">
     <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
  </a>
  <a href="http://mypy-lang.org/" target="_blank">
     <img src="http://www.mypy-lang.org/static/mypy_badge.svg" />
  </a>
</p>
<p align="center">
  <img src="https://github.com/strfx/deep-neural-transcriber/blob/main/docs/screenshot-generic.png?raw=true" alt="deep-neural-transcriber CLI"/>
</p>


# Table of Contents

* [About](#about)
* [Getting Started](#getting-started)
   * [Run](#run)
* [Developing](#developing)
   * [Run tests](#run-tests)
   * [Fine-tune DeepSpeech models](#fine-tune-deepspeech-models)
   * [Tip: Set up Jupyter Notebook](#tip-set-up-jupyter-notebook)
* [Contributing](#contributing)
* [Troubleshooting](#troubleshooting)

# About

The Deep Neural Transcriber generates subtitles for recorded university lectures. I developed this project as my bachelor thesis at Hochschule Luzern. Find a (very shortened) version of the abstract below:

> Most universities closed their facilities during the COVID-19 pandemic and switched to distance learning formats. Many lecturers record their courses and distribute the videos to the students afterward. Subtitles provided with the videos could further improve the accessibility and ultimately, enhance the students’ experience with distance learning material. However, manually transcribing and translating lectures is a tremendous effort: A professional human transcriber requires between four to ten hours to transcribe a single hour of audio.
>
> The Deep Neural Transcriber automatically generates subtitles for lecture videos. To tackle that problem, the Deep Neural Transcriber first leverages automated speech recognition (ASR), using Mozilla DeepSpeech, to transcribe the audio. In a second step, the transcript is translated into a target language using DeepL. This approach follows the cascade architecture approach in the spoken language translation (SLT) problem domain. Finally, the Deep Neural Transcriber generates subtitle files in various formats using the transcripts.
>
> With the Deep Neural Transcriber, we present a functioning, end-to-end pipeline to generate subtitles for lecture recordings automatically. Even with human post-processing, the system reduces the transcription time drastically, making it feasible for lecturers to produce subtitles for their lecture videos. The system is modular, i.e., each component of the cascade architecture can be replaced or improved independently. For example, one could experiment with replacing DeepSpeech with Facebook's wav2vec approach to improve transcription quality.

# Getting Started

To run the Deep Neural Transcriber, make sure you have these tools avalaible on your system:
  * `ffmpeg`
  * `sox`

Next, you have to obtain trained models. If you don't have custom models, you can use the pre-trained DeepSpeech models, see  [latest DeepSpeech release](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3). For each model (i.e., acoustic model and language model), create a separate directory under `models/`. The pre-trained models will be automatically downloaded when running `make init`.

Optionally: If you need the EuroparlST dataset, fetch it from [here](https://www.mllp.upv.es/europarl-st/) and extract it into the `dataset/` directory.

## Run

You can run the project using Docker. Therefore, you first have to build the image:
```sh
$ docker build -t deep-neural-transcriber:1.0 .
```

Run the container using:
```sh
$ docker run -e DEEPL_API_KEY=<your api key> -p 8080:8080 -t deep-neural-transcriber:1.0
```

# Developing

To start developing, install the dependencies in a virtual environment:
```sh
# Create and activate virtualenv
$ python -m venv venv && source venv/bin/activate

# Installs the dependencies and downloads Mozilla's pre-trained models.
$ make init
```

The last step also installs the project in `--editable` mode.

During development, you can run a development webserver using:

```sh
$ make devserver
```

## Run tests

To test the project, run (inside virtualenv):

```sh
$ make tests
```

The test suite uses a sample dataset, stored under `tests/data`.

## Fine-tune DeepSpeech models

Depending on your use case, you might want to fine-tune the pre-trained models on a custom dataset. Make sure to check out DeepSpeech's documentation if fine-tuning makes sense. If so, make sure you can run the training on GPUs.

Below you'll find a list of steps required to fine-tune your model: 


1. Format your dataset according to the [DeepSpeech Playbook](https://mozilla.github.io/deepspeech-playbook/DATA_FORMATTING.html)
2. Create a directory named `deepspeech-data` and create following structure:
  ```
  deepspeech-data/
  ├── checkpoints     # Training checkpoints
  │   ├── finetuned   # - will contain the checkpoints during fine-tuning
  │   └── pretrained  # - contains the pre-trained checkpoints
  ├── data            # Dataset to train on (dev/train/test partitions)
  ├── exported-model  # Training process will place the tuned model here
  └── summary         # Training will place summary files here for TensorBoard
  ```

We create this separate directory so we can transfer it easily to a GPU machine.

If everything is set up:
  1. Double-check the paths in `docker-compose-train.yml`
  2. Run `make train` to kick-off the training

## Tip: Set up Jupyter Notebook

If you plan to use Jupyter Notebooks and want to access the installed packages in the virtualenv, run:

```sh
$ pip install ipykernel
$ python -m ipykernel install --name=deep-neural-transcriber-venv
```

Run `jupyter notebook` and navigate to *Kernel* -> *Change Kernel* -> *deep-neural-transcriber-venv*. 

# Contributing

Contributions are welcome! If you plan major changes, please create an issue first to discuss the changes.

The codebase contains some `TODO`s. This is not (only) because I was lazy, but to give some pointers where the codebase could be improved in a future project.

# Troubleshooting

Deep Neural Transcriber depends heavily on the `deepspeech` package. That package has *a lot* of dependencies, which make dependency management a difficult task. We have experimented with the versions and found a working combination. We've placed the exact versions that are known to work in the requirements.txt.

Altough deepspeech 0.9.3 claims to require numpy 1.14.0, it works fine with
newer versions. However, their package claims that version, which means
resolvers like pip-tools will fail.
