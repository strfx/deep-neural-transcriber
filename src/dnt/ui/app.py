"""
Simple Web UI to serve the Deep Neural Transcriber MVP to users.
"""
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Union

from flask import Flask, render_template, request, send_from_directory
from flask.scaffold import F
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from dnt.cli import process
from dnt.ui.validation import Invalid, Valid, validate_into
from dnt.utils import detect_runtime, first, list_models

# Store uploaded videos and the generated subtitles in this directory.
# Caution: The directory is accessible by the user.
UPLOAD_FOLDER = Path("uploads")
# Limit the file types that can be uploaded
# Caution: This -only- checks the file's extension and not if the file is
# actually in that format.
ALLOWED_EXTENSIONS = {'mp4'}
# Path to where the models are stored. Required to locate the different models a
# user can select for transcription.
MODELS_PATH = Path("models/")
# We do not support providing a custom language model at the moment,
# therefore we always use Mozilla's pre-trained language model:
DEFAULT_LANGUAGE_MODEL = Path(
    'models/pretrained-v0.9.3/deepspeech-0.9.3-models.scorer'
)
# Detect whether we are running the deepspeech vanilla package or
# deepspeech-tflite.
RUNTIME = detect_runtime(MODELS_PATH)

if not DEFAULT_LANGUAGE_MODEL.is_file():
    raise RuntimeError(
        f"Can not find configured default language model at: {DEFAULT_LANGUAGE_MODEL}",
        "Please make sure that the language model (*.scorer file) is present."
    )


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER.absolute())
# Disable caching, otherwise you'll end up downloading older versions of some
# files (e.g., after you've transcribed the video with a different model).
# Disabling caching ensures that you'll always get the newest version.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@dataclass
class Submission:
    """
    Capture a video uploaded by the user for transcription.
    """

    # Uploaded video file.
    video: FileStorage
    # Selected (acoustic) model.
    model: Path
    # The location of the video on disk - will be set after calling save()
    # successfully.
    video_path: Path = field(init=False)

    def save(self, folder: Path):
        """
        Save the submitted video file to the filesystem.
        """
        if not self.video.filename:
            raise ValueError("No video found!")

        filename = secure_filename(self.video.filename)
        video_path = folder / filename
        self.video.save(video_path)
        self.video_path = video_path


def downloadable(target: Path):
    """
    Assembles the path to download `target` from.
    """
    return str(UPLOAD_FOLDER / target.name)


def validate_video_file(video: FileStorage) -> Union[Valid, Invalid]:
    if video.filename == '' or not video.filename:
        # Even though the user did not submit a video file, the browser might
        # just send and empty part without filename.
        # c.f. https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
        return Invalid(["No video file provided!"])

    extension = video.filename.rsplit('.', 1)[1].lower()
    if extension in ALLOWED_EXTENSIONS:
        return Valid(video)

    return Invalid(["Video file must have a valid extension!"])


def validate_model(name: str, available_models: Set) -> Union[Valid, Invalid]:
    if not isinstance(name, str) or name.strip() == "":
        return Invalid(["No model selected!"])

    # Locate the model on disk using the name as we don't want to trust a path
    # provided by the user.
    model_path = first(
        model['path'] for model in available_models if model['name'] == name
    )

    if isinstance(model_path, Path) and model_path.is_file():
        return Valid({'name': name, 'path': model_path})

    return Invalid(["Unable to find selected model on filesystem."])


@app.route('/')
def index(errors=[]):
    available_models = [model['name']
                        for model in list_models(MODELS_PATH, RUNTIME)]

    return render_template("index.html", available_models=available_models, errors=errors)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Generate subtitles for a user-provided video.
    """
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    available_models = list_models(MODELS_PATH, RUNTIME)

    val = validate_into(
        Submission,
        validate_video_file(request.files.get('video')),
        validate_model(request.form.get('model'), available_models)
    )

    if isinstance(val, Invalid):
        return index(errors=val.value)
    else:
        submission = val.value

    submission.save(UPLOAD_FOLDER)

    # We'll use the deep-neural-transcribers CLI to transcribe the video.
    # Therefore, prepare the docopt argument format here.
    arguments = {
        '<video_file>': submission.video_path,
        '--model': str(submission.model['path']),
        '--scorer': str(DEFAULT_LANGUAGE_MODEL),
        '--output': str(UPLOAD_FOLDER.absolute())
    }

    start = time.time()

    subtitle_files = process(arguments)

    end = time.time()
    duration = (end - start)

    context = {
        "video": downloadable(submission.video_path),
        "model": submission.model['name'],
        "duration": f"{duration: .4}"
    }

    # Assemble the context to deliver the subtitles
    for subtitles, subtitle_file in subtitle_files:
        if not subtitles.format in context:
            context[subtitles.format] = defaultdict(str)

        context[subtitles.format][subtitles.language_code] = downloadable(
            subtitle_file)

    return render_template("result.html", **context)


@app.route("/sysinfo")
def sysinfo():
    """
    Display some information about the system.
    """
    return render_template("info.html", runtime=RUNTIME)


@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename: str):
    """
    Download a file from the UPLOAD_FOLDER.
    """
    directory = str(UPLOAD_FOLDER.absolute())
    return send_from_directory(directory=directory, path=filename)


if __name__ == "__main__":
    app.run('0.0.0.0', port=8080)
