"""
This module contains common helpers and utils that are (encouraged to be) used
throughout the project.
"""
from pathlib import Path
from typing import Any, List, Literal

import deepspeech


def first(iterable, default=None) -> Any:
    """
    Return first item in `iterable` a default value.
    """
    return next(iter(iterable), default)


def lines(fname: Path) -> List[str]:
    """
    Returns all lines in given file.

    Note:
        This method reads the whole file into memory at once. If you work with
        big files, this might be a bad idea. Use generators for big files.

    """
    return fname.read_text(encoding="utf-8").splitlines()


def list_models(models_path: Path, supported_format: str):
    """
    Return a list of available models.

    Args:
        models_path: Directory which contains the models.
        supported_format: String indicating the supported model format
            (i.e., pbmm or tflite).

    Example:
        Given the directory structure:
        models_dir/
            my-awesome-model/
                model.pb
                model.pbmm
                model.tflite
            another-cool-model/
                cool-model.pbmm

        >>> list_models(Path("models/"), "tflite")
        [
            {"name": "my-awesome-model (tflite)", "path": Path("models/my-awesome-model/model.tflite")}
        ]

    Note:
        Currently only works for acoustic models. The scorer (LM) is the same
        for all models right now.

    """
    models = []
    immediate_subdirs = [d for d in models_path.iterdir() if d.is_dir()]

    for subdir in immediate_subdirs:
        model_path = first(subdir.glob('*.' + supported_format))
        model = {
            "name": f"{subdir.name} ({supported_format})",
            "path": model_path
        }
        models.append(model)

    return models


# DeepSpeech's model can be exported in multiple formats; Therefore, you must
# use the correct (deepspeech) runtime. 'deepspeech' supports the *.pb and
# *.pbmm models, 'deepspeech-tflite' the TensorFlow Lite format.
Runtime = Literal['pbmm', 'tflite']


def detect_runtime(models_dir: Path) -> Runtime:
    """
    Detect installed deepspeech runtime.

    Availability depends on the runtime, i.e. deepspeech (pbmm) or deepspeech-tflite
    Unfortunately, there is no straightforward way to tell which version is installed
    as both names install the same package. However, we can find out by try and error.

    Args:
        models_dir: Base directory that contains the DeepSpeech models.

    Note:
        When runtime is deepspeech-tflite, this method causes DeepSpeech to print an
        Error: "ERROR: Model provided has model identifier '^←6╝', should be 'TFL3'".
        This error can be ignored.

    Returns:
        A string that identifies the installed runtime.

    """
    some_pbmm_file = first(models_dir.glob("**/*.pbmm"))

    try:
        deepspeech.Model(str(some_pbmm_file))
        return 'pbmm'
    except (RuntimeError, TypeError):
        return 'tflite'


def listify(obj):
    """
    Ensure that `obj` is a list.
    """
    return [obj] if not isinstance(obj, list) else obj
