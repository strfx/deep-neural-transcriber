# Makefile for Deep Neural Transcriber
#
# Collects various commands for working with the codebase. Should make your
# life easier by not having to remember the commands.
#
# Important: Run this Makefile always in a virtualenv!
#
init:
	pip install --editable .
	pip install --upgrade -r requirements.txt

	mkdir -p models/pretrained-v0.9.3/
	wget -nc https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite -P models/pretrained-v0.9.3/
	wget -nc https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer -P models/pretrained-v0.9.3/


update-deps:
	pip install --upgrade pip-tools pip setuptools
	pip-compile

update: update-deps init

tests:
	pytest -v -x tests/

lint:
	mypy src/ --ignore-missing-imports

devserver: export FLASK_ENV = development
devserver:
	python -m dnt.ui.app

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .mypy_cache

train:
	docker-compose -f docker-compose-train.yml up

evaluate:
	docker-compose -f docker-compose-eval.yml up

.PHONY: tests clean train evaluate devserver init update-deps update lint
