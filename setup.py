import setuptools

with open("README.md", "r", encoding="utf-8") as fd:
    long_description = fd.read()

setuptools.setup(
    name="deep-neural-transcriber",  # Replace with your own username
    version="0.0.1",
    author="Claudio Pilotti",
    author_email="claudio.pilotti@bluewin.ch",
    description="Deep Neural Transcriber",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/strfx/deep-neural-transcriber",
    project_urls={
        "Bug Tracker": "https://github.com/strfx/deep-neural-transcriber/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ['deep-neural-transcriber=dnt.cli:main'],
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8.0",
    install_requires=[
        'deepspeech-tflite==0.9.3',
        'num2words',
        'Flask',
        'docopt',
        'tqdm',
        'pydub',
        'requests',
        'wave',
        'toolz',
        'nltk',
    ],
    extras_requires={
        'dev': [
            'pytest',
            'pytest-cov',
            'mypy'
        ]
    }
)
