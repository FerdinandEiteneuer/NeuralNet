import os

import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="NumpyNeuralNet",
    version="1.0",
    author="Ferdinand Eiteneuer",
    author_email="",
    description="ANN library with a keras like API, implemented using numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FerdinandEiteneuer/NeuralNet",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
)
