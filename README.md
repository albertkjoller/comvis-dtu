# 👁️ Computer Vision 👁️
The package contains functionalities for the course 02504 - Computer Vision (Spring 2023) @ DTU.
If you like what you see, give it a star! ⭐

A cheatsheet containing examples of the usage of functions (and solutions to specific exercises) is included in the root, both as a notebook and as a searchable HTML file. Be aware that file paths are relative to the system on which the cheatsheet was created - it should be fairly easy to insert your own paths, though.

## Installation

You can directly install the package using `pip` by running:

Either in editable mode:
```
pip install -e git+https://github.com/albertkjoller/comvis-dtu.git#egg=comvis-dtu
```

or in deployment mode:
```
pip install git+https://github.com/albertkjoller/comvis-dtu.git
```

## Development setup
### Environment

Create a conda environment, e.g. like below with Python 3.10:
```
conda create -n comvis-dtu python=3.10
conda activate comvis-dtu
```

Install local package in editable mode:
```
pip install -e .
```

### Pre-commit hooks

Additionally, install the `pre-commit` module for ensuring a general code structure in the remote repository.
```
pip install pre-commit
```

Install `pre-commit` on the repository with the settings specified in `.pre-commit-config.yaml`.
```
pre-commit install
```
