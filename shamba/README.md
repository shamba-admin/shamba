## Getting Started

### Local Installation

1. Install `gdal`. For MacOS, use `brew install gdal`.
2. Ensure you have the correct version of Python. `pyenv` is an easy-to-use tool to manage multiple Python versions: e.g. `pyenv install 3.10` then `pyenv local 3.10`.
3. Install `poetry` using `brew install poetry`.
4. Go to the `shamba` directory and install dependencies:
```sh
cd shamba
poetry install
poetry env use 3.10
eval $(poetry env activate)
```

### Run Server

From the `./server` directory, run the server in _development_ mode: `poetry run fastapi dev main.py`

### Format Files
```sh
poetry run black .

```

### Run Tests
```sh
poetry run pytest
```