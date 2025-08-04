## Getting Started

### Local Installation

1. Install `gdal`. The currently supported version of `gdal` is `3.11.3`. Check the [official documentation](https://gdal.org/en/stable/download.html#binaries) for how to install `gdal` for your specific operating system.
2. Please unzip the soil data file `./rasters/soil/hwsd.7z`.
3. Install `pyenv` which is an easy-to-use tool to manage multiple Python versions. Check the [official documentation](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) for how to install `pyenv` for your specific operating system.
4. Install Python `3.10`: `pyenv install 3.10` then `pyenv local 3.10`.
5. Install `poetry` by following the [official documentation](https://python-poetry.org/docs/).
6. Go to the `shamba` directory and install dependencies:
```sh
cd shamba
poetry install
poetry env use 3.10
eval $(poetry env activate)
```

The SHAMBA model can be used in two ways:

### Command-line version

The SHAMBA model can be run from the command line using the `shamba_command_line.py` script.

```sh
python shamba_command_line.py
```

The script will prompt you for the information it needs to run the model. It will also print out instructions on how to prepare input data to run the model.

### UI version

The SHAMBA model can be run as a client-server application using the `shamba/server` and `shamba/client` directories. The server is a _FastAPI_ application that runs the model and returns the results. The client is a [Preact](https://preactjs.com/) application that is built using [Esbuild](https://esbuild.github.io/). The client is used to display the results of the model in a web browser.

#### Produce Static Files

The purpose of the client code is to produce static files, i.e. JavaScript and CSS, that are then injected into the _server_ code. _FastAPI_ then serves these static files to browser clients.

See [the client README.md](shamba/client/README.md) for instructions on how to create the static files.

#### Run the Server

From the `./server` directory, run the server in _development_ mode: `poetry run fastapi dev main.py`

Go to: `http://127.0.0.1:8000/main/#/`


### Format Files
```sh
poetry run black .

```

### Run Tests
```sh
poetry run pytest
```
