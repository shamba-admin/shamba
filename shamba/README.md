## Getting Started

### Local Installation

1. Install `gdal`. For MacOS, use `brew install gdal`. The currently supported version of `gdal` is `3.11.3`
2. Please unzip the soil data file `./rasters/soil/hwsd.7z`.
3. Ensure you have the correct version of Python. `pyenv` is an easy-to-use tool to manage multiple Python versions: e.g. `pyenv install 3.10` then `pyenv local 3.10`.
4. Install `poetry` using `brew install poetry`.
5. Go to the `shamba` directory and install dependencies:
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

The SHAMBA model can be run as a client-server application using the `shamba/server` and `shamba/client` directories. The server is a FastAPI application that runs the model and returns the results. The client is a Preact application that is built using [Esbuild](https://esbuild.github.io/). The client is used to display the results of the model in a web browser.

#### Run the Client

See [the client README.md](shamba/client/README.md) for instructions on how to run the client.

#### Run the Server

From the `./server` directory, run the server in _development_ mode: `poetry run fastapi dev main.py`


### Format Files
```sh
poetry run black .

```

### Run Tests
```sh
poetry run pytest
```
