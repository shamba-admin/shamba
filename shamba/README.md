## Getting Started

In v1.2, SHAMBA is only available as a command-line version. A web application is targeted for future versions. You may find some proof-of-concept files in this repo, but it is not set up for full use yet. The `shamba_command_line.py` script will prompt you for the information it needs to run the model. It will also print out instructions on how to prepare input data to run the model.

Setup can either use a Docker container, or locally installed packages.


### 1. Docker Setup

1. Ensure that Docker is installed on your system. Check the [official documentation](https://docs.docker.com/get-docker/) for how to install Docker for your specific operating system.
2. From the `shamba` directory, run the following command:
```sh
docker compose up
```

#### Command-line version

To use the terminal interface from within the Docker container, follow these steps:

1. Find the container ID of the running container: `docker ps`. Grab the ID of the `shamba-server` container.
2. Run the following command: `docker exec -it <container-id> bash`
3. Run `cd ..` to go back to the `shamba` directory.
4. Run `poetry run python shamba_command_line.py` to run the model.


### 2. Local Setup

To set up without Docker, follow these instructions:

1. Install `pyenv` which is an easy-to-use tool to manage multiple Python versions. Check the [official documentation](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) for how to install `pyenv` for your specific operating system.
2. Install Python `3.10`: `pyenv install 3.10` then `pyenv local 3.10`.
3. Install `poetry` by following the [official documentation](https://python-poetry.org/docs/).
4. Go to the `shamba` directory and install dependencies:
```sh
cd shamba
poetry install
poetry env use 3.10
eval $(poetry env activate)
```

#### Command-line version

The SHAMBA model can be run from the command line using the `shamba_command_line.py` script.

```sh
python shamba_command_line.py
```

### Format Files
```sh
poetry run black .

```

### Run Tests
```sh
poetry run pytest
```