# Small-Holder Agriculture Mitigation Benefit Assessment (SHAMBA) model, version 1.2

Instructions on how to setup and use SHAMBA are below.

The docs folder contains the detailed scientific description of the model and a user guide. For documentation on how to implement the Plan Vivo approved methodology, see the Plan Vivo website at: https://www.planvivo.org/pt001 

See the instructions in the `shamba_command_line.py` file for details on how to run SHAMBA with the data-input-templates. The script is already set up to run the example input from the `example_SHAMBA_input_output_uganda_tech_spec.xlsx` file.

For help, contact shamba@ed.ac.uk

## Getting Started with SHAMBA

In v1.2, SHAMBA is only available as a command-line version. The `shamba_command_line.py` script will prompt you for the information it needs to run the model. It will also print out instructions on how to prepare input data to run the model.

A web application is targeted for future versions. You may find some proof-of-concept files in this repo, but this is not set up for full use yet and should be ignored.

Command-line setup can either use a Docker container, or locally installed packages.


### 1. Docker Setup

1. Ensure that Docker is installed on your system. Check the [official documentation](https://docs.docker.com/get-docker/) for how to install Docker for your specific operating system.
2. From the `shamba` directory, run the following command:
```sh
docker compose up
```

To use SHAMBA from within the Docker container, follow these steps:

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

The SHAMBA model can now be run from the command line using the `shamba_command_line.py` script:

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
