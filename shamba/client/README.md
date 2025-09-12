# SHAMBA Client

# Web Application Overview: PROOF OF CONCEPT ONLY, DO NOT USE FOR V1.2 MODEL RUNS

The SHAMBA model can be run as a client-server application using the `shamba/server` and `shamba/client` directories. The server is a _FastAPI_ application that runs the model and returns the results. The client is a [Preact](https://preactjs.com/) application that is built using [Esbuild](https://esbuild.github.io/). The client is used to display the results of the model in a web browser.

- The SHAMBA client is a [Preact](https://preactjs.com/) application that is built using [Esbuild](https://esbuild.github.io/).
- The project uses [Tailwind CSS](https://tailwindcss.com/) for styling.
- [Civet](https://github.com/danielx/civet) is used as the main language for the project.

In a Docker container, you can access the web application at `http://localhost:8000/main/#/`. To build locally, follow the instructions below.

## Prerequisites

- [Node](https://nodejs.org/en). The currently supported version of `node` is `v20.12.2`. We recommend to use [nvm](https://github.com/nvm-sh/nvm) to manage multiple `node` versions.
- [pnpm](https://pnpm.io/installation).

## Produce Static Files

The purpose of the client code is to produce static files, i.e. JavaScript and CSS, that are then injected into the _server_ code. _FastAPI_ then serves these static files to browser clients.

### Installation

Install the dependencies:

```sh
pnpm install
```

### Build Static Assets

[Esbuild](https://esbuild.github.io/) is used to build the static assets of the project.

The artiifacts of building the project will be placed in the `../server/static/js` directory. For more details, check the `esbuild.mjs` file.

Build the static assets:

```sh
pnpm build
pnpm styles
```

For local development, you can _watch_ for changes and rebuild automatically:

```sh
pnpm watch
pnpm styles:watch
```

## Run the Server

After generating the static files, from the `./server` directory, run the server in _development_ mode: `poetry run fastapi dev main.py`

Go to: `http://127.0.0.1:8000/main/#/`







