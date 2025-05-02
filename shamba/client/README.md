# SHAMBA Client

## Overview

- The SHAMBA client is a [Preact](https://preactjs.com/) application that is built using [Esbuild](https://esbuild.github.io/).
- The project uses [Tailwind CSS](https://tailwindcss.com/) for styling.
- [Civet](https://github.com/danielx/civet) is used as the main language for the project.

## Installation

Ensure that `pnpm` is installed on your system. If not, check the installation instructions [here](https://pnpm.io/installation).

Install the dependencies:

```sh
pnpm install
```

## Build Static Assets

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