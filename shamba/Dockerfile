FROM ghcr.io/osgeo/gdal:alpine-normal-latest

# Install minimal build dependencies
RUN apk add --no-cache \
    bash \
    curl \
    build-base

# Install pyenv
RUN curl -fsSL https://pyenv.run | bash

# Add pyenv to PATH
# ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH

# Install Python 3.10 using pyenv
RUN pyenv install 3.10
RUN pyenv global 3.10

# Verify Python version
RUN python --version

# Install pip and pipenv
RUN python -m ensurepip --upgrade && \
    pip install --upgrade pip pipenv

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
RUN pipenv install --system --deploy

# Copy the rest of the application code
# COPY . .
