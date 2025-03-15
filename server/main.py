from typing import Union

from fastapi import FastAPI
from shamba.model.main import run

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/dry-run")
def read_item(project_name: str):
    return run(project_name)
