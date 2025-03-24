from typing import Union

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
import model.main as main

app = FastAPI()

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/test")
def read_root():
    return {"Hello": "World"}


@app.post("/dry-run")
def dry_run(project_name: str = Body(..., embed=True)):
    return main.run(project_name)
