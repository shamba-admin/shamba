from typing import Union

from fastapi import FastAPI, Body
import model.main as main

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/dry-run")
def dry_run(project_name: str = Body(..., embed=True)):
    return main.run(project_name)
