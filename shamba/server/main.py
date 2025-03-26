from typing import Union, Dict

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
import model.main as main

app = FastAPI()

app.mount("/main", StaticFiles(directory="static", html=True), name="static")

@app.get("/test")
def read_root():
    return "Hello World"

@app.post("/dry-run")
def dry_run(data = Body(..., embed=True)):
    return main.run(data)
