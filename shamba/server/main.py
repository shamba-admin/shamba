from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
import model.main as main

app = FastAPI(debug=True)

app.mount("/main", StaticFiles(directory="static", html=True), name="static")


class SoilEmissions(BaseModel):
    soil_base_emissions: List[float]
    soil_project_emissions: List[float]
    soil_difference: List[float]

    class Config:
        arbitrary_types_allowed = True


@app.post("/dry-run", response_model=List[SoilEmissions])
def dry_run(
    data=Body(..., embed=True), name: str = Body(..., embed=True), use_api: bool = False
):
    return main.run(project_name=name, data=data, use_api=use_api)
