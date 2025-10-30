from pydantic import BaseModel, Field, conlist
from typing import Optional

# Atmospheric keys order: [NO2, CO, PM2_5, PM10]
class AtmosData(BaseModel):
    NO2: float = Field(..., description="Nitrogen Dioxide")
    CO: float = Field(..., description="Carbon Monoxide")
    PM2_5: float = Field(..., alias="PM2.5", description="Particulate matter 2.5")
    PM10: float = Field(..., description="Particulate matter 10")

    # convenience to list in the exact order used by the model
    def as_ordered_list(self) -> list[float]:
        return [self.NO2, self.CO, self.PM2_5, self.PM10]

class JsonPredictRequest(BaseModel):
    # Base64-encoded PNG/JPEG
    image_base64: str
    atmos: AtmosData

class PredictResponse(BaseModel):
    prob: float
    label: int
