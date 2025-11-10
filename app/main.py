from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from app.schemas import AtmosData, JsonPredictRequest, PredictResponse
from app.model_runtime import predict_from_base64, predict_from_bytes

app = FastAPI(title="Landfill Multimodal API", version="1.0.0")

@app.get("/live")
def live():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict_json(req: JsonPredictRequest):
    try:
        prob, label = predict_from_base64(req.image_base64, req.atmos.as_ordered_list())
        return PredictResponse(prob=prob, label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-multipart", response_model=PredictResponse)
async def predict_multipart(
    file: UploadFile = File(..., description="PNG/JPEG image data"),
    NO2: float = Form(...),
    CO: float = Form(...),
    PM2_5: float = Form(..., alias="PM2.5"),
    PM10: float = Form(...)
):
    try:
        img_bytes = await file.read()
        atmos = [NO2, CO, PM2_5, PM10]
        prob, label = predict_from_bytes(img_bytes, atmos)
        return PredictResponse(prob=prob, label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
