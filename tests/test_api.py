import base64
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _fake_png() -> bytes:
    img = Image.new("RGB", (224, 224), (120, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_live():
    r = client.get("/live")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_predict_json():
    b = _fake_png()
    b64 = base64.b64encode(b).decode("utf-8")
    payload = {
        "image_base64": b64,
        "atmos": {"NO2": 10.0, "CO": 0.3, "PM2.5": 25.0, "PM10": 40.0}
    }
    r = client.post("/predict", json=payload)
    # The model must be present; we only validate response shape/types here
    assert r.status_code == 200
    data = r.json()
    assert "prob" in data and "label" in data
    assert isinstance(data["prob"], float)
    assert data["label"] in (0, 1)

def test_predict_multipart():
    b = _fake_png()
    files = {"file": ("x.png", b, "image/png")}
    data = {"NO2": "10.0", "CO": "0.3", "PM2.5": "25.0", "PM10": "40.0"}
    r = client.post("/predict-multipart", files=files, data=data)
    assert r.status_code == 200
    data = r.json()
    assert "prob" in data and "label" in data
    assert isinstance(data["prob"], float)
    assert data["label"] in (0, 1)
