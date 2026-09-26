from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import load_config
from src.logger import setup_logging, get_logger
from src.inference import Predictor

setup_logging("INFO")
logger = get_logger(__name__)

cfg = load_config("configs/default.yaml")
predictor = Predictor(
    model_path = cfg["paths"]["artifacts"] + "/model.pt",
    vocab_path = cfg["paths"]["artifacts"] + "/vocab.json",
    cfg        = cfg,
)

app = FastAPI()


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")

    logger.info("predict | text=%r", req.text[:80])
    result = predictor.predict(req.text)
    return result
