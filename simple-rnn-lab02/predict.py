from src.config import load_config
from src.logger import setup_logging
from src.inference import Predictor

setup_logging("INFO")
cfg = load_config("configs/default.yaml")

predictor = Predictor(
    model_path = cfg["paths"]["artifacts"] + "/model.pt",
    vocab_path = cfg["paths"]["artifacts"] + "/vocab.json",
    cfg        = cfg,
)

test_cases = [
    "This movie was absolutely amazing, I loved every second of it!",
    "A brilliant masterpiece. Outstanding performances by the entire cast.",
    "Terrible movie. Complete waste of time and money.",
    "The worst film I have ever seen in my entire life.",
    "Great!",
    "Boring.",
    "I went into this film with very high expectations after reading all the positive reviews online but sadly the movie failed to deliver on almost every level.",
    "I don't think this movie was good.",
    "This film is not bad at all, actually quite enjoyable.",
    "It started slow but the ending saved everything.",
    "The special effects were great but the story was a mess.",
    "Not the worst movie I have seen.",
]

print()
for text in test_cases:
    result = predictor.predict(text)
    print(f"[{result['label'].upper():<8}] {result['probability']:.2f}  {text[:80]}")
