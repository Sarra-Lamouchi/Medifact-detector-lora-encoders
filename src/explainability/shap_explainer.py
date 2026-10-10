
import torch
import shap
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.data.preprocessor import ID2LABEL, LABEL2ID

BASE_NAME = "dmis-lab/biobert-base-cased-v1.2"


def load_final_model(adapter_dir="../models/biobert_lora", base_name=BASE_NAME):
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.eval()
    return model, tok


def make_pipeline(model, tok):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tok,
                    top_k=None, device=device, truncation=True, max_length=64)


def predict_proba(pipe, text):
    out = pipe(text)
    if isinstance(out[0], list):
        out = out[0]
    return {d["label"]: float(d["score"]) for d in out}


def explain(pipe, texts):
    return shap.Explainer(pipe)(texts)