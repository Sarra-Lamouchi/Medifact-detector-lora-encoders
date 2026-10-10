
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from src.models.common import evaluate, set_seed, train_model


def fine_tune_lora(model_name, train_loader, val_loader, class_weights,
                   id2label, label2id, epochs=8, lr=3e-4, patience=3,
                   r=16, alpha=32, dropout=0.1, seed=42):
    set_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha,
                     lora_dropout=dropout, target_modules=["query", "value"])
    model = get_peft_model(base, cfg)
    info = train_model(model, train_loader, val_loader, class_weights, id2label,
                       lr=lr, epochs=epochs, patience=patience)
    info.update({"r": r, "alpha": alpha})
    return model, info