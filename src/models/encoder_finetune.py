
from transformers import AutoModelForSequenceClassification

from src.models.common import evaluate, set_seed, train_model


def fine_tune_encoder(model_name, train_loader, val_loader, class_weights,
                      id2label, label2id, epochs=5, lr=2e-5, patience=2, seed=42):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    info = train_model(model, train_loader, val_loader, class_weights, id2label,
                       lr=lr, epochs=epochs, patience=patience)
    return model, info