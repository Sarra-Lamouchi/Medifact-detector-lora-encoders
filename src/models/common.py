
import random
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(train_ds, val_ds, test_ds, batch_size=8, seed=42):
    """Le seed contrôle aussi l'ordre de mélange du train."""
    g = torch.Generator()
    g.manual_seed(seed)
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    return train, val, test


def evaluate(model, loader, id2label):
    """Format unique pour full et LoRA : toujours un dict."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            preds += out.logits.argmax(-1).cpu().tolist()
            labels += b["labels"].cpu().tolist()
    names = [id2label[i] for i in range(len(id2label))]
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_per_class": dict(zip(names, f1_score(labels, preds, average=None, zero_division=0))),
        "preds": preds,
        "labels": labels,
        "report": classification_report(labels, preds, target_names=names, digits=4, zero_division=0),
    }


def clone_trainable_state(model):
    """Ne copie que les paramètres entraînables (LoRA : ~0,3M ; full : tous)."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}


def train_model(model, train_loader, val_loader, class_weights, id2label,
                lr, epochs, patience, weight_decay=0.01, warmup_ratio=0.1, verbose=True):
    model.to(DEVICE)
    weights = torch.as_tensor(class_weights, dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    best_f1, best_epoch, best_state, wait = -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for b in train_loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = loss_fn(logits, b["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()

        val_f1 = evaluate(model, val_loader, id2label)["f1_macro"]
        if verbose:
            print(f"Epoch {epoch}/{epochs} | loss {running/len(train_loader):.4f} | val F1 macro {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            best_state = clone_trainable_state(model)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping à l'epoch {epoch}")
                break

    # strict=False : best_state ne contient que les poids entraînables
    model.load_state_dict(best_state, strict=False)

    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "training_time_sec": time.time() - start,
        "peak_memory_gb": peak,          # /1e9 partout
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
    }