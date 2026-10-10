import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from src.explainability.shap_explainer import load_final_model, make_pipeline, predict_proba

model, tok = load_final_model(os.path.join(os.path.dirname(__file__), "..", "models", "biobert_lora"))
pipe = make_pipeline(model, tok)

def classify(claim):
    if not claim.strip():
        return {}
    return predict_proba(pipe, claim)

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Affirmation de santé (en anglais)"),
    outputs=gr.Label(num_top_classes=4, label="Probabilités"),
    title="MediFact-Detector",
    description=("BioBERT + LoRA entraîné sur PubHealth. Démo de recherche : le modèle ne "
                 "vérifie pas les faits (il ne voit que la phrase, sans preuve) et se trompe "
                 "souvent (F1 macro ≈ 0,51). Ne pas utiliser pour une décision de santé."),
    examples=[["Vitamin C cures the common cold."],
              ["Smoking increases the risk of lung cancer."],
              ["Drinking water is essential for life."],
              ["Exercise improves mental health."]])

if __name__ == "__main__":
    demo.launch()