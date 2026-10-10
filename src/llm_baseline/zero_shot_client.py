"""
zero_shot_client.py -- Baseline de classification via LLM en zero-shot.
Supporte : anthropic, openai, gemini, groq.
Modele par defaut : llama-3.1-8b-instant (Groq) -- repond directement,
sans raisonnement interne cache qui consomme les tokens de sortie.
"""
import os
import time
from typing import Optional

VALID_LABELS = ['true', 'false', 'mixture', 'unproven']

PROMPT_TEMPLATE = """You are a fact-checking expert specialized in public health claims.
Classify the following claim into exactly ONE of these four categories:
- true: the claim is factually accurate
- false: the claim is factually incorrect
- mixture: the claim is partially true and partially false
- unproven: there is insufficient evidence to confirm or deny the claim

Claim: "{claim}"

Respond with ONLY one word: true, false, mixture, or unproven."""


def build_prompt(claim: str) -> str:
    return PROMPT_TEMPLATE.format(claim=claim)


def parse_response(raw_response: str) -> Optional[str]:
    text = raw_response.strip().lower()
    for label in VALID_LABELS:
        if label in text:
            return label
    return None


class ZeroShotClassifier:
    def __init__(self, provider: str = "groq", model: str = "llama-3.1-8b-instant",
                 requests_per_minute: int = 25):
        self.provider = provider
        self.model = model
        self.min_delay = 60.0 / requests_per_minute
        self._last_call_time = 0
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "groq":
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY manquante.")
            return Groq(api_key=api_key)

        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY manquante.")
            genai.configure(api_key=api_key)
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
            return genai.GenerativeModel(self.model, safety_settings=safety_settings)

        else:
            raise ValueError(f"Provider non supporte : {self.provider}")

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_call_time = time.time()

    def classify(self, claim: str, max_retries: int = 3) -> Optional[str]:
        prompt = build_prompt(claim)
        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            try:
                if self.provider == "groq":
                    response = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=10,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    raw_text = response.choices[0].message.content or ""

                elif self.provider == "gemini":
                    response = self._client.generate_content(
                        prompt, generation_config={"max_output_tokens": 10}
                    )
                    if not response.candidates or response.candidates[0].finish_reason == 2:
                        return "unproven"
                    raw_text = response.text

                predicted = parse_response(raw_text)
                if predicted is not None:
                    return predicted
                else:
                    print(f"Reponse non parsable : {raw_text!r} pour claim : {claim[:50]}...")

            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    print("Rate limit atteint, attente de 30s...")
                    time.sleep(30)
                else:
                    print(f"Erreur (tentative {attempt+1}/{max_retries}) : {e}")
                    time.sleep(2 ** attempt)
        return None


def run_zero_shot_evaluation(df, classifier, sample_size=None):
    df_sample = df.sample(n=sample_size, random_state=42) if sample_size else df
    predictions = []
    for i, row in df_sample.iterrows():
        pred = classifier.classify(row['claim'])
        predictions.append(pred)
        if len(predictions) % 10 == 0:
            print(f"Progression : {len(predictions)}/{len(df_sample)}")
    df_sample = df_sample.copy()
    df_sample['llm_prediction'] = predictions
    return df_sample