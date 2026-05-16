from transformers import pipeline


class EmotionAnalyzer:
    """Analisador de emoções em texto usando HuggingFace Transformers."""

    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self):
        """Carrega o pipeline de classificação de emoções uma única vez."""
        print(f"[EmotionAnalyzer] Carregando modelo '{self.MODEL_NAME}'...")
        self.classifier = pipeline(
            "text-classification",
            model=self.MODEL_NAME,
            top_k=None,
        )
        print("[EmotionAnalyzer] Modelo carregado com sucesso.")

    def analyze(self, text: str) -> dict:
        """
        Analisa as emoções presentes em um texto.

        Args:
            text: Texto transcrito para análise emocional.

        Returns:
            Dicionário com emoção principal, confiança e lista completa.
        """
        print("[EmotionAnalyzer] Iniciando análise de emoções...")

        if not text or not text.strip():
            print("[EmotionAnalyzer] ERRO: Texto vazio recebido.")
            return {
                "primary_emotion": "unknown",
                "confidence": 0.0,
                "all_emotions": [],
            }

        try:
            results = self.classifier(text)[0]

            # Ordenar por score decrescente
            results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

            primary = results_sorted[0]

            output = {
                "primary_emotion": primary["label"],
                "confidence": round(primary["score"], 4),
                "all_emotions": [
                    {"emotion": r["label"], "score": round(r["score"], 4)}
                    for r in results_sorted
                ],
            }

            print(f"[EmotionAnalyzer] Emoção principal: {output['primary_emotion']} "
                  f"({output['confidence']})")
            return output

        except Exception as e:
            print(f"[EmotionAnalyzer] ERRO na análise: {e}")
            return {
                "primary_emotion": "error",
                "confidence": 0.0,
                "all_emotions": [],
            }
