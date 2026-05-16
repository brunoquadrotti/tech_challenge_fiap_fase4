import whisper


class AudioProcessor:
    """Processador de áudio usando OpenAI Whisper para transcrição."""

    def __init__(self, model_name="base"):
        """Carrega o modelo Whisper uma única vez."""
        print(f"[AudioProcessor] Carregando modelo Whisper '{model_name}'...")
        self.model = whisper.load_model(model_name)
        print("[AudioProcessor] Modelo carregado com sucesso.")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcreve um arquivo de áudio (.wav ou .mp3) para texto.

        Args:
            audio_path: Caminho para o arquivo de áudio.

        Returns:
            Texto transcrito do áudio.
        """
        print(f"[AudioProcessor] Iniciando transcrição: {audio_path}")

        try:
            result = self.model.transcribe(audio_path)
            text = result["text"].strip()
            print("[AudioProcessor] Transcrição concluída.")
            return text

        except FileNotFoundError:
            print(f"[AudioProcessor] ERRO: Arquivo não encontrado — {audio_path}")
            return ""

        except Exception as e:
            print(f"[AudioProcessor] ERRO na transcrição: {e}")
            return ""
