import os
import tempfile

import cv2
from deepface import DeepFace


class VideoProcessor:
    """Processador de vídeo para análise emocional facial com DeepFace."""

    def __init__(self):
        print("[VideoProcessor] Inicializado.")

    def analyze_video(self, video_path: str) -> dict:
        """
        Extrai um frame do vídeo e analisa a emoção facial.

        Args:
            video_path: Caminho para o arquivo de vídeo.

        Returns:
            Dicionário com emoção dominante, confiança e caminho do frame.
        """
        print(f"[VideoProcessor] Analisando vídeo: {video_path}")

        try:
            # Abrir vídeo
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("[VideoProcessor] ERRO: Não foi possível abrir o vídeo.")
                return {}

            # Extrair frame do meio do vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("[VideoProcessor] ERRO: Não foi possível extrair frame.")
                return {}

            # Salvar frame temporariamente
            frame_path = os.path.join(tempfile.gettempdir(), "frame_analysis.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"[VideoProcessor] Frame extraído e salvo em: {frame_path}")

            # Analisar emoção com DeepFace
            print("[VideoProcessor] Analisando emoção facial com DeepFace...")
            results = DeepFace.analyze(
                img_path=frame_path,
                actions=["emotion"],
                enforce_detection=False,
            )

            # DeepFace retorna lista quando há múltiplos rostos
            analysis = results[0] if isinstance(results, list) else results

            dominant_emotion = analysis["dominant_emotion"]
            emotions = analysis["emotion"]
            confidence = round(emotions[dominant_emotion] / 100.0, 4)

            output = {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "frame_path": frame_path,
            }

            print(f"[VideoProcessor] Emoção detectada: {dominant_emotion} "
                  f"({confidence:.1%})")
            return output

        except Exception as e:
            print(f"[VideoProcessor] ERRO na análise: {e}")
            return {}
