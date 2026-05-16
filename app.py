import os
import tempfile

import streamlit as st

from audio.processor import AudioProcessor
from audio.emotion_analyzer import EmotionAnalyzer

# Configuração da página
st.set_page_config(
    page_title="IA Multimodal - Saúde da Mulher",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_audio_processor():
    """Carrega o AudioProcessor uma única vez (cache do Streamlit)."""
    return AudioProcessor(model_name="base")


@st.cache_resource
def load_emotion_analyzer():
    """Carrega o EmotionAnalyzer uma única vez (cache do Streamlit)."""
    return EmotionAnalyzer()


# Sidebar com descrição do projeto
with st.sidebar:
    st.header("Sobre o Projeto")
    st.write(
        """
        Projeto acadêmico de IA multimodal voltado para a **saúde da mulher**.

        O sistema combina análise de áudio e vídeo para apoiar
        profissionais de saúde na triagem e acompanhamento de pacientes.

        **Tecnologias planejadas:**
        - Whisper (transcrição de áudio)
        - DeepFace (análise facial)
        - LangChain + ChromaDB (RAG)
        - Azure Blob Storage
        """
    )

# Título principal
st.title("🩺 IA Multimodal — Saúde da Mulher")
st.markdown("---")

# Upload de áudio
st.subheader("🎙️ Upload de Áudio")
audio_file = st.file_uploader(
    "Envie um arquivo de áudio (.wav ou .mp3)",
    type=["wav", "mp3"],
    key="audio_upload",
)

if audio_file is not None:
    st.success(f"Áudio carregado com sucesso: **{audio_file.name}**")
    st.audio(audio_file)

    # Transcrição com Whisper
    with st.spinner("Transcrevendo áudio com Whisper..."):
        # Salvar arquivo temporário
        tmp_path = None
        try:
            suffix = os.path.splitext(audio_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getbuffer())
                tmp_path = tmp.name

            # Transcrever
            processor = load_audio_processor()
            transcription = processor.transcribe(tmp_path)

        finally:
            # Remover arquivo temporário
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Exibir resultado
    if transcription:
        st.subheader("📝 Transcrição")
        st.write(transcription)

        # Análise emocional
        with st.spinner("Analisando emoções no texto..."):
            analyzer = load_emotion_analyzer()
            emotions = analyzer.analyze(transcription)

        st.subheader("🧠 Análise Emocional")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Emoção Principal", emotions["primary_emotion"].capitalize())
        with col2:
            st.metric("Confiança", f"{emotions['confidence']:.1%}")

        if emotions["all_emotions"]:
            st.write("**Todas as emoções detectadas:**")
            for item in emotions["all_emotions"]:
                st.write(f"- {item['emotion'].capitalize()}: {item['score']:.1%}")

        # Alertas clínicos baseados na emoção principal
        st.subheader("⚠️ Alerta Clínico")

        clinical_alerts = {
            "fear": ("warning", "Possível sinal de ansiedade ou insegurança emocional."),
            "sadness": ("error", "Possível sofrimento emocional detectado."),
            "anger": ("warning", "Possível estresse elevado identificado."),
            "disgust": ("warning", "Possível desconforto ou aversão emocional."),
            "surprise": ("info", "Reação de surpresa detectada — avaliar contexto."),
            "joy": ("info", "Estado emocional positivo identificado."),
            "neutral": ("info", "Estado emocional neutro — sem alertas relevantes."),
        }

        emotion = emotions["primary_emotion"].lower()
        alert_type, alert_msg = clinical_alerts.get(
            emotion, ("info", "Emoção não mapeada para alerta clínico.")
        )

        if alert_type == "error":
            st.error(f"🚨 {alert_msg}")
        elif alert_type == "warning":
            st.warning(f"⚠️ {alert_msg}")
        else:
            st.info(f"ℹ️ {alert_msg}")

        st.caption(
            "⚕️ **Disclaimer:** Este sistema não substitui avaliação médica profissional. "
            "Os resultados são indicativos e devem ser interpretados por um especialista."
        )
    else:
        st.warning("Não foi possível transcrever o áudio.")

st.markdown("---")

# Upload de vídeo
st.subheader("🎥 Upload de Vídeo")
video_file = st.file_uploader(
    "Envie um arquivo de vídeo (.mp4 ou .mov)",
    type=["mp4", "mov"],
    key="video_upload",
)

if video_file is not None:
    st.success(f"Vídeo carregado com sucesso: **{video_file.name}**")
    st.video(video_file)
