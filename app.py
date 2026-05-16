import os
import tempfile

import streamlit as st

from audio.processor import AudioProcessor
from audio.emotion_analyzer import EmotionAnalyzer
from video.processor import VideoProcessor
from rag.query import search_documents

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


@st.cache_resource
def load_video_processor():
    """Carrega o VideoProcessor uma única vez (cache do Streamlit)."""
    return VideoProcessor()


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

    # Análise emocional do vídeo
    with st.spinner("Analisando emoções no vídeo com DeepFace..."):
        # Salvar arquivo temporário
        tmp_video_path = None
        try:
            suffix = os.path.splitext(video_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_file.getbuffer())
                tmp_video_path = tmp.name

            # Analisar
            vp = load_video_processor()
            video_result = vp.analyze_video(tmp_video_path)

        finally:
            # Remover vídeo temporário
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    # Exibir resultados
    if video_result:
        st.subheader("🎭 Análise Emocional de Vídeo")

        # Exibir frame analisado
        if video_result.get("frame_path") and os.path.exists(video_result["frame_path"]):
            st.image(
                video_result["frame_path"],
                caption="Frame analisado",
                use_container_width=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Emoção Dominante", video_result["dominant_emotion"].capitalize())
        with col2:
            st.metric("Confiança", f"{video_result['confidence']:.1%}")
    else:
        st.warning("Não foi possível analisar o vídeo.")

st.markdown("---")

# Busca semântica RAG
st.subheader("📚 Busca Semântica em Protocolos Médicos")
st.write("Consulte a base de conhecimento com protocolos e documentos médicos.")

query = st.text_input(
    "Digite sua pergunta:",
    placeholder="Ex: Quais são os sintomas de depressão pós-parto?",
    key="rag_query",
)

if query:
    with st.spinner("Buscando nos documentos..."):
        results = search_documents(query)

    if results:
        st.write(f"**{len(results)} trecho(s) encontrado(s):**")
        for i, result in enumerate(results, 1):
            with st.expander(f"Resultado {i} — Similaridade: {result['score']:.2%}"):
                st.write(result["content"])
    else:
        st.info("Nenhum resultado encontrado. Verifique se os documentos foram ingeridos.")
