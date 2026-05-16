import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="IA Multimodal - Saúde da Mulher",
    page_icon="🩺",
    layout="centered",
)

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
