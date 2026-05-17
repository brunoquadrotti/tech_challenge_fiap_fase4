import os

from dotenv import load_dotenv
from openai import OpenAI

# Carregar variáveis de ambiente
load_dotenv()


class ReportGenerator:
    """Gerador de relatórios clínicos multimodais usando OpenAI API."""

    def __init__(self):
        """Inicializa o cliente OpenAI."""
        print("[ReportGenerator] Inicializando...")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não configurada no .env")

        self.client = OpenAI(api_key=api_key)
        print("[ReportGenerator] Cliente OpenAI inicializado.")

    def generate_report(
        self,
        audio_analysis: dict,
        video_analysis: dict,
        rag_results: list,
    ) -> str:
        """
        Gera relatório clínico consolidado a partir das análises multimodais.

        Args:
            audio_analysis: Resultado da análise emocional do áudio.
            video_analysis: Resultado da análise emocional do vídeo.
            rag_results: Trechos relevantes encontrados no RAG.

        Returns:
            Texto do relatório gerado.
        """
        print("[ReportGenerator] Gerando relatório clínico...")

        # Montar contexto para o prompt
        audio_context = self._format_audio(audio_analysis)
        video_context = self._format_video(video_analysis)
        rag_context = self._format_rag(rag_results)

        prompt = f"""
Você é um assistente clínico acadêmico especializado em saúde da mulher.
Com base nas análises multimodais abaixo, gere um relatório clínico consolidado.

## Análise Emocional do Áudio (transcrição)
{audio_context}

## Análise Emocional do Vídeo (expressão facial)
{video_context}

## Contexto Médico (base de conhecimento RAG)
{rag_context}

## Estrutura do Relatório
Gere o relatório com as seguintes seções:
1. **Resumo Emocional** — síntese das emoções detectadas nas duas modalidades
2. **Emoções Detectadas** — lista das emoções identificadas no áudio e vídeo
3. **Possíveis Sinais de Sofrimento Emocional** — indicadores relevantes
4. **Contexto Médico** — informações relevantes encontradas na base de conhecimento
5. **Recomendação Preventiva** — sugestões gerais de acompanhamento
6. **Disclaimer** — incluir obrigatoriamente: "Este sistema não substitui avaliação médica profissional. Os resultados são indicativos e devem ser interpretados por um especialista qualificado."

Mantenha linguagem profissional, empática e objetiva.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um assistente clínico acadêmico."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            report = response.choices[0].message.content.strip()
            print("[ReportGenerator] Relatório gerado com sucesso.")
            return report

        except Exception as e:
            print(f"[ReportGenerator] ERRO ao gerar relatório: {e}")
            return ""

    def _format_audio(self, audio_analysis: dict) -> str:
        """Formata dados da análise de áudio para o prompt."""
        if not audio_analysis:
            return "Nenhuma análise de áudio disponível."

        emotion = audio_analysis.get("primary_emotion", "N/A")
        confidence = audio_analysis.get("confidence", 0)
        return f"Emoção principal: {emotion} (confiança: {confidence:.1%})"

    def _format_video(self, video_analysis: dict) -> str:
        """Formata dados da análise de vídeo para o prompt."""
        if not video_analysis:
            return "Nenhuma análise de vídeo disponível."

        emotion = video_analysis.get("dominant_emotion", "N/A")
        confidence = video_analysis.get("confidence", 0)
        return f"Emoção dominante: {emotion} (confiança: {confidence:.1%})"

    def _format_rag(self, rag_results: list) -> str:
        """Formata resultados do RAG para o prompt."""
        if not rag_results:
            return "Nenhum contexto médico disponível na base de conhecimento."

        formatted = []
        for i, result in enumerate(rag_results, 1):
            content = result.get("content", "")[:300]
            formatted.append(f"Trecho {i}: {content}")

        return "\n".join(formatted)
