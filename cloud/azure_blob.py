import os

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Carregar variáveis de ambiente
load_dotenv()


class AzureBlobService:
    """Serviço simples de upload para Azure Blob Storage."""

    def __init__(self):
        """Inicializa conexão com Azure Blob Storage."""
        print("[AzureBlobService] Inicializando conexão...")

        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_CONTAINER_NAME")

        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING não configurada no .env")

        if not self.container_name:
            raise ValueError("AZURE_CONTAINER_NAME não configurada no .env")

        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

        print("[AzureBlobService] Conexão estabelecida com sucesso.")

    def upload_file(self, local_file_path: str, blob_name: str) -> str:
        """
        Faz upload de um arquivo local para o Azure Blob Storage.

        Args:
            local_file_path: Caminho do arquivo local.
            blob_name: Nome do blob no container.

        Returns:
            URL do blob enviado.
        """
        print(f"[AzureBlobService] Enviando: {local_file_path} → {blob_name}")

        try:
            blob_client = self.container_client.get_blob_client(blob_name)

            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            blob_url = blob_client.url
            print(f"[AzureBlobService] Upload concluído: {blob_url}")
            return blob_url

        except FileNotFoundError:
            print(f"[AzureBlobService] ERRO: Arquivo não encontrado — {local_file_path}")
            return ""

        except Exception as e:
            print(f"[AzureBlobService] ERRO no upload: {e}")
            return ""
