from typing import Any, Dict, List, Optional
# from FlagEmbedding import FlagReranker
from langchain_community.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from app.Config.config import examples
# from FlagEmbedding import FlagICLModel
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] ="0, 1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class NVEmbed(HuggingFaceEmbeddings):
    eos_token: Optional[str] = None
    """End of sentence token to use."""
    query_instruction: Optional[str] = ""
    embed_instruction: Optional[str] = ""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + text + self.eos_token for text in texts]

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = self.query_instruction + text + self.eos_token
        return self.embed_documents([text])[0]