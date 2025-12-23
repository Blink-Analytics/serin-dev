# engine.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

class VectorEngine:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        """
        Generates dense vectors for a list of texts.
        """
        # Convert to list to ensure compatibility
        text_list = list(texts)
        return self.model.encode(text_list, convert_to_tensor=True, show_progress_bar=True)
    
    @staticmethod
    def compute_similarity(embedding_a, embedding_b):
        """
        Returns cosine similarity between two vectors.
        """
        # util.cos_sim returns a tensor matrix; we extract the single float value
        return float(util.cos_sim(embedding_a, embedding_b)[0][0])