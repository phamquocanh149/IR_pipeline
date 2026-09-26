"""Models package — exposes abstractions and factory."""

from ir_system.models.base import Model
from ir_system.models.embedding import EmbeddingModel
from ir_system.models.single_vector import SingleVectorEmbeddingModel
from ir_system.models.multi_vector import MultiVectorEmbeddingModel
from ir_system.models.cross_encoder import CrossEncoderModel
from ir_system.models.factory import create_model

__all__ = [
    "Model",
    "EmbeddingModel",
    "SingleVectorEmbeddingModel",
    "MultiVectorEmbeddingModel",
    "CrossEncoderModel",
    "create_model",
]
