import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Centralized model configuration"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "microsoft/DialoGPT-medium"  # Better for conversations
    device_priority: list = None

    def __post_init__(self):
        if self.device_priority is None:
            self.device_priority = ["cuda", "mps", "cpu"]


@dataclass
class RetrievalConfig:
    """RAG-specific configuration"""
    top_k: int = 5
    similarity_cutoff: float = 0.75
    max_context_length: int = 4000
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class OptimizationConfig:
    """Hill climbing and optimization settings"""
    max_retries: int = 3
    quality_threshold: float = 0.7
    temperature: float = 0.7
    max_new_tokens: int = 128
    convergence_threshold: float = 0.01