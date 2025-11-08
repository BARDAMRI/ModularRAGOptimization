"""
Enumeration for vector database storing methods
"""

from enum import Enum
from typing import Dict, List


class StoringMethod(Enum):
    """
    Enumeration of available vector database storing methods.
    """
    CHROMA = "chroma"
    SIMPLE = "simple"
    LLAMA_INDEX = "llama_index"  # Backward compatibility alias for simple

    @classmethod
    def get_all_methods(cls) -> List[str]:
        """
        Get all available storing method values.

        Returns:
            List[str]: List of storing method strings
        """
        return [method.value for method in cls]

    @classmethod
    def get_descriptions(cls) -> Dict[str, str]:
        """
        Get descriptions for each storing method.

        Returns:
            Dict[str, str]: Method name to description mapping
        """
        return {
            cls.CHROMA.value: "ChromaDB - Persistent vector database with advanced filtering capabilities",
            cls.SIMPLE.value: "Simple Storage - LlamaIndex default storage with local persistence",
            cls.LLAMA_INDEX.value: "LlamaIndex (Alias) - Same as Simple Storage for backward compatibility"
        }

    @classmethod
    def get_recommendations(cls) -> Dict[str, str]:
        """
        Get recommendations for when to use each method.

        Returns:
            Dict[str, str]: Method name to recommendation mapping
        """
        return {
            cls.CHROMA.value: "Best for: Large datasets, advanced filtering, production deployments",
            cls.SIMPLE.value: "Best for: Small to medium datasets, quick prototyping, simple use cases",
            cls.LLAMA_INDEX.value: "Best for: Legacy code compatibility (same as Simple)"
        }

    @classmethod
    def is_valid_method(cls, method: str) -> bool:
        """
        Check if a storing method is valid.

        Args:
            method (str): Method to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return method in cls.get_all_methods()

    @classmethod
    def get_default(cls) -> str:
        """
        Get the default storing method.

        Returns:
            str: Default method
        """
        return cls.CHROMA.value

    def __str__(self) -> str:
        """String representation"""
        return self.value

    def __repr__(self) -> str:
        """Representation"""
        return f"StoringMethod.{self.name}"


# Convenience constants for backward compatibility
CHROMA = StoringMethod.CHROMA.value
SIMPLE = StoringMethod.SIMPLE.value
LLAMA_INDEX = StoringMethod.LLAMA_INDEX.value

# List of all methods for easy access
ALL_STORING_METHODS = StoringMethod.get_all_methods()
