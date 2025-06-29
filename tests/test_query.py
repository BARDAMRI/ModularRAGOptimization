import unittest
from unittest.mock import MagicMock
import numpy as np
from llama_index.core.schema import MetadataMode

from config import INDEX_SOURCE_URL
from modules.indexer import load_vector_db
from utility.embedding_utils import get_query_vector


class TestEmbeddingComparison(unittest.TestCase):
    """
    Test class for comparing custom embedding implementations against
    vector database and embedding model reference implementations.
    """

    def setUp(self):
        """Set up test fixtures with real vector DB and embedding model"""
        self.vector_db, self.embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
        self.vector_store = getattr(self.vector_db, "_vector_store", None)
        self.test_query = "This is a sample query for testing."

        # Get a sample node for testing
        nodes = self.vector_db.as_retriever(similarity_top_k=1).retrieve(self.test_query)
        self.sample_node = nodes[0].node
        self.sample_node_id = self.sample_node.node_id
        self.sample_score = nodes[0].score

    def test_text_to_vector_conversion_accuracy(self):
        """Test custom text-to-vector conversion matches embedding model"""
        # Get text content using LlamaIndex method
        text_content = self.sample_node.get_content(metadata_mode=MetadataMode.EMBED)

        # Method 1: Direct embedding model conversion
        reference_embedding = self.embedding_model.get_text_embedding(text_content)
        reference_vector = self._normalize_vector(np.array(reference_embedding))

        # Method 2: Custom conversion (your implementation)
        custom_vector = self._custom_text_to_vector(text_content)

        # Method 3: Stored vector from database
        stored_vector = self._normalize_vector(np.array(self.vector_store.data.embedding_dict[self.sample_node_id]))

        # Test vector dimensions match
        self.assertEqual(reference_vector.shape, custom_vector.shape,
                         "Custom vector dimensions should match reference")
        self.assertEqual(reference_vector.shape, stored_vector.shape,
                         "All vectors should have same dimensions")

        # Test custom implementation accuracy
        distance_custom_vs_reference = np.linalg.norm(custom_vector - reference_vector)
        distance_custom_vs_stored = np.linalg.norm(custom_vector - stored_vector)

        self.assertLess(distance_custom_vs_reference, 1e-5,
                        "Custom conversion should closely match embedding model")
        self.assertLess(distance_custom_vs_stored, 0.1,
                        "Custom conversion should be reasonably close to stored vector")

        print(f"Distance custom vs reference: {distance_custom_vs_reference:.8f}")
        print(f"Distance custom vs stored: {distance_custom_vs_stored:.8f}")

    def test_cosine_similarity_function_accuracy(self):
        """Test custom cosine similarity function matches vector DB method"""
        # Get query vector
        query_vector = self._normalize_vector(get_query_vector(self.test_query, self.embedding_model))
        stored_vector = self._normalize_vector(np.array(self.vector_store.data.embedding_dict[self.sample_node_id]))

        # Method 1: Vector DB similarity score (reference)
        reference_similarity = self.sample_score

        # Method 2: Custom cosine similarity implementation
        custom_similarity = self._custom_cosine_similarity(query_vector, stored_vector)

        # Method 3: NumPy dot product (for validation)
        numpy_similarity = np.dot(query_vector, stored_vector)

        # Test custom implementation matches NumPy
        self.assertAlmostEqual(custom_similarity, numpy_similarity, places=10,
                               msg="Custom cosine similarity should match NumPy dot product")

        # Test custom implementation matches vector DB (within tolerance)
        similarity_difference = abs(custom_similarity - reference_similarity)
        self.assertLess(similarity_difference, 1e-5,
                        f"Custom similarity should closely match vector DB score. "
                        f"Difference: {similarity_difference}")

        print(f"Reference similarity: {reference_similarity:.8f}")
        print(f"Custom similarity: {custom_similarity:.8f}")
        print(f"NumPy similarity: {numpy_similarity:.8f}")

    def test_embedding_normalization_consistency(self):
        """Test normalization produces consistent unit vectors"""
        # Test with various vector magnitudes
        test_vectors = [
            np.array([3.0, 4.0, 0.0]),  # Standard case
            np.array([1e-10, 1e-10, 1e-10]),  # Very small
            np.array([1e6, 1e6, 1e6]),  # Very large
            np.array([1.0, 0.0, 0.0]),  # Already normalized
        ]

        for i, vector in enumerate(test_vectors):
            with self.subTest(i=i):
                normalized = self._normalize_vector(vector)
                norm = np.linalg.norm(normalized)

                if np.linalg.norm(vector) > 0:  # Skip zero vectors
                    self.assertAlmostEqual(norm, 1.0, places=10,
                                           msg=f"Vector {i} should be normalized to unit length")

    def test_embedding_pipeline_end_to_end(self):
        """Test complete embedding pipeline matches expected workflow"""
        test_text = "End-to-end pipeline test text"

        # Step 1: Convert text to embedding using your pipeline
        custom_embedding = self._custom_text_to_vector(test_text)

        # Step 2: Get reference embedding
        reference_embedding = self._normalize_vector(
            np.array(self.embedding_model.get_text_embedding(test_text))
        )

        # Step 3: Test query similarity pipeline
        query_text = "pipeline test"
        query_vector = self._normalize_vector(get_query_vector(query_text, self.embedding_model))

        custom_similarity = self._custom_cosine_similarity(query_vector, custom_embedding)
        reference_similarity = np.dot(query_vector, reference_embedding)

        # Validate pipeline consistency
        pipeline_difference = abs(custom_similarity - reference_similarity)
        self.assertLess(pipeline_difference, 1e-4,
                        "End-to-end pipeline should produce consistent results")

        print(f"Pipeline similarity difference: {pipeline_difference:.8f}")

    def test_vector_retrieval_accuracy(self):
        """Test vector retrieval from database matches expected format"""
        # Test stored vector retrieval
        self.assertIn(self.sample_node_id, self.vector_store.data.embedding_dict,
                      "Node ID should exist in vector store")

        stored_embedding = self.vector_store.data.embedding_dict[self.sample_node_id]

        # Validate stored vector properties
        self.assertIsInstance(stored_embedding, (list, np.ndarray))
        stored_array = np.array(stored_embedding)
        self.assertGreater(len(stored_array), 0, "Stored vector should not be empty")
        self.assertTrue(np.isfinite(stored_array).all(), "Stored vector should contain finite values")

        # Test normalization status
        original_norm = np.linalg.norm(stored_array)
        print(f"Stored vector original norm: {original_norm:.6f}")

        # Most vectors should be close to normalized
        if 0.8 < original_norm < 1.2:
            print("Vector appears to be normalized")
        else:
            print("Vector requires normalization")

    def test_similarity_score_bounds(self):
        """Test similarity scores are within expected bounds"""
        query_vector = self._normalize_vector(get_query_vector(self.test_query, self.embedding_model))
        stored_vector = self._normalize_vector(np.array(self.vector_store.data.embedding_dict[self.sample_node_id]))

        similarity = self._custom_cosine_similarity(query_vector, stored_vector)

        # Cosine similarity should be between -1 and 1
        self.assertGreaterEqual(similarity, -1.0, "Cosine similarity should be >= -1")
        self.assertLessEqual(similarity, 1.0, "Cosine similarity should be <= 1")

        # For normalized vectors, similarity should be reasonable
        if similarity < 0.1:
            print(f"Low similarity detected: {similarity:.6f}")
        elif similarity > 0.8:
            print(f"High similarity detected: {similarity:.6f}")
        else:
            print(f"Moderate similarity: {similarity:.6f}")

    def _custom_text_to_vector(self, text):
        """
        Your custom text-to-vector conversion implementation.
        Replace this with your actual implementation.
        """
        # Example implementation - replace with your actual method
        embedding = self.embedding_model.get_text_embedding(text)
        embedding_array = np.array(embedding)

        # Apply your custom normalization logic
        norm = np.linalg.norm(embedding_array)
        if norm > 1.1 or norm < 0.9:  # Not normalized
            embedding_array = self._normalize_vector(embedding_array)

        return embedding_array

    def _custom_cosine_similarity(self, vector1, vector2):
        """
        Your custom cosine similarity implementation.
        Replace this with your actual implementation.
        """
        # Example implementation - replace with your actual method
        # Ensure vectors are normalized
        v1_norm = self._normalize_vector(vector1)
        v2_norm = self._normalize_vector(vector2)

        # Calculate cosine similarity
        return np.dot(v1_norm, v2_norm)

    def _normalize_vector(self, vector):
        """Helper method to normalize vectors"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector


if __name__ == "__main__":
    unittest.main()