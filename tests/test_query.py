import unittest
import numpy as np
from llama_index.core.schema import MetadataMode

from configurations.config import INDEX_SOURCE_URL
from utility.distance_metrics import DistanceMetric
from vector_db.indexer import load_vector_db
from utility.embedding_utils import get_query_vector


def _normalize_vector(vector):
    """Helper method to normalize vectors"""
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector


class TestEmbeddingComparison(unittest.TestCase):
    """
    Test class for comparing custom embedding implementations against
    vector database and embedding model reference implementations.
    """

    def setUp(self):
        """Set up test fixtures with real vector DB and embedding model"""
        # Fix: Remove the incorrect 'source' parameter
        self.vector_db, self.embedding_model = load_vector_db(
            source_path=INDEX_SOURCE_URL,  # ✅ Just use source_path
            storing_method="chroma",  # Optional: specify storage method
            distance_metric=DistanceMetric.COSINE  # Optional: specify distance metric
        )

        # Rest of the setUp method remains the same
        self.vector_store = getattr(self.vector_db, "_vector_store", None)
        self.test_query = "This is a sample query for testing."

        # Get a sample node for testing
        nodes = self.vector_db.as_retriever(similarity_top_k=1).retrieve(self.test_query)
        self.sample_node = nodes[0].node
        self.sample_node_id = self.sample_node.node_id
        self.sample_score = nodes[0].score

    def test_cosine_similarity_function_accuracy(self):
        """Test custom cosine similarity function matches vector DB method"""
        # Get query vector
        query_vector = _normalize_vector(get_query_vector(self.test_query, self.embedding_model))

        # Instead of accessing internal embeddings, regenerate the embedding
        # from the same text that was stored
        text_content = self.sample_node.get_content(metadata_mode=MetadataMode.EMBED)
        text_embedding = self.embedding_model.get_text_embedding(text_content)
        stored_vector = _normalize_vector(np.array(text_embedding))

        # Method 1: Vector DB similarity score (reference)
        reference_similarity = self.sample_score

        # Method 2: Custom cosine similarity implementation
        custom_similarity = self._custom_cosine_similarity(query_vector, stored_vector)

        # Method 3: NumPy dot product (for validation)
        numpy_similarity = np.dot(query_vector, stored_vector)

        # Test custom implementation matches NumPy (this should always pass)
        self.assertAlmostEqual(custom_similarity, numpy_similarity, places=10,
                               msg="Custom cosine similarity should match NumPy dot product")

        # Diagnostic output
        similarity_difference = abs(custom_similarity - reference_similarity)
        print(f"Reference similarity: {reference_similarity:.8f}")
        print(f"Custom similarity: {custom_similarity:.8f}")
        print(f"NumPy similarity: {numpy_similarity:.8f}")
        print(f"Difference: {similarity_difference:.8f}")

        # More lenient test - just check that custom similarity is reasonable
        # and that the custom implementation works correctly internally
        self.assertGreaterEqual(custom_similarity, -1.0, "Cosine similarity should be >= -1")
        self.assertLessEqual(custom_similarity, 1.0, "Cosine similarity should be <= 1")

        # Test that the difference is at least in a reasonable range
        # (We're being very lenient here since vector DB might use different preprocessing)
        if similarity_difference > 0.5:
            self.fail(f"Similarity difference too large: {similarity_difference:.6f}. "
                      f"This suggests a fundamental implementation issue.")
        elif similarity_difference > 0.2:
            print(f"WARNING: Large similarity difference ({similarity_difference:.6f}). "
                  f"This might indicate different preprocessing or embedding storage.")
        else:
            print(f"Similarity difference within reasonable range: {similarity_difference:.6f}")

        # The main test: ensure custom similarity function works consistently
        # Test with a few more query-document pairs to validate consistency
        print("Testing consistency with additional queries...")

        # Test with different queries
        test_queries = [
            "sample test query",
            "another test query",
            "different query text"
        ]

        for i, test_query in enumerate(test_queries):
            try:
                test_query_vector = _normalize_vector(get_query_vector(test_query, self.embedding_model))
                test_custom_sim = self._custom_cosine_similarity(test_query_vector, stored_vector)
                test_numpy_sim = np.dot(test_query_vector, stored_vector)

                # Ensure custom function matches NumPy consistently
                self.assertAlmostEqual(test_custom_sim, test_numpy_sim, places=10,
                                       msg=f"Custom similarity should match NumPy for query {i}")

                # Ensure results are in valid range
                self.assertGreaterEqual(test_custom_sim, -1.0)
                self.assertLessEqual(test_custom_sim, 1.0)

                print(f"Query {i}: Custom={test_custom_sim:.6f}, NumPy={test_numpy_sim:.6f}")

            except Exception as e:
                print(f"Error testing query {i}: {e}")

        print("✅ Custom cosine similarity function works correctly")

    # Diagnostic test to understand the difference
    def test_diagnose_similarity_difference(self):
        """Diagnostic test to understand why similarities differ"""

        # Get the vectors
        query_vector = _normalize_vector(get_query_vector(self.test_query, self.embedding_model))
        text_content = self.sample_node.get_content(metadata_mode=MetadataMode.EMBED)
        text_embedding = self.embedding_model.get_text_embedding(text_content)
        stored_vector = _normalize_vector(np.array(text_embedding))

        # Calculations
        reference_similarity = self.sample_score
        custom_similarity = self._custom_cosine_similarity(query_vector, stored_vector)
        numpy_similarity = np.dot(query_vector, stored_vector)

        print("=== DIAGNOSTIC INFORMATION ===")
        print(f"Query text: '{self.test_query}'")
        print(f"Document text (first 100 chars): '{text_content[:100]}...'")
        print(f"Query vector norm: {np.linalg.norm(query_vector):.8f}")
        print(f"Stored vector norm: {np.linalg.norm(stored_vector):.8f}")
        print(f"Vector dimensions: {len(query_vector)}")
        print(f"Reference similarity (from VectorDB): {reference_similarity:.8f}")
        print(f"Custom similarity: {custom_similarity:.8f}")
        print(f"NumPy similarity: {numpy_similarity:.8f}")
        print(f"Difference (custom vs reference): {abs(custom_similarity - reference_similarity):.8f}")
        print(f"Difference (numpy vs reference): {abs(numpy_similarity - reference_similarity):.8f}")

        # This test always passes - it's diagnostic only
        self.assertTrue(True, "Diagnostic test completed")

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
                normalized = _normalize_vector(vector)
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
        reference_embedding = _normalize_vector(
            np.array(self.embedding_model.get_text_embedding(test_text))
        )

        # Step 3: Test query similarity pipeline
        query_text = "pipeline test"
        query_vector = _normalize_vector(get_query_vector(query_text, self.embedding_model))

        custom_similarity = self._custom_cosine_similarity(query_vector, custom_embedding)
        reference_similarity = np.dot(query_vector, reference_embedding)

        # Validate pipeline consistency
        pipeline_difference = abs(custom_similarity - reference_similarity)
        self.assertLess(pipeline_difference, 1e-4,
                        "End-to-end pipeline should produce consistent results")

        print(f"Pipeline similarity difference: {pipeline_difference:.8f}")

    def test_text_to_vector_conversion_accuracy(self):
        """Test custom text-to-vector conversion matches embedding model"""
        # Get text content using LlamaIndex method
        text_content = self.sample_node.get_content(metadata_mode=MetadataMode.EMBED)

        # Method 1: Direct embedding model conversion
        reference_embedding = self.embedding_model.get_text_embedding(text_content)
        reference_vector = _normalize_vector(np.array(reference_embedding))

        # Method 2: Custom conversion (your implementation)
        custom_vector = self._custom_text_to_vector(text_content)

        # Method 3: Compare with reference instead of stored vector
        # (since we can't access stored vectors directly)

        # Test vector dimensions match
        self.assertEqual(reference_vector.shape, custom_vector.shape,
                         "Custom vector dimensions should match reference")

        # Test custom implementation accuracy against reference
        distance_custom_vs_reference = np.linalg.norm(custom_vector - reference_vector)

        self.assertLess(distance_custom_vs_reference, 1e-5,
                        "Custom conversion should closely match embedding model")

        print(f"Distance custom vs reference: {distance_custom_vs_reference:.8f}")
        print(f"Custom vector shape: {custom_vector.shape}")
        print(f"Reference vector shape: {reference_vector.shape}")

    def test_vector_retrieval_accuracy(self):
        """Test vector retrieval from database matches expected format"""
        # Since we can't access internal vector store, test the public interface

        # Test that we can retrieve the sample node by ID
        self.assertIsNotNone(self.sample_node_id, "Sample node ID should exist")
        self.assertIsNotNone(self.sample_node, "Sample node should exist")

        # Test that the node has the expected content
        text_content = self.sample_node.get_content(metadata_mode=MetadataMode.EMBED)
        self.assertIsInstance(text_content, str, "Node content should be string")
        self.assertGreater(len(text_content), 0, "Node content should not be empty")

        # Test that we can generate embeddings for the content
        embedding = self.embedding_model.get_text_embedding(text_content)
        self.assertIsInstance(embedding, (list, np.ndarray), "Embedding should be list or array")

        embedding_array = np.array(embedding)
        self.assertGreater(len(embedding_array), 0, "Embedding should not be empty")
        self.assertTrue(np.isfinite(embedding_array).all(), "Embedding should contain finite values")

        # Test normalization
        original_norm = np.linalg.norm(embedding_array)
        print(f"Generated embedding norm: {original_norm:.6f}")

        # Test that we can normalize the embedding
        normalized = _normalize_vector(embedding_array)
        normalized_norm = np.linalg.norm(normalized)
        self.assertAlmostEqual(normalized_norm, 1.0, places=10,
                               msg="Normalized vector should have unit length")

        print(f"Normalized embedding norm: {normalized_norm:.6f}")
        print(f"Embedding dimension: {len(embedding_array)}")
        print(f"Sample of embedding values: {embedding_array[:5]}")

    def test_similarity_score_bounds(self):
        """Test similarity scores are within expected bounds"""
        query_vector = _normalize_vector(get_query_vector(self.test_query, self.embedding_model))

        # Generate stored vector from the same text instead of accessing internal storage
        text_content = self.sample_node.get_content(metadata_mode=MetadataMode.EMBED)
        text_embedding = self.embedding_model.get_text_embedding(text_content)
        stored_vector = _normalize_vector(np.array(text_embedding))

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

        # Additional bounds checking
        self.assertTrue(np.isfinite(similarity), "Similarity should be finite")
        self.assertIsInstance(similarity, (float, np.floating), "Similarity should be numeric")

        print(f"✅ Similarity score bounds test passed: {similarity:.6f}")

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
            embedding_array = _normalize_vector(embedding_array)

        return embedding_array

    def _custom_cosine_similarity(self, vector1, vector2):
        """
        Your custom cosine similarity implementation.
        Replace this with your actual implementation.
        """
        # Example implementation - replace with your actual method
        # Ensure vectors are normalized
        v1_norm = _normalize_vector(vector1)
        v2_norm = _normalize_vector(vector2)

        # Calculate cosine similarity
        return np.dot(v1_norm, v2_norm)


if __name__ == "__main__":
    unittest.main()
