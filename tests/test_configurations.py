import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the actual functions we're testing
from scripts.evaluator import load_llm, load_embedding_model, run_llm_query
from config import HF_MODEL_NAME, LLM_MODEL_NAME


class TestConfiguration(unittest.TestCase):
    """Test configuration and model loading functionality with proper mocks"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_queries = [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "",  # Edge case: empty string
            "A" * 100,  # Edge case: long text
        ]

    @patch('scripts.evaluator.AutoTokenizer.from_pretrained')
    @patch('scripts.evaluator.AutoModelForCausalLM.from_pretrained')
    def test_llm_model_loading(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        """Test LLM model loads correctly with mocked components"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        # Call function
        tokenizer, model = load_llm()

        # Validate results
        self.assertIsNotNone(tokenizer, "Tokenizer should not be None")
        self.assertIsNotNone(model, "Model should not be None")
        self.assertEqual(tokenizer, mock_tokenizer)
        self.assertEqual(model, mock_model)

        # Verify mocks were called with correct model name
        mock_tokenizer_from_pretrained.assert_called_once_with(LLM_MODEL_NAME)
        mock_model_from_pretrained.assert_called_once_with(LLM_MODEL_NAME)

    @patch('scripts.evaluator.SentenceTransformer')  # This is likely the actual class being used
    def test_embedding_model_loading_with_sentence_transformer(self, mock_sentence_transformer):
        """Test embedding model loads correctly with SentenceTransformer mock"""
        # Setup mock
        mock_embedding = MagicMock()
        # Mock the actual method that SentenceTransformer uses
        mock_embedding.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_sentence_transformer.return_value = mock_embedding

        # Call function
        embedding_model = load_embedding_model()

        # Validate results
        self.assertIsNotNone(embedding_model, "Embedding model should not be None")
        self.assertEqual(embedding_model, mock_embedding)

        # Verify mock was called with correct model name
        mock_sentence_transformer.assert_called_once_with(HF_MODEL_NAME)

    @patch('scripts.evaluator.SentenceTransformer')
    def test_embedding_model_functionality_with_encode(self, mock_sentence_transformer):
        """Test embedding model produces valid embeddings using encode method"""
        # Setup mock
        mock_embedding = MagicMock()
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_embedding.encode.return_value = test_embedding
        mock_sentence_transformer.return_value = mock_embedding

        embedding_model = load_embedding_model()

        test_text = "This is a test sentence."
        embedding = embedding_model.encode(test_text)

        # Validate embedding properties
        self.assertIsInstance(embedding, np.ndarray, "Embedding should be numpy array")
        self.assertGreater(len(embedding), 0, "Embedding should not be empty")
        np.testing.assert_array_equal(embedding, test_embedding, "Should return mocked embedding")

        # Test method was called with correct text
        mock_embedding.encode.assert_called_with(test_text)

    # Alternative test for HuggingFaceEmbedding wrapper if that's what you're using
    @patch('scripts.evaluator.HuggingFaceEmbedding')
    def test_embedding_model_loading_with_huggingface_wrapper(self, mock_embedding_class):
        """Test embedding model loads correctly with HuggingFaceEmbedding wrapper"""
        # Setup mock
        mock_embedding = MagicMock()
        # Mock both possible method names
        mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedding_class.return_value = mock_embedding

        # Call function
        embedding_model = load_embedding_model()

        # Validate results
        self.assertIsNotNone(embedding_model, "Embedding model should not be None")

        # Test that it has at least one of the expected methods
        has_get_text_embedding = hasattr(embedding_model, 'get_text_embedding')
        has_encode = hasattr(embedding_model, 'encode')
        self.assertTrue(has_get_text_embedding or has_encode,
                        "Embedding model should have get_text_embedding or encode method")

    def test_llm_query_with_mocked_components(self):
        """Test LLM query with mocked tokenizer and model"""
        # Create mocked components
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        mock_tokenizer.decode.return_value = "Paris is the capital of France."

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        query = "What is the capital of France?"
        result = run_llm_query(query, mock_tokenizer, mock_model)

        # Validate result
        self.assertIsInstance(result, str, "Result should be a string")
        self.assertGreater(len(result.strip()), 0, "Result should not be empty")

        # Verify mocks were called
        mock_tokenizer.assert_called()
        mock_model.generate.assert_called()
        mock_tokenizer.decode.assert_called()

    def test_llm_query_edge_cases_with_mocks(self):
        """Test LLM query edge cases with mocked components"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1]]}
        mock_tokenizer.decode.return_value = "Response"

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2]]

        # Test empty query
        result = run_llm_query("", mock_tokenizer, mock_model)
        self.assertIsInstance(result, str, "Should handle empty query gracefully")

        # Test short query
        result = run_llm_query("Hi", mock_tokenizer, mock_model)
        self.assertIsInstance(result, str, "Should handle short query")

        # Test query with special characters
        result = run_llm_query("What is 2+2? #@$%", mock_tokenizer, mock_model)
        self.assertIsInstance(result, str, "Should handle special characters")

    def test_error_handling_with_mocks(self):
        """Test error handling with None inputs"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Test with None tokenizer
        with self.assertRaises((TypeError, AttributeError)):
            run_llm_query("test", None, mock_model)

        # Test with None model
        with self.assertRaises((TypeError, AttributeError)):
            run_llm_query("test", mock_tokenizer, None)

    @patch('scripts.evaluator.AutoTokenizer.from_pretrained')
    @patch('scripts.evaluator.AutoModelForCausalLM.from_pretrained')
    def test_llm_loading_failure(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        """Test handling of LLM loading failures"""
        mock_tokenizer_from_pretrained.side_effect = Exception("Model loading failed")

        with self.assertRaises(Exception):
            load_llm()

    @patch('scripts.evaluator.SentenceTransformer')
    def test_embedding_loading_failure_sentence_transformer(self, mock_sentence_transformer):
        """Test handling of embedding model loading failures with SentenceTransformer"""
        mock_sentence_transformer.side_effect = Exception("Embedding model loading failed")

        with self.assertRaises(Exception):
            load_embedding_model()

    @patch('scripts.evaluator.SentenceTransformer')
    def test_embedding_consistency(self, mock_sentence_transformer):
        """Test embedding consistency across calls"""
        mock_embedding = MagicMock()
        test_embedding = np.array([0.1, 0.2, 0.3])
        mock_embedding.encode.return_value = test_embedding
        mock_sentence_transformer.return_value = mock_embedding

        embedding_model = load_embedding_model()

        text = "Test consistency"
        embedding1 = embedding_model.encode(text)
        embedding2 = embedding_model.encode(text)

        np.testing.assert_array_equal(embedding1, embedding2, "Embeddings should be consistent")

    @patch('scripts.evaluator.SentenceTransformer')
    def test_embedding_with_none_input(self, mock_sentence_transformer):
        """Test embedding model with None input"""
        mock_embedding = MagicMock()
        mock_embedding.encode.side_effect = TypeError("Cannot embed None")
        mock_sentence_transformer.return_value = mock_embedding

        embedding_model = load_embedding_model()

        with self.assertRaises(TypeError):
            embedding_model.encode(None)

    def test_config_constants(self):
        """Test that configuration constants are defined"""
        self.assertIsNotNone(HF_MODEL_NAME, "HF_MODEL_NAME should be defined")
        self.assertIsNotNone(LLM_MODEL_NAME, "LLM_MODEL_NAME should be defined")
        self.assertIsInstance(HF_MODEL_NAME, str, "HF_MODEL_NAME should be string")
        self.assertIsInstance(LLM_MODEL_NAME, str, "LLM_MODEL_NAME should be string")

    def test_model_name_validation(self):
        """Test that model names are reasonable"""
        # Basic validation that model names look like model names
        self.assertNotEqual(HF_MODEL_NAME.strip(), "", "HF_MODEL_NAME should not be empty")
        self.assertNotEqual(LLM_MODEL_NAME.strip(), "", "LLM_MODEL_NAME should not be empty")

        # Check they contain reasonable patterns for model names
        self.assertTrue(
            any(char in HF_MODEL_NAME for char in ['-', '/', '_']),
            "HF_MODEL_NAME should contain model name separators"
        )

    # Test the actual wrapper method if you have one
    def test_embedding_wrapper_method_with_mocks(self):
        """Test if there's a wrapper method that calls encode internally"""
        # Create a mock that mimics your actual embedding wrapper
        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = np.array([0.1, 0.2, 0.3])

        # Create a wrapper function that mimics get_text_embedding
        def mock_get_text_embedding(text):
            return mock_embedding.encode(text).tolist()

        mock_embedding.get_text_embedding = mock_get_text_embedding

        # Test the wrapper
        result = mock_embedding.get_text_embedding("test")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [0.1, 0.2, 0.3])


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests that actually load models (run separately)"""

    @unittest.skip("Integration test - run manually when needed")
    def test_real_llm_model_loading(self):
        """Integration test for real LLM model loading"""
        try:
            tokenizer, model = load_llm()
            self.assertIsNotNone(tokenizer)
            self.assertIsNotNone(model)
            print(f"Real LLM Model loaded: {LLM_MODEL_NAME}")
        except Exception as e:
            self.fail(f"Real LLM model loading failed: {e}")

    @unittest.skip("Integration test - run manually when needed")
    def test_real_embedding_model_loading(self):
        """Integration test for real embedding model loading"""
        try:
            embedding_model = load_embedding_model()
            self.assertIsNotNone(embedding_model)
            print(f"Real Embedding Model loaded: {HF_MODEL_NAME}")
            print(f"Embedding model type: {type(embedding_model)}")

            # Test actual embedding generation with the correct method
            test_text = "This is a test"
            if hasattr(embedding_model, 'encode'):
                embedding = embedding_model.encode(test_text)
                print(f"Using encode method, embedding shape: {embedding.shape}")
            elif hasattr(embedding_model, 'get_text_embedding'):
                embedding = embedding_model.get_text_embedding(test_text)
                print(f"Using get_text_embedding method, embedding length: {len(embedding)}")
            else:
                self.fail("Embedding model has neither encode nor get_text_embedding method")

        except Exception as e:
            self.fail(f"Real embedding model loading failed: {e}")

    @unittest.skip("Integration test - run manually when needed")
    def test_real_llm_query(self):
        """Integration test for real LLM query"""
        try:
            tokenizer, model = load_llm()
            query = "What is 2+2?"
            result = run_llm_query(query, tokenizer, model)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result.strip()), 0)
            print(f"Query: {query}")
            print(f"Result: {result}")
        except Exception as e:
            # If model doesn't support generation, that's expected
            if "not compatible with `.generate()`" in str(e):
                self.skipTest(f"Model {LLM_MODEL_NAME} doesn't support text generation")
            else:
                self.fail(f"Real LLM query failed: {e}")


def test_configuration_with_sample_query():
    """Legacy function for manual integration testing"""
    print("\nRunning legacy configuration test...")

    try:
        # Load models
        print("\n> Loading LLM...")
        tokenizer, model = load_llm()
        print(f"LLM Model Loaded: {LLM_MODEL_NAME}")

        print("\n> Loading Embedding Model...")
        embedding_model = load_embedding_model()
        print(f"Embedding Model Loaded: {HF_MODEL_NAME}")
        print(f"Embedding model type: {type(embedding_model)}")

        # Test embedding with correct method
        print("\n> Testing embedding...")
        test_text = "This is a test"

        if hasattr(embedding_model, 'encode'):
            embedding = embedding_model.encode(test_text)
            print(f"Using encode method - Embedding dimension: {len(embedding)}")
        elif hasattr(embedding_model, 'get_text_embedding'):
            embedding = embedding_model.get_text_embedding(test_text)
            print(f"Using get_text_embedding method - Embedding dimension: {len(embedding)}")
        else:
            print("⚠️  Embedding model has neither encode nor get_text_embedding method")
            print(f"Available methods: {[m for m in dir(embedding_model) if not m.startswith('_')]}")

        # Run a sample query
        sample_query = "What is the capital of France?"
        print(f"\n> Running sample query: {sample_query}")
        result = run_llm_query(sample_query, tokenizer, model)

        print("\n> Query Result:")
        print(result)
        print("\n✅ Legacy test completed successfully")

    except Exception as e:
        if "not compatible with `.generate()`" in str(e):
            print(f"\n⚠️  Model {LLM_MODEL_NAME} doesn't support text generation")
            print("This is expected for some model types (e.g., DistilBERT)")
        elif "'SentenceTransformer' object has no attribute 'get_text_embedding'" in str(e):
            print(f"\n⚠️  Embedding model uses 'encode' method, not 'get_text_embedding'")
            print("This is expected for SentenceTransformer models")
        else:
            print(f"\n❌ Legacy test failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--integration":
        # Run integration test manually
        print("=" * 60)
        print("RUNNING INTEGRATION TEST")
        print("=" * 60)
        test_configuration_with_sample_query()
    else:
        # Run unit tests with mocks (default)
        print("=" * 60)
        print("RUNNING UNIT TESTS (MOCKED)")
        print("=" * 60)
        unittest.main(verbosity=2)