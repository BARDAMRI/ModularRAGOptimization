# test_indexer_compatible.py - Test that works with existing indexer.py

import sys
import os
import shutil

from utility.distance_metrics import DistanceMetric
from utility.vector_db_utils import parse_source_path, validate_url, download_and_save_from_url, \
    download_and_save_from_hf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch, MagicMock


class TestIndexerCompatible(unittest.TestCase):
    """Compatible tests for existing indexer functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test data directory before running tests"""
        # Get the root directory (parent of tests directory)
        test_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(test_dir)

        # Create test data directory in root
        cls.test_data_dir = os.path.join(root_dir, "data", "public_corpus")
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Store root dir for cleanup
        cls.root_dir = root_dir

        # Create some test documents
        test_docs = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on data.",
            "Deep learning uses neural networks with multiple layers."
        ]

        for i, content in enumerate(test_docs):
            file_path = os.path.join(cls.test_data_dir, f"test_doc_{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)

        # Verify directory was created
        print(f"Created test data directory: {cls.test_data_dir}")
        print(f"Directory exists: {os.path.exists(cls.test_data_dir)}")
        print(f"Files in directory: {os.listdir(cls.test_data_dir)}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test data directory after tests"""
        # Clean up in root directory
        data_path = os.path.join(cls.root_dir, "data")
        storage_path = os.path.join(cls.root_dir, "storage")

        if os.path.exists(data_path):
            shutil.rmtree(data_path)
            print(f"Removed {data_path}")
        if os.path.exists(storage_path):
            shutil.rmtree(storage_path)
            print(f"Removed {storage_path}")

    def test_parse_source_path_hf(self):
        """Test parsing Hugging Face dataset source path"""
        source_type, corpus_name = parse_source_path("wikipedia:20220301.en")

        self.assertEqual(source_type, "hf")
        self.assertEqual(corpus_name, "wikipedia_20220301.en")
        print("Parse source path test passed: HF format")

    def test_parse_source_path_url(self):
        """Test parsing URL source path"""
        source_type, corpus_name = parse_source_path("https://example.com/corpus.txt")

        self.assertEqual(source_type, "url")
        self.assertEqual(corpus_name, "corpus.txt")
        print("Parse source path test passed: URL format")

    def test_validate_url_https(self):
        """Test URL validation with HTTPS"""
        # This should pass
        result = validate_url("https://example.com/data")
        self.assertTrue(result)
        print("URL validation test passed: HTTPS")

    def test_validate_url_http_fails(self):
        """Test URL validation with HTTP (should fail in existing code)"""
        with self.assertRaises(ValueError):
            validate_url("http://example.com/data")
        print("URL validation test passed: HTTP properly rejected")

    def test_validate_url_invalid_domain(self):
        """Test URL validation with invalid domain"""
        with self.assertRaises(ValueError):
            validate_url("https://malicious-site.com/data")
        print("URL validation test passed: Invalid domain rejection")

    @patch("utility.vector_db_utils.requests.get")
    @patch("utility.vector_db_utils.os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_download_from_url_https(self, mock_open, mock_makedirs, mock_requests_get):
        """Test downloading from URL with HTTPS (existing functionality)"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"Sample corpus data"]
        mock_requests_get.return_value.__enter__.return_value = mock_response

        # Call the function with HTTPS URL (should work with existing code)
        download_and_save_from_url("https://example.com/data", "test_dir")

        # Verify calls
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_requests_get.assert_called_once_with("https://example.com/data", stream=True)
        print("Download from URL test passed: HTTPS")

    @patch("utility.vector_db_utils.load_dataset")
    @patch("utility.vector_db_utils.os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_download_from_hf(self, mock_open, mock_makedirs, mock_load_dataset):
        """Test downloading from Hugging Face"""

        # Create a proper mock dataset that mimics HuggingFace dataset behavior
        mock_dataset = MagicMock()

        # Add required attributes
        mock_dataset.column_names = ['text']

        # Mock dataset length
        mock_dataset.__len__ = MagicMock(return_value=2)

        # Mock dataset indexing (for dataset[0] access)
        mock_dataset.__getitem__ = MagicMock(side_effect=[
            {"text": "Sample text 1"},  # dataset[0]
            {"text": "Sample text 2"}  # dataset[1]
        ])

        # Mock dataset iteration
        mock_dataset.__iter__ = MagicMock(return_value=iter([
            {"text": "Sample text 1"},
            {"text": "Sample text 2"}
        ]))

        mock_load_dataset.return_value = mock_dataset

        # Call the function
        download_and_save_from_hf("wikipedia", "20220301.en", "test_dir", max_docs=2)

        # Verify calls
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_load_dataset.assert_called_once_with("wikipedia", "20220301.en", split="train", trust_remote_code=True)

        # Verify that files were written (should be called twice - once for each document)
        assert mock_open().write.call_count == 2

        print("Download from HF test passed")

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        source_type, name = parse_source_path("invalid_format")
        self.assertEqual(source_type, "local")
        self.assertTrue(name)
        print("Fallback to local path handling passed for invalid input")

        print("Error handling test passed")

    def test_indexer_functions_exist(self):
        """Test that all required functions exist and are callable"""
        functions_to_test = [
            parse_source_path,
            validate_url,
            download_and_save_from_hf,
            download_and_save_from_url
        ]

        for func in functions_to_test:
            self.assertTrue(callable(func))

        print("Function existence test passed")

    def test_load_vector_db_import(self):
        """Test that load_vector_db can be imported"""
        try:
            from vector_db.indexer import load_vector_db
            self.assertTrue(callable(load_vector_db))
            print("load_vector_db import test passed")
        except ImportError as e:
            self.fail(f"Failed to import load_vector_db: {e}")

    def test_vector_db_basic_functionality(self):
        """Test basic vector database functionality without accessing internals"""

        try:
            from vector_db.indexer import load_vector_db
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")

        try:
            # Create vector database
            vector_db, embed_model = load_vector_db(
                source_path=self.test_data_dir,
                storing_method="chroma",
                distance_metric=DistanceMetric.EUCLIDEAN
            )

            # Test basic interface methods
            self.assertIsNotNone(vector_db)
            self.assertIsNotNone(embed_model)

            # Test stats method (should exist on VectorDBInterface)
            stats = vector_db.get_stats()
            self.assertIsNotNone(stats)
            print(f"Vector DB stats: {stats}")

            # Test retrieve method (should exist on VectorDBInterface)
            results = vector_db.retrieve("test query", top_k=1)
            self.assertIsNotNone(results)
            print(f"Retrieve test: Found {len(results)} results")

            print("✅ Basic vector DB functionality test passed")

        except Exception as e:
            self.fail(f"Basic vector DB test failed: {e}")

    # Alternative: Mock-based version if you want to avoid real vector DB creation
    @patch("vector_db.indexer.VectorDBFactory.create_vector_db")
    @patch("vector_db.indexer._get_cached_embedding_model")
    def test_vector_store_type_and_query_mocked(self, mock_get_embedding, mock_create_db):
        """Test vector store creation with mocks"""
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.core.vector_stores import VectorStoreQuery
            from vector_db.indexer import load_vector_db
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")

        # Mock the embedding model
        mock_embedding = MagicMock()
        mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]  # Dummy vector
        mock_get_embedding.return_value = mock_embedding

        # Mock the vector database
        mock_vector_db = MagicMock()
        mock_vector_store = MagicMock(spec=ChromaVectorStore)
        mock_vector_db.vector_store = mock_vector_store
        mock_vector_db.get_stats.return_value = {"documents": 3, "method": "chroma"}
        mock_create_db.return_value = mock_vector_db

        # Mock query result
        mock_query_result = MagicMock()
        mock_query_result.nodes = [MagicMock()]
        mock_query_result.ids = ["doc_0"]
        mock_query_result.similarities = [0.95]
        mock_query_result.nodes[0].text = "Sample document text"
        mock_vector_store.query.return_value = mock_query_result

        # Test the function
        vector_db, embed_model = load_vector_db(
            source_path=self.test_data_dir,
            storing_method="chroma",
            distance_metric=DistanceMetric.EUCLIDEAN
        )

        # Verify mocks were called
        mock_create_db.assert_called_once()
        mock_get_embedding.assert_called_once()

        # Test vector store type
        self.assertIsInstance(vector_db.vector_store, ChromaVectorStore)

        print("Mocked vector store test passed")

    def test_load_vector_db_with_euclidean(self):
        """Test that vector DB can be created with Euclidean distance"""
        from vector_db.indexer import load_vector_db
        from utility.distance_metrics import DistanceMetric

        vector_db, _ = load_vector_db(
            source_path=self.test_data_dir,
            storing_method="chroma",
            distance_metric=DistanceMetric.EUCLIDEAN
        )

        self.assertIsNotNone(vector_db)
        print("✅ Vector DB created with Euclidean distance successfully")


def run_quick_test():
    """Run a quick functional test"""
    print("Running Quick Indexer Compatibility Test")
    print("=" * 40)

    try:
        # Test source path parsing
        source_type, corpus_name = parse_source_path("wikipedia:20220301.en")
        print(f"Parsed HF path: {source_type}, {corpus_name}")

        # Test URL validation
        try:
            validate_url("https://example.com/test")
            print("HTTPS URL validation: PASS")
        except ValueError:
            print("HTTPS URL validation: FAIL")

        try:
            validate_url("http://example.com/test")
            print("HTTP URL validation: FAIL (expected - should be rejected)")
        except ValueError:
            print("HTTP URL validation: PASS (correctly rejected)")

        # Test function imports
        from vector_db.indexer import load_vector_db
        print("load_vector_db import: PASS")

        print("Quick compatibility test completed successfully")
        return True

    except Exception as e:
        print(f"Quick compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_quick_test()
    else:
        # Run unit tests
        print("Running Indexer Compatibility Tests")
        print("=" * 35)
        unittest.main(verbosity=2)
