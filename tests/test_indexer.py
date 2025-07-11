# test_indexer_compatible.py - Test that works with existing indexer.py

import sys
import os
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch, MagicMock
from vector_db.simple_index import (
    parse_source_path,
    validate_url,
    download_and_save_from_hf,
    download_and_save_from_url
)


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

    @patch("modules.indexer.requests.get")
    @patch("modules.indexer.os.makedirs")
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

    @patch("modules.indexer.load_dataset")
    @patch("modules.indexer.os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_download_from_hf(self, mock_open, mock_makedirs, mock_load_dataset):
        """Test downloading from Hugging Face"""
        # Mock dataset
        mock_dataset = [{"text": "Sample text 1"}, {"text": "Sample text 2"}]
        mock_load_dataset.return_value = mock_dataset

        # Call the function
        download_and_save_from_hf("wikipedia", "20220301.en", "test_dir", max_docs=2)

        # Verify calls
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_load_dataset.assert_called_once_with("wikipedia", "20220301.en", split="train", trust_remote_code=True)
        print("Download from HF test passed")

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test invalid source path
        with self.assertRaises(ValueError):
            parse_source_path("invalid_format")

        # Test invalid URL scheme
        with self.assertRaises(ValueError):
            validate_url("ftp://example.com/data")

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

    @patch("modules.indexer.os.path.exists")
    def test_load_vector_db_import(self, mock_exists):
        """Test that load_vector_db can be imported"""
        try:
            from modules.indexer import load_vector_db
            self.assertTrue(callable(load_vector_db))
            print("load_vector_db import test passed")
        except ImportError as e:
            self.fail(f"Failed to import load_vector_db: {e}")

    def test_vector_store_type_and_query(self):
        """Test the type of the vector store and Euclidean distance query"""
        # Save current working directory
        original_cwd = os.getcwd()

        try:
            # Change to root directory for the test
            os.chdir(self.root_dir)
            print(f"Changed working directory to: {os.getcwd()}")

            # Ensure directory still exists before test
            data_dir = os.path.join("data", "public_corpus")
            if not os.path.exists(data_dir):
                print(f"WARNING: Directory {data_dir} does not exist, recreating...")
                os.makedirs(data_dir, exist_ok=True)
                # Create a test document
                with open(os.path.join(data_dir, "test_doc.txt"), "w") as f:
                    f.write("Test document for vector store testing.")

            from modules.indexer import load_vector_db
            from llama_index.vector_stores.chroma import ChromaVectorStore

            vector_db, embed_model = load_vector_db(source="local")
            vector_store = vector_db.vector_store

            # Check type
            self.assertIsInstance(vector_store, ChromaVectorStore)
            print(f"Vector store type: {type(vector_store).__name__}")

            # Create a dummy query embedding
            dummy_text = "This is a test query."
            dummy_vector = embed_model.get_text_embedding(dummy_text)

            # Query nearest document using VectorStoreQuery
            from llama_index.core.vector_stores import VectorStoreQuery

            query = VectorStoreQuery(
                query_embedding=dummy_vector,
                similarity_top_k=1
            )

            query_result = vector_store.query(query)

            # Check results
            self.assertIsNotNone(query_result)
            self.assertTrue(len(query_result.nodes) > 0, "No results returned from query")

            print(f"Query returned {len(query_result.nodes)} results")
            if query_result.ids:
                print("Nearest document ID:", query_result.ids[0])
            if query_result.similarities:
                print("Similarity score:", query_result.similarities[0])
            if query_result.nodes:
                print("Document text (start):",
                      query_result.nodes[0].text[:100] if query_result.nodes[0].text else "No text")

        finally:
            # Always restore original working directory
            os.chdir(original_cwd)


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
        from modules.indexer import load_vector_db
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
