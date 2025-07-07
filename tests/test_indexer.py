# test_indexer_compatible.py - Test that works with existing indexer.py

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch, MagicMock
from modules.indexer import (
    parse_source_path,
    validate_url,
    download_and_save_from_hf,
    download_and_save_from_url
)


class TestIndexerCompatible(unittest.TestCase):
    """Compatible tests for existing indexer functionality"""

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