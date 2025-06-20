# Python
import unittest
from unittest.mock import patch, MagicMock
from modules.indexer import (
    download_and_save_from_hf,
    download_and_save_from_url,
    parse_source_path,
    load_vector_db,
)


class TestIndexerFunctions(unittest.TestCase):
    def setUp(self):
        self.target_dir = "test_dir"
        self.sample_text = "Sample text"
        self.sample_corpus = "Sample corpus text"

    @patch("modules.indexer.load_dataset")
    @patch("modules.indexer.os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_download_and_save_from_hf(self, mock_open, mock_makedirs, mock_load_dataset):
        # Mock dataset
        mock_load_dataset.return_value = [{"text": self.sample_text}] * 5

        # Call the function
        download_and_save_from_hf("dataset_name", "config", self.target_dir, max_docs=3)

        # Assertions
        mock_makedirs.assert_called_once_with(self.target_dir, exist_ok=True)
        self.assertEqual(mock_open.call_count, 3)
        for i in range(3):
            mock_open.assert_any_call(f"{self.target_dir}/doc_{i}.txt", "w", encoding="utf-8")

    @patch("modules.indexer.requests.get")
    @patch("modules.indexer.os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_download_and_save_from_url(self, mock_open, mock_makedirs, mock_requests_get):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [self.sample_corpus.encode("utf-8")]
        mock_requests_get.return_value = mock_response

        # Call the function
        download_and_save_from_url("http://example.com", self.target_dir)

        # Assertions
        mock_makedirs.assert_called_once_with(self.target_dir, exist_ok=True)
        mock_open.assert_called_once_with(f"{self.target_dir}/corpus.txt", "w", encoding="utf-8")
        mock_open().write.assert_called_once_with(self.sample_corpus)

    def test_parse_source_path(self):
        # Test URL source path
        self.assertEqual(parse_source_path("http://example.com/corpus"), ("url", "corpus"))
        # Test Hugging Face source path
        self.assertEqual(parse_source_path("hf://dataset:config"), ("hf", "dataset_config"))

    @patch("modules.indexer.HuggingFaceEmbedding")
    @patch("modules.indexer.SimpleDirectoryReader")
    @patch("modules.indexer.StorageContext.from_defaults")
    @patch("modules.indexer.load_index_from_storage")
    @patch("modules.indexer.os.path.exists", side_effect=[False, True])
    def test_load_vector_db_url_existing(
        self, mock_path_exists, mock_load_index_from_storage, mock_storage_context, mock_reader, mock_embedding
    ):
        # Mock dependencies
        mock_embedding.return_value = MagicMock()
        mock_storage_context.return_value = MagicMock()
        mock_load_index_from_storage.return_value = MagicMock()

        # Call the function
        result = load_vector_db(source="url", source_path="http://example.com/corpus")

        # Assertions
        mock_load_index_from_storage.assert_called_once()
        self.assertIsNotNone(result)

    @patch("modules.indexer.HuggingFaceEmbedding")
    @patch("modules.indexer.SimpleDirectoryReader")
    @patch("modules.indexer.GPTVectorStoreIndex.from_documents")
    @patch("modules.indexer.os.path.exists", return_value=False)
    @patch("modules.indexer.os.makedirs")
    def test_load_vector_db_url_new(
        self, mock_makedirs, mock_path_exists, mock_from_documents, mock_reader, mock_embedding
    ):
        # Mock dependencies
        mock_embedding.return_value = MagicMock()
        mock_reader.return_value.load_data.return_value = [{"text": self.sample_text}]
        mock_from_documents.return_value = MagicMock()

        # Call the function
        result = load_vector_db(source="url", source_path="http://example.com/corpus")

        # Assertions
        mock_reader.return_value.load_data.assert_called_once()
        mock_from_documents.assert_called_once()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()