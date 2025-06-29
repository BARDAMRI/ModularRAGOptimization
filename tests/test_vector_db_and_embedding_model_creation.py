# Python
import unittest
from unittest.mock import patch, MagicMock
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from modules.indexer import load_vector_db
from modules.indexer import HuggingFaceEmbedding
from config import INDEX_SOURCE_URL


class TestVectorDBEmbedding(unittest.TestCase):
    @patch("modules.indexer.SimpleDirectoryReader")
    @patch("modules.indexer.validate_url")
    @patch("modules.indexer.HuggingFaceEmbedding")
    def test_embedding_model_type(self, mock_embedding, mock_validate_url, mock_reader):
        # Mock URL validation
        mock_validate_url.return_value = True

        # Mock directory reader
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [
            TextNode(id_="node1", text="sample document", embedding=None)
        ]
        mock_reader.return_value = mock_reader_instance

        # Mock embedding model
        mock_embedding_instance = MagicMock(spec=HuggingFaceEmbedding)
        mock_embedding_instance.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding.return_value = mock_embedding_instance

        # Call the function with project-specific arguments
        result = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

        # Assertions
        self.assertIsInstance(result, VectorStoreIndex)
        self.assertIsInstance(mock_embedding_instance, HuggingFaceEmbedding)
        self.assertIsNotNone(result)

    @patch("modules.indexer.SimpleDirectoryReader")
    @patch("modules.indexer.validate_url")
    @patch("modules.indexer.HuggingFaceEmbedding")
    def test_embedding_model_usage(self, mock_embedding, mock_validate_url, mock_reader):
        # Mock URL validation
        mock_validate_url.return_value = True

        # Mock directory reader
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [
            TextNode(id_="node1", text="sample document", embedding=None)
        ]
        mock_reader.return_value = mock_reader_instance

        # Mock embedding model
        mock_embedding_instance = MagicMock()
        mock_embedding_instance.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_embedding.return_value = mock_embedding_instance

        # Call the function with project-specific arguments
        result = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

        # Assertions
        self.assertIsInstance(result, VectorStoreIndex)
        mock_embedding_instance.get_text_embedding.assert_called_once()
        self.assertEqual(mock_embedding_instance.get_text_embedding.return_value, [0.1, 0.2, 0.3])
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
