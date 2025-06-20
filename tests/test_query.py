# python
import unittest
from unittest.mock import MagicMock
import numpy as np
from modules.query import retrieve_context


class TestRetrieveContext(unittest.TestCase):
    def test_retrieve_context_with_string_query_convert_to_vector(self):
        mock_vector_db = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_db.as_retriever.return_value = mock_retriever
        mock_embed_model = MagicMock()
        mock_embed_model.get_text_embedding.return_value = np.array([1.0, 2.0])

        mock_retriever.retrieve.return_value = [
            MagicMock(get_content=lambda: "Document 1"),
            MagicMock(get_content=lambda: "Document 2"),
        ]

        result = retrieve_context(
            query="test query",
            vector_db=mock_vector_db,
            embed_model=mock_embed_model,
            top_k=2,
            similarity_cutoff=0.5
        )

        self.assertEqual(result, "Document 1\nDocument 2")
        mock_embed_model.get_text_embedding.assert_any_call("test query")
        mock_embed_model.get_text_embedding.assert_any_call("Document 1")
        mock_embed_model.get_text_embedding.assert_any_call("Document 2")
        self.assertEqual(mock_embed_model.get_text_embedding.call_count, 3)
        mock_retriever.retrieve.assert_called_once()

    def test_retrieve_context_with_vector_query(self):
        mock_vector_db = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_db.as_retriever.return_value = mock_retriever
        mock_embed_model = MagicMock()
        mock_embed_model.get_text_embedding.return_value = np.array([1.0, 2.0])

        mock_retriever.retrieve.return_value = [
            MagicMock(get_content=lambda: "Document 1"),
            MagicMock(get_content=lambda: "Document 2"),
        ]

        result = retrieve_context(
            query=np.array([1.0, 2.0]),
            vector_db=mock_vector_db,
            embed_model=mock_embed_model,
            top_k=2,
            similarity_cutoff=0.5
        )

        self.assertEqual(result, "Document 1\nDocument 2")
        mock_retriever.retrieve.assert_called_once()

    def test_retrieve_context_with_invalid_query(self):
        mock_vector_db = MagicMock()

        with self.assertRaises(TypeError):
            retrieve_context(
                query=123,
                vector_db=mock_vector_db,
                embed_model=None,
                top_k=2,
                similarity_cutoff=0.5
            )


if __name__ == "__main__":
    unittest.main()
