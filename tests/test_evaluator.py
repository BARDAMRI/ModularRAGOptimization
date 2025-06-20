# Python
import unittest
from unittest.mock import patch, MagicMock

import torch

from scripts.evaluator import (
    load_llm,
    load_embedding_model,
    run_llm_query,
    compare_answers_with_embeddings,
    judge_with_llm,
    sanity_check,
    enumerate_top_documents,
    hill_climb_documents,
)


class TestEvaluatorFunctions(unittest.TestCase):
    @patch("scripts.evaluator.AutoTokenizer.from_pretrained")
    @patch("scripts.evaluator.AutoModelForCausalLM.from_pretrained")
    def test_load_llm(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        tokenizer, model = load_llm()
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)

    @patch("scripts.evaluator.SentenceTransformer")
    def test_load_embedding_model(self, mock_sentence_transformer):
        mock_sentence_transformer.return_value = MagicMock()
        model = load_embedding_model()
        self.assertIsNotNone(model)

    @patch("scripts.evaluator.load_llm")
    def test_run_llm_query(self, mock_load_llm):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_llm.return_value = (mock_tokenizer, mock_model)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Generated answer"
        query = "What is the largest country?"
        result = run_llm_query(query, mock_tokenizer, mock_model)
        self.assertEqual(result, "Generated answer")

    @patch("scripts.evaluator.load_embedding_model")
    def test_compare_answers_with_embeddings(self, mock_load_embedding_model):
        mock_model = MagicMock()
        mock_load_embedding_model.return_value = mock_model
        mock_model.encode.side_effect = lambda x, convert_to_tensor: torch.tensor([1.0, 2.0])
        mock_model.pytorch_cos_sim.side_effect = lambda x, y: torch.tensor([[0.9]])
        result = compare_answers_with_embeddings("query", "answer1", "answer2")
        self.assertIn(result, ["Optimized", "Original", "Tie"])

    @patch("scripts.evaluator.load_llm")
    def test_judge_with_llm(self, mock_load_llm):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_llm.return_value = (mock_tokenizer, mock_model)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Optimized"
        result = judge_with_llm("query", "answer1", "answer2", model=mock_model, tokenizer=mock_tokenizer)
        self.assertEqual(result, "Optimized")

    @patch("scripts.evaluator.retrieve_context")
    @patch("scripts.evaluator.compare_answers_with_embeddings")
    @patch("scripts.evaluator.judge_with_llm")
    @patch("scripts.evaluator.load_embedding_model")
    @patch("scripts.evaluator.load_llm")
    def test_sanity_check(self, mock_load_llm, mock_load_embedding_model, mock_judge_with_llm,
                          mock_compare_answers_with_embeddings, mock_retrieve_context):
        mock_vector_db = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_llm.return_value = (mock_tokenizer, mock_model)
        mock_embedding_model = MagicMock()
        mock_load_embedding_model.return_value = mock_embedding_model
        mock_retrieve_context.side_effect = ["Original answer from vector", "Optimized answer from vector"]
        mock_compare_answers_with_embeddings.return_value = "Optimized"
        mock_judge_with_llm.return_value = "Optimized"

        # Test with vector query
        vector_query = torch.tensor([1.0, 2.0])
        result = sanity_check("query1", "query2", mock_vector_db, vector_query=vector_query, convert_to_vector=True)
        self.assertEqual(result["final_decision"], "Optimized")
        self.assertEqual(result["original_answer"], "Original answer from vector")
        self.assertEqual(result["optimized_answer"], "Optimized answer from vector")

        # Test without vector query
        mock_retrieve_context.side_effect = None
        mock_run_llm_query = MagicMock(side_effect=["Original answer", "Optimized answer"])
        with patch("scripts.evaluator.run_llm_query", mock_run_llm_query):
            result = sanity_check("query1", "query2", mock_vector_db, convert_to_vector=False)
            self.assertEqual(result["final_decision"], "Optimized")
            self.assertEqual(result["original_answer"], "Original answer")
            self.assertEqual(result["optimized_answer"], "Optimized answer")

    @patch("scripts.evaluator.ResultsLogger")
    @patch("scripts.evaluator.plot_score_distribution")
    def test_enumerate_top_documents(self, mock_plot_score_distribution, mock_results_logger):
        mock_index = MagicMock()
        mock_index.as_retriever.return_value.retrieve.return_value = [
            MagicMock(node=MagicMock(get_content=lambda: "Document content"), score=0.95)
        ]

        # Test with valid query
        result = enumerate_top_documents(0, 1, "query", mock_index, MagicMock(), top_k=1)
        self.assertEqual(result["query"], "query")
        self.assertEqual(len(result["top_documents"]), 1)

        # Test with empty query
        mock_index.as_retriever.return_value.retrieve.return_value = []  # Mock no results for empty query
        result = enumerate_top_documents(0, 1, "", mock_index, MagicMock(), top_k=1)
        self.assertEqual(result["query"], "")
        self.assertEqual(len(result["top_documents"]), 0)

    @patch("scripts.evaluator.ResultsLogger")
    def test_hill_climb_documents(self, mock_results_logger):
        mock_index = MagicMock()
        mock_index.as_retriever.return_value.retrieve.return_value = [
            MagicMock(node=MagicMock(get_content=lambda: "Document content"), score=0.95)
        ]
        mock_llm_model = MagicMock()
        mock_llm_model.generate.return_value = torch.tensor([[50256, 50257, 50258]])  # Mock valid outputs
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = lambda x, skip_special_tokens: "Mocked Answer"
        mock_embedding_model = MagicMock()
        mock_embedding_model.get_text_embedding.side_effect = lambda x: [1.0, 2.0]

        # Test with valid query
        result = hill_climb_documents(0, 1, "query", mock_index, mock_llm_model, mock_tokenizer, mock_embedding_model)
        self.assertEqual(result["query"], "query")
        self.assertIsNotNone(result["answer"])
        self.assertIsNotNone(result["context"])

        # Test with empty query
        mock_index.as_retriever.return_value.retrieve.return_value = []  # Mock no results for empty query
        result = hill_climb_documents(0, 1, "", mock_index, mock_llm_model, mock_tokenizer, mock_embedding_model)
        self.assertEqual(result["query"], "")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["context"])


if __name__ == "__main__":
    unittest.main()
