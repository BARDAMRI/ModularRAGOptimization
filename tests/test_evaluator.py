# test_evaluator.py - Fixed version with corrected mocking
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from scripts.evaluator import (
    load_llm,
    load_model,
    load_embedding_model,
    run_llm_query,
    compare_answers_with_embeddings,
    multi_method_comparison,
    judge_with_llm,
    advanced_sanity_check,
    enumerate_top_documents,
    hill_climb_documents,
)
from utility.similarity_calculator import SimilarityMethod


class TestEvaluatorFunctions(unittest.TestCase):
    """Test suite for evaluator functions with updated signatures and functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_embedding_model = MagicMock()
        self.mock_index = MagicMock()

    @patch("scripts.evaluator.AutoTokenizer")
    @patch("scripts.evaluator.AutoModelForCausalLM")
    def test_load_llm(self, mock_model_cls, mock_tokenizer_cls):
        """Test LLM loading with correct return order (model, tokenizer)."""
        # Arrange
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance

        # Act
        model, tokenizer = load_llm()

        # Assert
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        # Verify model comes first, tokenizer second
        self.assertEqual(model, mock_model_instance)
        self.assertEqual(tokenizer, mock_tokenizer_instance)
        # Verify pad_token was set
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_load_model_alias(self):
        """Test that load_model is an alias for load_llm."""
        with patch("scripts.evaluator.load_llm") as mock_load_llm:
            mock_load_llm.return_value = (self.mock_model, self.mock_tokenizer)

            model, tokenizer = load_model()

            mock_load_llm.assert_called_once()
            self.assertEqual(model, self.mock_model)
            self.assertEqual(tokenizer, self.mock_tokenizer)

    @patch("scripts.evaluator.HuggingFaceEmbedding")
    def test_load_embedding_model(self, mock_hf_embedding):
        """Test embedding model loading using HuggingFaceEmbedding."""
        # Arrange
        mock_embedding_instance = MagicMock()
        mock_hf_embedding.return_value = mock_embedding_instance

        # Act
        model = load_embedding_model()

        # Assert
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_embedding_instance)
        mock_hf_embedding.assert_called_once()

    def test_run_llm_query_with_generate(self):
        """Test LLM query with generative model."""
        # Arrange
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        mock_tokenizer.decode.return_value = "Generated answer"
        query = "What is the largest country?"

        # Act
        result = run_llm_query(query, mock_model, mock_tokenizer)

        # Assert
        self.assertEqual(result, "Generated answer")
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()

    def test_run_llm_query_without_generate(self):
        """Test LLM query with masked language model (no generate method)."""
        # Arrange
        mock_model = MagicMock()
        del mock_model.generate  # Remove generate method
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = np.array([1.0, 2.0])
        mock_model.return_value = mock_outputs
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        query = "Test query"

        # Act
        result = run_llm_query(query, mock_model, mock_tokenizer)

        # Assert
        self.assertIsInstance(result, str)
        self.assertIn("[1. 2.]", result)  # String representation of numpy array

    @patch("scripts.evaluator.load_embedding_model")
    @patch("scripts.evaluator.get_text_embedding")
    @patch("scripts.evaluator.calculate_similarity")
    def test_compare_answers_with_embeddings(self, mock_calc_sim, mock_get_emb, mock_load_emb):
        """Test answer comparison using embeddings with new similarity calculator."""
        # Arrange
        mock_load_emb.return_value = self.mock_embedding_model
        mock_get_emb.side_effect = [
            np.array([1.0, 2.0]),  # query embedding
            np.array([1.0, 1.8]),  # original answer embedding
            np.array([1.1, 2.1])  # optimized answer embedding
        ]
        mock_calc_sim.side_effect = [0.8, 0.9]  # original_sim, optimized_sim

        # Act
        result = compare_answers_with_embeddings("query", "answer1", "answer2", SimilarityMethod.COSINE)

        # Assert
        self.assertEqual(result, "Optimized")
        self.assertEqual(mock_calc_sim.call_count, 2)

    @patch("scripts.evaluator.load_embedding_model")
    @patch("scripts.evaluator.get_text_embedding")
    @patch("scripts.evaluator.calculate_similarity")
    def test_compare_answers_tie_detection(self, mock_calc_sim, mock_get_emb, mock_load_emb):
        """Test tie detection in answer comparison."""
        # Arrange
        mock_load_emb.return_value = self.mock_embedding_model
        mock_get_emb.side_effect = [np.array([1.0, 2.0])] * 3
        mock_calc_sim.side_effect = [0.85, 0.8505]  # Very close scores

        # Act
        result = compare_answers_with_embeddings("query", "answer1", "answer2")

        # Assert
        self.assertEqual(result, "Tie")

    @patch("scripts.evaluator.compare_answers_with_embeddings")
    def test_multi_method_comparison(self, mock_compare):
        """Test multi-method comparison functionality."""
        # Arrange
        mock_compare.side_effect = ["Optimized", "Original", "Optimized"]  # 2 optimized, 1 original
        methods = [SimilarityMethod.COSINE, SimilarityMethod.DOT_PRODUCT, SimilarityMethod.EUCLIDEAN]

        # Act
        result = multi_method_comparison("query", "answer1", "answer2", methods)

        # Assert
        self.assertEqual(result["consensus"], "Optimized")
        self.assertEqual(result["confidence"], 2 / 3)  # 2 out of 3 votes
        self.assertEqual(result["votes"]["Optimized"], 2)
        self.assertEqual(result["total_methods"], 3)

    @patch("scripts.evaluator.load_llm")
    def test_judge_with_llm_success(self, mock_load_llm):
        """Test LLM judgment with successful response."""
        # Arrange
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_llm.return_value = (mock_model, mock_tokenizer)

        # Fix: Mock the tokenizer call properly
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs
        mock_inputs_with_to = MagicMock()
        mock_inputs_with_to.to.return_value = mock_inputs
        # Override the tokenizer return to have a .to() method
        mock_tokenizer.return_value = mock_inputs_with_to

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Optimized"
        mock_model.device = torch.device("cpu")

        # Act
        result = judge_with_llm("query", "answer1", "answer2")

        # Assert
        self.assertEqual(result, "Optimized")

    @patch("scripts.evaluator.load_llm")
    def test_judge_with_llm_partial_match(self, mock_load_llm):
        """Test LLM judgment with partial word matching."""
        # Arrange
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_llm.return_value = (mock_model, mock_tokenizer)

        # Fix: Mock the tokenizer call properly
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_inputs_with_to = MagicMock()
        mock_inputs_with_to.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs_with_to

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "The optimized answer is better"
        mock_model.device = torch.device("cpu")

        # Act
        result = judge_with_llm("query", "answer1", "answer2")

        # Assert
        self.assertEqual(result, "Optimized")

    @patch("scripts.evaluator.load_llm")
    def test_judge_with_llm_error_handling(self, mock_load_llm):
        """Test LLM judgment error handling."""
        # Arrange
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_llm.return_value = (mock_model, mock_tokenizer)
        mock_model.generate.side_effect = Exception("GPU error")

        # Act
        result = judge_with_llm("query", "answer1", "answer2")

        # Assert
        self.assertEqual(result, "Tie")  # Should fallback to Tie on error

    @patch("scripts.evaluator.get_text_embedding")
    @patch("scripts.evaluator.calculate_similarities")
    def test_enumerate_top_documents_with_vectors(self, mock_calc_sims, mock_get_emb):
        """Test document enumeration with vector conversion."""
        # Arrange
        mock_node = MagicMock()
        mock_node.get_content.return_value = "Document content for testing"
        mock_node_with_score = MagicMock()
        mock_node_with_score.node = mock_node
        mock_node_with_score.score = 0.95

        self.mock_index.as_retriever.return_value.retrieve.return_value = [mock_node_with_score]

        mock_get_emb.side_effect = [
            np.array([1.0, 2.0]),  # query vector
            np.array([1.1, 2.1])  # document vector
        ]
        mock_calc_sims.return_value = np.array([0.92])

        # Act
        result = enumerate_top_documents(
            0, 1, "test query", self.mock_index, self.mock_embedding_model,
            top_k=1, convert_to_vector=True, similarity_method=SimilarityMethod.COSINE
        )

        # Assert
        self.assertEqual(result["query"], "test query")
        self.assertEqual(result["similarity_method"], "SimilarityMethod.COSINE")
        self.assertEqual(len(result["top_documents"]), 1)
        self.assertEqual(result["top_documents"][0]["custom_score"], 0.92)
        self.assertTrue(result["convert_to_vector"])

    def test_enumerate_top_documents_without_vectors(self):
        """Test document enumeration without vector conversion (fallback mode)."""
        # Arrange
        mock_node = MagicMock()
        mock_node.get_content.return_value = "Document content"
        mock_node_with_score = MagicMock()
        mock_node_with_score.node = mock_node
        mock_node_with_score.score = 0.85

        self.mock_index.as_retriever.return_value.retrieve.return_value = [mock_node_with_score]

        # Act
        result = enumerate_top_documents(
            0, 1, "test query", self.mock_index, self.mock_embedding_model,
            convert_to_vector=False
        )

        # Assert
        self.assertEqual(result["query"], "test query")
        self.assertEqual(len(result["top_documents"]), 1)
        self.assertEqual(result["top_documents"][0]["similarity_method"], "llamaindex_default")
        self.assertEqual(result["top_documents"][0]["original_score"], 0.85)
        self.assertIsNone(result["top_documents"][0]["custom_score"])

    def test_enumerate_top_documents_empty_results(self):
        """Test document enumeration with no results."""
        # Arrange
        self.mock_index.as_retriever.return_value.retrieve.return_value = []

        # Act
        result = enumerate_top_documents(
            0, 1, "empty query", self.mock_index, self.mock_embedding_model
        )

        # Assert
        self.assertEqual(result["query"], "empty query")
        self.assertEqual(len(result["top_documents"]), 0)
        self.assertEqual(result["total_documents"], 0)

    @patch("scripts.evaluator.get_text_embedding")
    @patch("scripts.evaluator.calculate_similarity")
    def test_hill_climb_documents_with_vectors(self, mock_calc_sim, mock_get_emb):
        """Test hill climbing with vector-based similarity scoring."""
        # Arrange
        mock_node = MagicMock()
        mock_node.get_content.return_value = "Context for testing"
        mock_node_with_score = MagicMock()
        mock_node_with_score.node = mock_node

        self.mock_index.as_retriever.return_value.retrieve.return_value = [mock_node_with_score]

        mock_llm_model = MagicMock()
        mock_llm_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        mock_tokenizer.decode.return_value = "Generated answer"

        # Fix: Use a more robust mocking approach that always returns valid embeddings
        mock_get_emb.return_value = np.array([1.0, 2.0])  # Always return valid embedding
        mock_calc_sim.return_value = 0.88

        # Act
        result = hill_climb_documents(
            0, 1, "test query", self.mock_index, mock_llm_model, mock_tokenizer,
            self.mock_embedding_model, convert_to_vector=True,
            similarity_method=SimilarityMethod.COSINE
        )

        # Assert
        self.assertEqual(result["query"], "test query")
        self.assertEqual(result["answer"], "Generated answer")  # Should find valid answer
        self.assertEqual(result["context"], "Context for testing")
        self.assertEqual(result["method_info"]["method"], "SimilarityMethod.COSINE")
        self.assertEqual(result["method_info"]["score"], 0.88)
        # Verify that embedding and similarity calculation were called
        self.assertGreaterEqual(mock_get_emb.call_count, 2)  # Called for query and answer
        mock_calc_sim.assert_called_once()

    def test_hill_climb_documents_without_vectors(self):
        """Test hill climbing with word overlap scoring (fallback mode)."""
        # Arrange
        mock_node = MagicMock()
        mock_node.get_content.return_value = "Context content"
        mock_node_with_score = MagicMock()
        mock_node_with_score.node = mock_node

        self.mock_index.as_retriever.return_value.retrieve.return_value = [mock_node_with_score]

        mock_llm_model = MagicMock()
        mock_llm_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        mock_tokenizer.decode.return_value = "machine learning answer"

        # Act
        result = hill_climb_documents(
            0, 1, "machine learning query", self.mock_index, mock_llm_model,
            mock_tokenizer, self.mock_embedding_model, convert_to_vector=False
        )

        # Assert
        self.assertEqual(result["query"], "machine learning query")
        self.assertEqual(result["answer"], "machine learning answer")
        self.assertEqual(result["method_info"]["method"], "word_overlap")
        # Should have some overlap score > 0 due to "machine learning" in both
        self.assertGreater(result["method_info"]["score"], 0)

    def test_hill_climb_documents_no_contexts(self):
        """Test hill climbing with no retrieved contexts."""
        # Arrange
        self.mock_index.as_retriever.return_value.retrieve.return_value = []

        # Act
        result = hill_climb_documents(
            0, 1, "empty query", self.mock_index, MagicMock(), MagicMock(),
            self.mock_embedding_model
        )

        # Assert
        self.assertEqual(result["query"], "empty query")
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["context"])

    @patch("scripts.evaluator.run_llm_query")
    @patch("scripts.evaluator.multi_method_comparison")
    @patch("scripts.evaluator.judge_with_llm")
    @patch("scripts.evaluator.load_llm")
    def test_advanced_sanity_check_success(self, mock_load_llm, mock_judge, mock_multi, mock_run_query):
        """Test advanced sanity check with successful execution."""
        # Arrange
        mock_load_llm.return_value = (self.mock_model, self.mock_tokenizer)
        mock_run_query.side_effect = ["Original answer", "Optimized answer"]
        mock_multi.return_value = {
            "consensus": "Optimized",
            "confidence": 0.8,
            "individual_results": {"cosine": "Optimized", "dot_product": "Original"}
        }
        mock_judge.return_value = "Optimized"

        # Act
        result = advanced_sanity_check("query1", "query2", MagicMock())

        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["final_decision"], "Optimized")
        self.assertEqual(result["original_query"], "query1")
        self.assertEqual(result["optimized_query"], "query2")

    @patch("scripts.evaluator.load_llm")
    def test_advanced_sanity_check_error_handling(self, mock_load_llm):
        """Test advanced sanity check error handling."""
        # Arrange
        mock_load_llm.side_effect = Exception("Model loading failed")

        # Act
        result = advanced_sanity_check("query1", "query2", MagicMock())

        # Assert
        self.assertFalse(result["success"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)