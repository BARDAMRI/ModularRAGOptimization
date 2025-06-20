# Python
import unittest
from unittest.mock import patch, MagicMock
import sys
import termios
from main import flush_input, run_query_evaluation, run_analysis


class TestMainFunctions(unittest.TestCase):
    @patch("termios.tcflush")
    def test_flush_input(self, mock_tcflush):
        # Test that flush_input calls termios.tcflush
        flush_input()
        mock_tcflush.assert_called_once_with(sys.stdin, termios.TCIFLUSH)

    @patch("main.load_model")
    @patch("main.load_vector_db")
    @patch("main.input", side_effect=["y", "e", "exit"])
    @patch("main.HuggingFaceEmbedding")
    @patch("main.ResultsLogger")
    @patch("main.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"question": "Sample question"}\n')
    def test_run_query_evaluation_enumeration_mode(
            self, mock_open, mock_path_exists, mock_results_logger, mock_embedding, mock_input, mock_load_vector_db,
            mock_load_model
    ):
        # Mock dependencies
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_load_vector_db.return_value = MagicMock()
        mock_embedding.return_value = MagicMock()
        mock_results_logger.return_value = MagicMock()

        # Run the function
        run_query_evaluation()

        # Assert that the logger logged results
        mock_results_logger.return_value.log.assert_called()

    @patch("main.load_model")
    @patch("main.load_vector_db")
    @patch("main.input", side_effect=["n", "exit"])
    @patch("main.query_model")
    def test_run_query_evaluation_query_mode(
            self, mock_query_model, mock_input, mock_load_vector_db, mock_load_model
    ):
        # Mock dependencies
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_load_vector_db.return_value = MagicMock()
        mock_query_model.return_value = {"error": None, "question": "Sample question", "answer": "Sample answer"}

        # Run the function
        run_query_evaluation()

        # Assert that query_model was called
        mock_query_model.assert_called()

    @patch("main.ResultsLogger")
    @patch("main.plot_score_distribution")
    def test_run_analysis(self, mock_plot_score_distribution, mock_results_logger):
        # Mock dependencies
        mock_results_logger.return_value = MagicMock()

        # Run the function
        run_analysis()

        # Assert that the logger summarized scores
        mock_results_logger.return_value.summarize_scores.assert_called_once()

        # Assert that the score distribution was plotted
        mock_plot_score_distribution.assert_called_once()


if __name__ == "__main__":
    unittest.main()
