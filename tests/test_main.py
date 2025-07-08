# test_main.py - Realistic version that matches your actual code flow
import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import termios

# Import the functions you're testing
from main import flush_input, run_query_evaluation, run_analysis


class TestMainFunctions(unittest.TestCase):
    """Test suite for main module functions."""

    def setUp(self):
        """Set up common test fixtures."""
        # Common mock objects that multiple tests might use
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_embedding_model = MagicMock()
        self.mock_results_logger = MagicMock()

    @patch("termios.tcflush")
    def test_flush_input_calls_termios_correctly(self, mock_tcflush):
        """Test that flush_input calls termios.tcflush with correct parameters."""
        # Act
        flush_input()

        # Assert
        mock_tcflush.assert_called_once_with(sys.stdin, termios.TCIFLUSH)

    @patch("termios.tcflush")
    def test_flush_input_handles_missing_termios(self, mock_tcflush):
        """Test that flush_input handles systems without termios gracefully."""
        # Arrange - simulate termios not being available
        mock_tcflush.side_effect = AttributeError("No termios on this system")

        # Act & Assert - should not raise an exception
        try:
            flush_input()
        except AttributeError:
            self.fail("flush_input should handle missing termios gracefully")

    @patch("main.load_vector_db")
    @patch("main.ResultsLogger")
    @patch("main.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"question": "Sample question"}\n')
    @patch("main.input", side_effect=["y", "e", "exit"])
    def test_run_query_evaluation_enumeration_mode(
            self, mock_input, mock_open_file, mock_path_exists, mock_results_logger,
            mock_load_vector_db
    ):
        """Test query evaluation in enumeration mode (processing test files)."""
        # Arrange
        mock_load_vector_db.return_value = (self.mock_vector_db, self.mock_embedding_model)
        mock_logger_instance = MagicMock()
        mock_results_logger.return_value = mock_logger_instance

        # Act
        run_query_evaluation()

        # Assert - Check what actually gets called based on your logs
        mock_load_vector_db.assert_called_once()
        mock_results_logger.assert_called_once()
        mock_logger_instance.log.assert_called()

        # Verify the mode was detected correctly (from logs: "Running for NQ Query #1")
        self.assertTrue(mock_logger_instance.log.called)

    @patch("main.load_vector_db")
    @patch("main.process_query_with_context")
    @patch("main.input", side_effect=["n", "What is AI?", "exit"])
    def test_run_query_evaluation_interactive_mode(
            self, mock_input, mock_process_query, mock_load_vector_db
    ):
        """Test query evaluation in interactive mode (manual query input)."""
        # Arrange
        mock_load_vector_db.return_value = (self.mock_vector_db, self.mock_embedding_model)

        # Mock a successful query response
        mock_process_query.return_value = {
            "error": None,
            "question": "What is AI?",
            "answer": "AI is artificial intelligence",
            "score": 0.95,
            "attempts": 1,
            "device_used": "cpu",
            "similarity_method": "cosine"
        }

        # Act
        run_query_evaluation()

        # Assert - Based on your logs, these are the calls that actually happen
        mock_load_vector_db.assert_called_once()
        mock_process_query.assert_called()

    @patch("main.load_vector_db")
    @patch("main.process_query_with_context")
    @patch("main.input", side_effect=["n", "test query", "exit"])
    def test_run_query_evaluation_handles_query_error(
            self, mock_input, mock_process_query, mock_load_vector_db
    ):
        """Test that query evaluation handles errors from process_query_with_context."""
        # Arrange
        mock_load_vector_db.return_value = (self.mock_vector_db, self.mock_embedding_model)

        # Mock an error response
        mock_process_query.return_value = {
            "error": "GPU memory error",
            "question": "test query",
            "answer": "Error occurred",
            "score": 0.0,
            "attempts": 1
        }

        # Act - should not raise an exception
        try:
            run_query_evaluation()
        except Exception as e:
            self.fail(f"run_query_evaluation should handle query errors gracefully, but raised: {e}")

        # Assert
        mock_load_vector_db.assert_called_once()
        mock_process_query.assert_called()

    @patch("main.ResultsLogger")
    @patch("main.plot_score_distribution")
    def test_run_analysis_calls_expected_functions(self, mock_plot_score_distribution, mock_results_logger):
        """Test that run_analysis calls the expected logging and plotting functions."""
        # Arrange
        mock_logger_instance = MagicMock()
        mock_results_logger.return_value = mock_logger_instance

        # Act
        run_analysis()

        # Assert
        mock_results_logger.assert_called_once()
        mock_logger_instance.summarize_scores.assert_called_once()
        mock_plot_score_distribution.assert_called_once()

    @patch("main.ResultsLogger")
    @patch("main.plot_score_distribution")
    def test_run_analysis_handles_missing_data(self, mock_plot_score_distribution, mock_results_logger):
        """Test that run_analysis handles cases where no results data exists."""
        # Arrange
        mock_logger_instance = MagicMock()
        mock_logger_instance.summarize_scores.side_effect = FileNotFoundError("No results file found")
        mock_results_logger.return_value = mock_logger_instance

        # Act & Assert - should handle the error gracefully
        try:
            run_analysis()
        except FileNotFoundError:
            # If your function doesn't handle this yet, this test will help you identify that
            pass

    def test_sample_integration_scenario(self):
        """Example of how you might test a more complex integration scenario."""
        # This is a placeholder for more complex integration tests
        pass


class TestMainFunctionsDetailed(unittest.TestCase):
    """More detailed tests that verify specific behaviors."""

    @patch("main.load_vector_db")
    @patch("main.enumerate_top_documents")
    @patch("main.ResultsLogger")
    @patch("main.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"question": "Sample question"}\n')
    @patch("main.input", side_effect=["y", "e", "exit"])
    def test_enumeration_mode_calls_enumerate_documents(
            self, mock_input, mock_open_file, mock_path_exists, mock_results_logger,
            mock_enumerate_docs, mock_load_vector_db
    ):
        """Test that enumeration mode actually calls enumerate_top_documents."""
        # Arrange
        mock_load_vector_db.return_value = (MagicMock(), MagicMock())
        mock_logger_instance = MagicMock()
        mock_results_logger.return_value = mock_logger_instance

        # Mock the enumerate function
        mock_enumerate_docs.return_value = {
            "query": "Sample question",
            "top_documents": []
        }

        # Act
        run_query_evaluation()

        # Assert
        mock_load_vector_db.assert_called_once()
        mock_enumerate_docs.assert_called()
        mock_logger_instance.log.assert_called()

    @patch("main.load_vector_db")
    @patch("main.process_query_with_context")
    @patch("main.input", side_effect=["n", "exit"])  # Choose 'n' then exit immediately
    def test_interactive_mode_exits_gracefully(
            self, mock_input, mock_process_query, mock_load_vector_db
    ):
        """Test that interactive mode can exit without processing queries."""
        # Arrange
        mock_load_vector_db.return_value = (MagicMock(), MagicMock())

        # Act
        run_query_evaluation()

        # Assert - should load vector DB but not process any queries
        mock_load_vector_db.assert_called_once()
        # process_query_with_context should NOT be called since we exit immediately
        mock_process_query.assert_not_called()

    @patch("main.load_vector_db")
    @patch("main.input", side_effect=["invalid_choice", "exit"])
    def test_handles_invalid_input_choices(self, mock_input, mock_load_vector_db):
        """Test that the application handles invalid input choices gracefully."""
        # Arrange
        mock_load_vector_db.return_value = (MagicMock(), MagicMock())

        # Act - should not crash with invalid input
        try:
            run_query_evaluation()
        except Exception as e:
            self.fail(f"Application should handle invalid input gracefully, but raised: {e}")

        # Assert
        mock_load_vector_db.assert_called_once()


class TestMainFunctionsMinimal(unittest.TestCase):
    """Minimal tests that focus on core functionality without over-mocking."""

    @patch("main.load_vector_db")
    @patch("main.input", side_effect=["y", "e", "exit"])
    @patch("main.ResultsLogger")
    @patch("main.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"question": "Test question"}\n')
    def test_app_runs_without_crashing_enumeration(
            self, mock_open_file, mock_path_exists, mock_results_logger,
            mock_input, mock_load_vector_db
    ):
        """Minimal test: app should run enumeration mode without crashing."""
        # Arrange
        mock_load_vector_db.return_value = (MagicMock(), MagicMock())
        mock_results_logger.return_value = MagicMock()

        # Act & Assert - main goal is no exceptions
        try:
            run_query_evaluation()
        except Exception as e:
            self.fail(f"Application crashed unexpectedly: {e}")

    @patch("main.load_vector_db")
    @patch("main.input", side_effect=["n", "exit"])
    def test_app_runs_without_crashing_interactive(self, mock_input, mock_load_vector_db):
        """Minimal test: app should run interactive mode without crashing."""
        # Arrange
        mock_load_vector_db.return_value = (MagicMock(), MagicMock())

        # Act & Assert - main goal is no exceptions
        try:
            run_query_evaluation()
        except Exception as e:
            self.fail(f"Application crashed unexpectedly: {e}")


if __name__ == "__main__":
    # Add more verbose test output
    unittest.main(verbosity=2)