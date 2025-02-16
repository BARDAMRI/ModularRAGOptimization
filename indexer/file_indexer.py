# indexer/file_indexer.py
from .file_manager import list_text_files, read_file
from config import DATA_DIR


class FileIndexer:
    """
    This class builds an index of files found in a specified directory.
    The index is stored as a dictionary mapping file names to their contents.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.index = {}  # Dictionary to store the index: {file_name: content}

    def build_index(self):
        """
        Build the index by reading all text files in the data directory.

        Returns:
            dict: The built index.
        """
        files = list_text_files(self.data_dir)
        for file_path in files:
            content = read_file(file_path)
            # Here you might add additional processing (e.g., tokenization or summarization)
            file_name = file_path.split(os.sep)[-1]
            self.index[file_name] = content
        return self.index

    def get_index(self):
        """
        Return the current index.

        Returns:
            dict: The current index.
        """
        return self.index
