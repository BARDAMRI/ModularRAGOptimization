# indexer/file_manager.py
import os


def list_text_files(directory):
    """
    List all text files in the given directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        list: A list of paths to text files.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            files.append(os.path.join(directory, file))
    return files


def read_file(file_path):
    """
    Read the content of a text file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content