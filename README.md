# LlamaIndex Project

This project is a modular implementation of a file indexing and search system using Python. It is designed with a clear separation between core logic and user interface, making it easy to integrate with GUI frameworks like PyQt5 in the future.

## Features

- **File Management:**  
  List and read `.txt` files from a specified directory.
- **Indexing:**  
  Build an index mapping file names to their contents.
- **Query Optimization:**  
  Contains a placeholder for query optimization. Extend or replace this functionality as needed.
- **Logging:**  
  Log key events and errors to a log file for easier debugging and monitoring.
- **Modular Design:**  
  Organized codebase with a clear folder structure, making it simple to modify, extend, or integrate with other systems.

## Project Structure
llama_index_project/
├── config.py           # Configuration settings for the project
├── main.py             # Main entry point for the application
├── indexer/            # Modules related to file indexing and query optimization
│   ├── file_manager.py # Functions for listing and reading files
│   ├── file_indexer.py # Builds and manages the file index
│   └── query_optimizer.py # Contains query optimization and search functions
└── utils/              # Utility modules for the project
    ├── __init__.py     # Package initializer for the utils module
    └── logger.py       # Logging configuration and helper functions

## Requirements

- Python 3.10.3 or later
- Standard Python libraries (no external dependencies required at the moment)

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/BARDAMRI/ModularRAGOptimization.git
   cd ModularRAGOptimization

2. **Set Up Your Data Directory:**
   Create a directory (default is ./data) and place your .txt files there. You can modify the directory path in config.py if needed.
   -   mkdir data
    -  Add your .txt files into the 'data' directory.


3. **Run the Application:**
   Execute the main script:
   ```python
   python main.py

   
