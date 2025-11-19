import os
import re
import string

import datasets

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the dataset from HuggingFace
dataset_name = "ojoos/pubmed_abstracts"
pubmedqa_splits = {
    "pqa_unlabeled": "pqa_unlabeled",
    "pqa_artificial": "pqa_artificial",
    "pqa_labeled": "pqa_labeled",
}

#  load all datasets into one dataset
datasets_dict = {}
for config_name in pubmedqa_splits.values():
    datasets_dict[config_name] = datasets.load_dataset("qiaojin/PubMedQA", config_name)
    print(f"Dataset '{dataset_name}' with config {config_name} loaded successfully.")

"""## Final Task

### Subtask:
Present the extracted title and abstract to the user.

## Summary:

### Data Analysis Key Findings
*   The PubMed ID `25429730` was successfully extracted from the 'pubid' field of the first entry in the 'train' split of the `benchmark` (qiaojin/PubMedQA) dataset.
*   Using the extracted PubMed ID, the following article title was retrieved from PubMed: "Group 2 innate lymphoid cells (ILC2s) are increased in chronic rhinosinusitis with nasal polyps or eosinophilia."
*   The corresponding abstract for the article was also successfully retrieved and displayed.

### Insights or Next Steps
*   The retrieved title and abstract provide crucial context for understanding the content of the article associated with the given PubMed ID, which can be valuable for downstream tasks such as research summarization or classification within the PubMedQA dataset.
*   For a production system, implementing robust error handling and retry mechanisms for the `Entrez` queries would be beneficial to manage potential network issues or invalid PubMed IDs.
"""


def normalize_text(text):
    """Normalizes text by lowercasing, removing punctuation, and stripping extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


"""# Task
Extract all unique PubMed IDs (PMIDs) from the 'train' split of the `qiaojin/PubMedQA` dataset, then use these PMIDs to fetch article titles and abstracts from PubMed in batches. Finally, process the fetched data and save it into multiple CSV files, each containing columns for PMID, Title, and Abstract.

## Extract All PMIDs from PubMedQA

### Subtask:
Iterate through the 'train' split of the `benchmark` (qiaojin/PubMedQA) and collect all unique 'pubid' values into a list.

**Reasoning**:
To extract all unique PubMed IDs, I will iterate through the 'train' split of `benchmark`, add each 'pubid' to a set to ensure uniqueness, convert the set to a list, and then print the total count of unique IDs.
"""

unique_pubmed_ids = set()

for config_name, benchmark in datasets_dict.items():
    unique_set_ids = set()
    for entry in benchmark['train']:
        unique_pubmed_ids.add(entry['pubid'])
        unique_set_ids.add(entry['pubid'])
    print(f"Extracted PubMed IDs from dataset split: {config_name}. Total unique IDs added: {len(unique_set_ids)}")
unique_pubmed_ids_list = list(unique_pubmed_ids)

print(f"Total number of unique PubMed IDs found: {len(unique_pubmed_ids_list)}")

"""## Fetch Article Details from PubMed in Batches

### Subtask:
Use the collected unique PubMed IDs to fetch article titles and abstracts from PubMed. Implement batch processing to efficiently query the Entrez database and handle potential rate limits or connection issues.

## Fetch Article Details from PubMed in Batches

### Subtask:
Use the collected unique PubMed IDs to fetch article titles and abstracts from PubMed. Implement batch processing to efficiently query the Entrez database and handle potential rate limits or connection issues.

**Reasoning**:
To avoid overwhelming the NCBI Entrez servers and adhere to their usage guidelines, it's essential to implement batch processing with a reasonable delay between requests. I will define a `batch_size` and a `delay` between batches. The code will then iterate through the `unique_pubmed_ids_list` in batches, fetch data for each batch, and collect the results.

**Reasoning**:
To efficiently retrieve details for all unique PubMed IDs, I will implement batch processing using `Entrez.efetch`. This involves iterating through the IDs in batches, making API calls, parsing the XML responses, extracting the PMID, title, and abstract for each article, and storing them. A delay will be introduced between batches to respect NCBI's usage policies.
"""

import time
from Bio import Entrez

# Ensure Entrez email is set from previous step, or set it here if running standalone
if not hasattr(Entrez, 'email') or not Entrez.email:
    Entrez.email = "bardamri1702@gmail.com"  # IMPORTANT: Replace with your actual email address

# Define batch size and delay
batch_size = 500  # Number of PubMed IDs to fetch per batch
delay = 0.5  # Delay in seconds between batches to avoid overloading the server

all_articles_data = []

print(f"Starting to fetch {len(unique_pubmed_ids_list)} articles from PubMed in batches of {batch_size}...")

# Iterate through unique_pubmed_ids_list in batches
for i in range(0, len(unique_pubmed_ids_list), batch_size):
    batch_ids = unique_pubmed_ids_list[i: i + batch_size]
    ids_string = ",".join(map(str, batch_ids))

    try:
        handle = Entrez.efetch(db="pubmed", id=ids_string, retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        if 'PubmedArticle' in records:
            for pubmed_article in records['PubmedArticle']:
                pmid = None
                title = "N/A"
                abstract = "N/A"

                # Extract PMID
                if 'MedlineCitation' in pubmed_article and 'PMID' in pubmed_article['MedlineCitation']:
                    pmid = str(pubmed_article['MedlineCitation']['PMID'])

                # Extract Title and Abstract
                if 'MedlineCitation' in pubmed_article and 'Article' in pubmed_article['MedlineCitation']:
                    article_info = pubmed_article['MedlineCitation']['Article']

                    if 'ArticleTitle' in article_info:
                        title = str(article_info['ArticleTitle'])

                    if 'Abstract' in article_info and 'AbstractText' in article_info['Abstract']:
                        # AbstractText can be a list of strings if there are multiple paragraphs
                        abstract = " ".join(article_info['Abstract']['AbstractText'])

                if pmid:
                    all_articles_data.append({
                        'PMID': pmid,
                        'Title': title,
                        'Abstract': abstract
                    })

        print(
            f"Processed batch {i // batch_size + 1}/{(len(unique_pubmed_ids_list) + batch_size - 1) // batch_size}. Fetched {len(all_articles_data)} articles so far.")

    except Exception as e:
        print(f"Error fetching batch starting with ID {batch_ids[0]}: {e}")
        # Optionally, save partial results here or retry specific IDs

    time.sleep(delay)

print(f"Finished fetching. Total articles retrieved: {len(all_articles_data)}")

"""## Process and Save Fetched Data into Multiple CSV Files

### Subtask:
Convert the collected article data into a pandas DataFrame and save it into multiple CSV files. Each CSV file should contain columns for PMID, Title, and Abstract. Implement a strategy to split the data across multiple files to manage file size or for easier processing.

**Reasoning**:
First, I will convert the `all_articles_data` list of dictionaries into a pandas DataFrame. This will provide a structured table format, which is easier to manipulate and save to CSV.
"""

import pandas as pd

df_articles = pd.DataFrame(all_articles_data)

print(f"DataFrame created with {len(df_articles)} rows and {len(df_articles.columns)} columns.")
print(df_articles.head())

"""**Reasoning**:
To manage file sizes and adhere to the subtask requirement of saving to multiple CSV files, I will first determine a suitable `chunk_size` for splitting the DataFrame. Then, I will iterate through the DataFrame in chunks and save each chunk to a separate CSV file, indicating the file name and the number of rows saved.


"""

import numpy as np

# Define the number of rows per CSV file
chunk_size = 50000

# Calculate the number of files needed
num_files = int(np.ceil(len(df_articles) / chunk_size))

print(f"Saving {len(df_articles)} articles into {num_files} CSV files...")

data_path = os.path.join(PROJECT_PATH, "data", "pubmed_selected_articles", "train")
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
os.chdir(data_path)
# Iterate and save to multiple CSV files
for i in range(num_files):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size
    df_chunk = df_articles.iloc[start_idx:end_idx]

    output_filename = os.path.join(data_path, f"pubmed_articles_part_{i + 1}.csv")
    df_chunk.to_csv(output_filename, index=False)

    print(f"Saved {len(df_chunk)} rows to {output_filename}")

print("All articles saved successfully into multiple CSV files.")

"""## Final Task

### Subtask:
Summarize the successful download and storage of the PubMed metadata, including the number of CSV files generated and their approximate location.

## Summary:

### Data Analysis Key Findings
*   A total of 211,269 unique PubMed IDs were extracted from the 'train' split of the `qiaojin/PubMedQA` dataset.
*   Batch processing was implemented to fetch article details from PubMed, using a batch size of 500 IDs and a delay of 0.5 seconds between requests.
*   Out of the extracted IDs, 211,049 articles were successfully retrieved, including their PMID, Title, and Abstract.
*   The fetched data was organized into a pandas DataFrame and then split into 5 CSV files, each containing approximately 50,000 rows, saved in the current working directory. The last file contained 11,049 rows.

### Insights or Next Steps
*   The successful implementation of batch processing demonstrates an effective strategy for handling large-scale API requests while adhering to service provider guidelines, which is crucial for data collection from external sources like NCBI Entrez.
*   The structured data (PMID, Title, Abstract) saved in multiple CSV files is now ready for further analysis, such as text mining, topic modeling, or building a search index.
"""
