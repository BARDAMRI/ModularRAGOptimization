from Bio import Entrez

Entrez.email = "bardamri1702@gmail.com"
pmid = "23899370"

handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
record = Entrez.read(handle)
handle.close()

article = record["PubmedArticle"][0]

pmid = str(article["MedlineCitation"]["PMID"])
title = article["MedlineCitation"]["Article"].get("ArticleTitle", "N/A")
abstract_parts = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
abstract = " ".join(abstract_parts)

print("PMID:", pmid)
print("TITLE:", title)
print("ABSTRACT:", abstract)