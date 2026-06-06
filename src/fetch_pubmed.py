import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# ── PubMed Search Topics ───────────────────────────────────
QUERIES = [
    "diabetes mellitus treatment machine learning",
    "hypertension management clinical outcomes",
    "cancer immunotherapy clinical trials",
    "cardiovascular disease prediction AI",
    "mental health depression treatment outcomes",
    "COVID-19 long term effects clinical",
    "antibiotic resistance treatment outcomes",
    "obesity management pharmacological treatment",
]

MAX_PER_QUERY = 50  # Total ~400 papers

def search_pubmed(query: str, max_results: int = 50) -> list:
    """PubMed se paper IDs fetch karo."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db"      : "pubmed",
        "term"    : query,
        "retmax"  : max_results,
        "retmode" : "json",
        "sort"    : "relevance",
        "datetype": "pdat",
        "mindate" : "2020",
        "maxdate" : "2024",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_details(pmids: list) -> list:
    """Paper details fetch karo."""
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db"     : "pubmed",
        "id"     : ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    root    = ET.fromstring(r.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        try:
            # Title
            title_el = article.find(".//ArticleTitle")
            title    = title_el.text if title_el is not None else ""

            # Abstract
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (el.text or "") for el in abstract_parts
            ).strip()

            # PMID
            pmid_el = article.find(".//PMID")
            pmid    = pmid_el.text if pmid_el is not None else ""

            # Journal
            journal_el = article.find(".//Journal/Title")
            journal    = journal_el.text if journal_el is not None else ""

            # Year
            year_el = article.find(".//PubDate/Year")
            year    = year_el.text if year_el is not None else ""

            # Authors
            authors = []
            for author in article.findall(".//Author")[:3]:
                ln = author.find("LastName")
                fn = author.find("ForeName")
                if ln is not None:
                    name = ln.text
                    if fn is not None:
                        name += f" {fn.text}"
                    authors.append(name)
            authors_str = ", ".join(authors)

            if title and abstract and len(abstract) > 100:
                articles.append({
                    "pmid"    : pmid,
                    "title"   : title,
                    "abstract": abstract,
                    "journal" : journal,
                    "year"    : year,
                    "authors" : authors_str,
                    "url"     : f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        except Exception:
            continue

    return articles


def main():
    print("=" * 60)
    print("Medical RAG — PubMed Data Fetch")
    print("=" * 60)

    all_papers = []
    seen_pmids = set()

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] Query: {query[:50]}...")

        try:
            pmids = search_pubmed(query, MAX_PER_QUERY)
            # Remove duplicates
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            if not new_pmids:
                print("   No new papers found")
                continue

            papers = fetch_details(new_pmids)
            all_papers.extend(papers)
            print(f"   Fetched: {len(papers)} papers")
            time.sleep(1)  # Rate limit respect

        except Exception as e:
            print(f"   Error: {e}")
            continue

    # Save
    df = pd.DataFrame(all_papers).drop_duplicates(subset=["pmid"])
    df.to_csv(RAW / "pubmed_papers.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Total papers saved : {len(df)}")
    print(f"Topics covered     : {len(QUERIES)}")
    print(f"File               : data/raw/pubmed_papers.csv")
    print(f"Columns            : {df.columns.tolist()}")
    print("=" * 60)
    print("\nNext → src/build_index.py")


if __name__ == "__main__":
    main()