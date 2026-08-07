"""
Data source connectors for the Knowledge Graph Explorer.
------------------------------------------------------------
Three free, no-auth-required APIs, each answering a different question:

- PubMed (via Entrez)      -> "what does the research literature say?"
- ClinicalTrials.gov       -> "what active trials exist right now?"
- Wikipedia REST summary   -> "what's the plain-language context?"

Pulling from more than one source is what turns this from a single-query
NLP demo into something that actually helps someone orient themselves on
a disease: the literature graph, the trials that are currently recruiting,
and a layperson summary side by side.
"""

import urllib.parse

import requests
from Bio import Entrez, Medline

CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Wikimedia's robot policy rejects requests with no User-Agent (403), so every
# request identifies this app per https://w.wiki/4wJS.
_HEADERS = {"User-Agent": "NeuroKnowledgeGraphExplorer/1.0 (https://github.com/lmarshall-boop/Knowledge-Graph)"}


def fetch_pubmed_abstracts(query, max_results=15, email="your_email@example.com"):
    """Search PubMed and return {"abstracts": [...], "total_count": int}.

    `total_count` is PubMed's count of ALL matching records, not just the
    ones pulled -- it's what lets the UI honestly say "searching across
    12,482 PubMed records" instead of just "10 abstracts", since esearch
    reports the full match count regardless of retmax.

    Uses MEDLINE format (not a blank-line-split hack) so each record's PMID
    is parsed reliably -- that PMID is what lets the UI link every extracted
    relationship back to its actual source citation.
    """
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    total_count = int(record.get("Count", 0))

    if not ids:
        return {"abstracts": [], "total_count": total_count}

    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()

    abstracts = []
    for rec in records:
        abstract = rec.get("AB", "").strip()
        if len(abstract) < 200:
            continue
        abstracts.append({
            "pmid": rec.get("PMID", ""),
            "title": rec.get("TI", "(untitled)"),
            "abstract": abstract,
        })
    return {"abstracts": abstracts, "total_count": total_count}


def fetch_clinical_trials(query, max_results=10):
    """Return {"trials": [...], "total_count": int} of matching ClinicalTrials.gov studies.

    `total_count` (via countTotal=true) is the full count of registered
    studies matching the term, not just the page pulled here.
    """
    params = {
        "query.term": query,
        "pageSize": max_results,
        "countTotal": "true",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName",
    }
    try:
        resp = requests.get(CLINICAL_TRIALS_API, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return {"trials": [], "total_count": 0}

    payload = resp.json()
    trials = []
    for study in payload.get("studies", []):
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
        nct_id = ident.get("nctId", "")
        trials.append({
            "nct_id": nct_id,
            "title": ident.get("briefTitle", "(untitled)"),
            "status": status.get("overallStatus", "UNKNOWN"),
            "phase": ", ".join(design.get("phases", [])) or "N/A",
            "sponsor": sponsor.get("name", "N/A"),
            "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        })
    return {"trials": trials, "total_count": payload.get("totalCount", len(trials))}


def fetch_wikipedia_summary(term):
    """Return a plain-language summary string for `term`, or None."""
    title = urllib.parse.quote(term.replace(" ", "_"))
    try:
        resp = requests.get(WIKIPEDIA_SUMMARY_API.format(title), headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "title": data.get("title", term),
            "extract": data.get("extract", ""),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except requests.RequestException:
        return None
