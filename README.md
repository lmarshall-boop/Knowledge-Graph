# Neurodegenerative Disease Knowledge Graph Explorer

An NLP tool that pulls real research abstracts from PubMed, extracts typed
subject-verb-object relationships using spaCy's dependency parser, cross-references
active ClinicalTrials.gov studies, and visualizes it all as an interactive,
color-coded knowledge graph — built as a self-directed exploration of applying
NLP to neurodegenerative disease research.

## What it does
1. Pick a disease preset (Alzheimer's, Parkinson's, ALS, Huntington's, FTD, Lewy body dementia) or enter a custom PubMed search term
2. Pulls matching abstracts live from PubMed via NCBI's Entrez API, plus a plain-language Wikipedia overview and active ClinicalTrials.gov studies
3. Extracts relationship triples like `(amyloid-beta, disrupt, neuronal function)`, each typed by entity category (disease / gene-protein / drug-chemical / anatomy / process), flagged for negation ("X does *not* activate Y"), and linked back to its source PMID
4. Renders the relationships as an interactive, color-coded knowledge graph, with a searchable table, CSV/GraphML export, and the underlying source abstracts alongside the recruiting trials

## Try it
Live app: *(add your Streamlit Community Cloud link here once deployed)*

## Run it yourself
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
- **Data sources:** NCBI's free PubMed E-utilities API (via Biopython's `Entrez` + `Medline` modules), ClinicalTrials.gov's public API v2, and Wikipedia's REST summary API
- **NLP:** spaCy's `en_core_web_sm` dependency parser for triple extraction ([nlp_utils.py](nlp_utils.py)), with a curated neurodegenerative-disease lexicon for entity typing/synonym normalization and a negation detector
- **Graph:** NetworkX for graph structure, pyvis for the interactive rendering
- **Interface:** Streamlit, deployed on Streamlit Community Cloud

## Known limitations
The subject-verb-object extraction is a beginner-level heuristic based on
dependency parsing, not a trained relation-extraction model — it misses
complex or passive-voice sentences and occasionally produces a noisy triple.
Entity typing is lexicon-based rather than a trained biomedical NER model
(e.g. scispaCy), so anything outside the curated term list falls back to a
generic "OTHER" category. Improving either (a proper relation-extraction
model, or swapping in scispaCy for entity typing) would be a natural next step.

## Disclaimer
This is a research-exploration and portfolio project, not a medical or
diagnostic tool. Extracted relationships are computed automatically and may
contain NLP errors — always verify against the original source abstract.

## Motivation
Built while exploring the intersection of NLP, knowledge graphs, and
neurodegenerative disease research.
