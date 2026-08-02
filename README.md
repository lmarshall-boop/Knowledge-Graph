# Neurodegenerative Disease Knowledge Graph Explorer

A small NLP tool that pulls real research abstracts from PubMed, extracts
subject-verb-object relationships using spaCy's dependency parser, and
visualizes them as a knowledge graph — built as a self-directed exploration
of applying NLP to neurodegenerative disease research.

## What it does
1. Takes a search topic (e.g. "Alzheimer's disease amyloid microglia")
2. Pulls matching abstracts live from PubMed via NCBI's Entrez API
3. Extracts simple relationship triples like `(amyloid plaques, disrupt, neuronal function)`
4. Renders the relationships as an interactive knowledge graph

## Try it
Live app: *(add your Streamlit Community Cloud link here once deployed)*

## Run it yourself
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
- **Data source:** NCBI's free PubMed E-utilities API, via Biopython's `Entrez` module
- **NLP:** spaCy's `en_core_web_sm` dependency parser for triple extraction
- **Graph:** NetworkX for graph structure, Matplotlib for visualization
- **Interface:** Streamlit, deployed on Streamlit Community Cloud

## Known limitations
The subject-verb-object extraction is a beginner-level heuristic based on
dependency parsing, not a trained relation-extraction model — it misses
complex or passive-voice sentences and occasionally produces a noisy triple.
Improving this (e.g. with coreference resolution, or a proper relation
extraction model) would be a natural next step.

## Motivation
Built while exploring the intersection of NLP, knowledge graphs, and
neurodegenerative disease research.
