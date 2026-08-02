"""
Neurodegenerative Disease Knowledge Graph Explorer
----------------------------------------------------
A small Streamlit app that pulls PubMed abstracts on a topic you choose,
extracts simple subject-verb-object relationships with spaCy, and draws
them as a knowledge graph.

Run locally with:  streamlit run app.py
Or deploy free at:  share.streamlit.io  (connect this file's GitHub repo)
"""

import streamlit as st
from Bio import Entrez
import spacy
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Setup ----
# NCBI requires you to identify yourself with an email address.
# Replace this with your own email before you deploy the app.
Entrez.email = "your_email@example.com"

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


def fetch_abstracts(query, max_results=15):
    """Search PubMed and return a list of abstract texts for the query."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]

    if not ids:
        return []

    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    raw = handle.read()
    handle.close()

    # PubMed returns all abstracts concatenated with blank lines between records;
    # this is a simple split, not perfect, but good enough for a first version.
    abstracts = [a.strip() for a in raw.split("\n\n") if len(a.strip()) > 200]
    return abstracts


def get_span_for_token(token):
    """Expand a single token to its full noun phrase, e.g. 'brain' -> 'the brain'."""
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            return chunk.text
    return token.text


def extract_triples(text, nlp):
    """Pull (subject, verb, object) triples out of text using spaCy's
    dependency parser. This is a beginner-friendly heuristic, not a
    full relation-extraction model -- it will miss some sentences and
    that's expected and fine to say out loud if asked about it."""
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        subj, verb, obj = None, None, None
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subj = token
                verb = token.head
            if token.dep_ in ("dobj", "pobj", "attr") and verb is not None:
                if token.head == verb or token.head.head == verb:
                    obj = token
        if subj is not None and verb is not None and obj is not None:
            triples.append((get_span_for_token(subj), verb.lemma_, get_span_for_token(obj)))
    return triples


def build_graph(triples):
    G = nx.DiGraph()
    for subj, verb, obj in triples:
        G.add_edge(subj, obj, label=verb)
    return G


# ---- UI ----
st.set_page_config(page_title="Neuro Knowledge Graph Explorer", layout="wide")
st.title("Neurodegenerative Disease Knowledge Graph Explorer")
st.write(
    "Enter a topic, pull real PubMed abstracts, and see the relationships "
    "extracted from them as a graph."
)

query = st.text_input(
    "PubMed search term",
    value="Alzheimer's disease amyloid microglia",
)
max_results = st.slider("Number of abstracts to pull", 5, 30, 15)

if st.button("Build knowledge graph"):
    with st.spinner("Fetching abstracts from PubMed..."):
        abstracts = fetch_abstracts(query, max_results)

    if not abstracts:
        st.warning("No abstracts found. Try a different search term.")
    else:
        st.success(f"Pulled {len(abstracts)} abstracts.")

        all_triples = []
        for abstract in abstracts:
            all_triples.extend(extract_triples(abstract, nlp))

        if not all_triples:
            st.warning("No relationships were extracted from these abstracts. Try another topic.")
        else:
            st.write(f"Extracted {len(all_triples)} relationships.")

            G = build_graph(all_triples)

            fig, ax = plt.subplots(figsize=(12, 9))
            pos = nx.spring_layout(G, seed=42, k=1.2)
            nx.draw(
                G, pos, ax=ax, with_labels=True, node_color="lightblue",
                node_size=1800, font_size=7, arrowsize=15,
            )
            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
            st.pyplot(fig)

            with st.expander("See extracted triples as a table"):
                st.table(all_triples)
