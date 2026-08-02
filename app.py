"""
Neurodegenerative Disease Knowledge Graph Explorer
----------------------------------------------------
A small Streamlit app that pulls PubMed abstracts on a topic you choose,
extracts simple subject-verb-object relationships with spaCy, and draws
them as an interactive knowledge graph.

Run locally with:  streamlit run app.py
Or deploy free at:  share.streamlit.io  (connect this file's GitHub repo)
"""

import streamlit as st
from Bio import Entrez
import spacy
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import re

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

    abstracts = [a.strip() for a in raw.split("\n\n") if len(a.strip()) > 200]
    return abstracts


def normalize_text(text):
    """Lowercase and strip leading articles so 'the plaque phenotype' and
    'plaque phenotype' collapse into the same graph node instead of
    cluttering the graph as separate nodes."""
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", text)
    return text


def get_span_for_token(token):
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            return chunk.text
    return token.text


def extract_triples(text, nlp):
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
            triples.append((
                normalize_text(get_span_for_token(subj)),
                verb.lemma_,
                normalize_text(get_span_for_token(obj)),
            ))
    return triples


def build_graph(triples, max_nodes=None):
    """Build the full graph, then optionally keep only the most-connected
    nodes so the visualization stays readable instead of turning into
    a hairball."""
    G = nx.DiGraph()
    for subj, verb, obj in triples:
        if G.has_edge(subj, obj):
            continue  # skip exact duplicate edges
        G.add_edge(subj, obj, label=verb)

    if max_nodes and G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    return G


def render_pyvis_graph(G):
    net = Network(height="650px", width="100%", directed=True, notebook=False, bgcolor="#ffffff")
    net.from_nx(G)
    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -12000, "springLength": 180, "springConstant": 0.02}},
      "nodes": {"font": {"size": 16}, "color": {"background": "#a9d0f5"}},
      "edges": {"font": {"size": 10, "align": "middle"}, "color": {"color": "#888888"}}
    }
    """)
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=680, scrolling=True)


# ---- UI ----
st.set_page_config(page_title="Neuro Knowledge Graph Explorer", layout="wide")
st.title("Neurodegenerative Disease Knowledge Graph Explorer")
st.write(
    "Enter a topic, pull real PubMed abstracts, and see the relationships "
    "extracted from them as an interactive graph -- drag nodes, zoom, and "
    "hover to explore."
)

query = st.text_input(
    "PubMed search term",
    value="Alzheimer's disease amyloid microglia",
)
col1, col2 = st.columns(2)
with col1:
    max_results = st.slider("Number of abstracts to pull", 5, 30, 10)
with col2:
    max_nodes = st.slider(
        "Max nodes to show (keeps the graph readable)", 15, 100, 40,
        help="Only the most-connected nodes are kept when the graph gets larger than this.",
    )

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
            G = build_graph(all_triples, max_nodes=max_nodes)
            st.write(
                f"Extracted {len(all_triples)} relationships total -- "
                f"showing the {G.number_of_nodes()} most-connected nodes "
                f"({G.number_of_edges()} edges) below."
            )
            render_pyvis_graph(G)

            with st.expander("See all extracted triples as a table"):
                st.table(all_triples)
