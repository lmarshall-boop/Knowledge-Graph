"""
Neurodegenerative Disease Knowledge Graph Explorer
----------------------------------------------------
A Streamlit app that pulls PubMed abstracts on a topic you choose, extracts
typed subject-verb-object relationships with spaCy, cross-references active
ClinicalTrials.gov studies, and draws it all as an interactive, color-coded
knowledge graph.

Run locally with:  streamlit run app.py
Or deploy free at:  share.streamlit.io  (connect this file's GitHub repo)
"""

import csv
import io
import tempfile

import networkx as nx
import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from nlp_utils import ENTITY_COLORS, extract_triples
from sources import fetch_clinical_trials, fetch_pubmed_abstracts, fetch_wikipedia_summary

DISEASE_PRESETS = [
    ("Alzheimer's disease", "Alzheimer's disease amyloid microglia"),
    ("Parkinson's disease", "Parkinson's disease alpha-synuclein"),
    ("ALS", "amyotrophic lateral sclerosis TDP-43"),
    ("Huntington's disease", "Huntington's disease huntingtin"),
    ("Frontotemporal dementia", "frontotemporal dementia tau"),
    ("Lewy body dementia", "Lewy body dementia"),
]

# ---- Page setup & styling --------------------------------------------
st.set_page_config(
    page_title="Neuro Knowledge Graph Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg, #4b3f8f 0%, #7a5fc9 50%, #b892e8 100%);
    padding: 1.75rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.25rem;
}
.hero h1 { margin: 0; font-size: 1.9rem; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }
.legend-chip {
    display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
    color: white; font-size: 0.78rem; margin-right: 0.4rem; margin-bottom: 0.3rem;
}
.disclaimer {
    font-size: 0.8rem; color: #8a8f98; border-top: 1px solid #33363f;
    padding-top: 0.75rem; margin-top: 1.5rem;
}
[data-testid="stMetric"] {
    background: rgba(127,127,127,0.08); padding: 0.75rem 1rem; border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


nlp = load_spacy_model()

cached_pubmed = st.cache_data(ttl=3600, show_spinner=False)(fetch_pubmed_abstracts)
cached_trials = st.cache_data(ttl=3600, show_spinner=False)(fetch_clinical_trials)
cached_wiki = st.cache_data(ttl=86400, show_spinner=False)(fetch_wikipedia_summary)


# ---- Graph construction & rendering -----------------------------------
def build_graph(triples, max_nodes=None):
    """Build the full graph, then optionally keep only the most-connected
    nodes so the visualization stays readable instead of turning into a
    hairball."""
    G = nx.DiGraph()
    for t in triples:
        subj, obj = t["subject"], t["object"]
        if subj == obj or G.has_edge(subj, obj):
            continue
        G.add_node(subj, type=t["subject_type"])
        G.add_node(obj, type=t["object_type"])
        G.add_edge(subj, obj, label=t["verb"], negated=t["negated"],
                   sentence=t["sentence"], source_id=t["source_id"])

    if max_nodes and G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    return G


def render_pyvis_graph(G, highlight=""):
    net = Network(height="650px", width="100%", directed=True, notebook=False, bgcolor="#0e1117", font_color="white")

    highlight = highlight.strip().lower()
    for node, data in G.nodes(data=True):
        entity_type = data.get("type", "OTHER")
        color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS["OTHER"])
        size = 26 if highlight and highlight in node.lower() else 16
        net.add_node(node, label=node, color=color, title=f"{node} ({entity_type})", size=size)

    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        negated = data.get("negated", False)
        if negated:
            label = f"NOT {label}"
        net.add_edge(
            u, v, label=label,
            color="#e8746b" if negated else "#9aa0ab",
            dashes=negated,
            title=data.get("sentence", ""),
        )

    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -12000, "springLength": 180, "springConstant": 0.02}},
      "nodes": {"font": {"size": 16, "color": "#ffffff"}},
      "edges": {"font": {"size": 10, "align": "middle", "color": "#dddddd"}}
    }
    """)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        path = f.name
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=680, scrolling=True)


def triples_to_dataframe(triples):
    return pd.DataFrame([{
        "Subject": t["subject"], "Subject type": t["subject_type"],
        "Relation": ("NOT " if t["negated"] else "") + t["verb"],
        "Object": t["object"], "Object type": t["object_type"],
        "Source PMID": t["source_id"],
        "Sentence": t["sentence"],
    } for t in triples])


# ---- Sidebar ------------------------------------------------------------
if "query" not in st.session_state:
    st.session_state.query = DISEASE_PRESETS[0][1]

with st.sidebar:
    st.markdown("### 🧠 Explore a disease")
    for label, preset_query in DISEASE_PRESETS:
        if st.button(label, key=preset_query):
            st.session_state.query = preset_query
            st.rerun()

    st.markdown("---")
    query = st.text_input("Custom PubMed search term", key="query")

    with st.expander("Advanced settings"):
        max_results = st.slider("Abstracts to pull", 5, 30, 10)
        max_nodes = st.slider("Max graph nodes", 15, 100, 40)
        include_trials = st.checkbox("Include ClinicalTrials.gov", value=True)
        include_wiki = st.checkbox("Include Wikipedia overview", value=True)
        hide_negated = st.checkbox("Hide negated relationships", value=False)

    st.markdown(
        "<div class='disclaimer'>Research exploration tool only — "
        "not medical advice. Relationships are extracted computationally "
        "and may contain NLP errors; always verify against the source "
        "abstract.</div>",
        unsafe_allow_html=True,
    )

# ---- Header --------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>Neurodegenerative Disease Knowledge Graph Explorer</h1>
  <p>Live PubMed research + active clinical trials, mapped as a typed, explorable graph.</p>
</div>
""", unsafe_allow_html=True)

build_clicked = st.button("🔎 Build knowledge graph", type="primary")

if build_clicked:
    with st.spinner("Fetching abstracts, trials, and context..."):
        abstracts = cached_pubmed(query, max_results)
        trials = cached_trials(query) if include_trials else []
        wiki = cached_wiki(query) if include_wiki else None

    st.session_state["last_result"] = {
        "abstracts": abstracts, "trials": trials, "wiki": wiki, "query": query,
    }

result = st.session_state.get("last_result")

if not result:
    st.info("Pick a preset in the sidebar or enter your own search term, then build the graph.")
else:
    abstracts, trials, wiki = result["abstracts"], result["trials"], result["wiki"]

    if wiki:
        with st.container():
            st.markdown(f"**{wiki['title']}** — {wiki['extract']}  \n[Read more]({wiki['url']})")

    if not abstracts:
        st.warning("No abstracts found for this search term. Try a different query.")
    else:
        all_triples = []
        for a in abstracts:
            all_triples.extend(extract_triples(a["abstract"], nlp, source_id=a["pmid"]))

        if hide_negated:
            all_triples = [t for t in all_triples if not t["negated"]]

        G = build_graph(all_triples, max_nodes=max_nodes)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Abstracts pulled", len(abstracts))
        m2.metric("Entities in graph", G.number_of_nodes())
        m3.metric("Relationships", G.number_of_edges())
        m4.metric("Active trials", len(trials))

        st.markdown(
            "".join(
                f"<span class='legend-chip' style='background:{color}'>{etype.replace('_',' ').title()}</span>"
                for etype, color in ENTITY_COLORS.items() if etype != "OTHER"
            ),
            unsafe_allow_html=True,
        )

        tab_graph, tab_table, tab_trials, tab_sources = st.tabs(
            ["🕸️ Graph", "📋 Relationships", "🧪 Clinical trials", "📚 Sources"]
        )

        with tab_graph:
            highlight = st.text_input("Highlight a node (e.g. 'tau')", "")
            if G.number_of_nodes() == 0:
                st.warning("No relationships were extracted from these abstracts. Try another topic.")
            else:
                render_pyvis_graph(G, highlight=highlight)

        with tab_table:
            df = triples_to_dataframe(all_triples)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            graphml_str = "\n".join(nx.generate_graphml(G))
            c1, c2 = st.columns(2)
            c1.download_button("Download relationships (CSV)", csv_buf.getvalue(),
                                file_name="relationships.csv", mime="text/csv")
            c2.download_button("Download graph (GraphML)", graphml_str,
                                file_name="graph.graphml", mime="application/xml")

        with tab_trials:
            if not trials:
                st.info("No active trials found, or trial lookup was disabled.")
            else:
                st.dataframe(pd.DataFrame(trials), use_container_width=True, hide_index=True,
                             column_config={"url": st.column_config.LinkColumn("Link")})

        with tab_sources:
            for a in abstracts:
                with st.expander(f"{a['title']} (PMID {a['pmid']})"):
                    st.write(a["abstract"])
                    if a["pmid"]:
                        st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)")
