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

import io

import networkx as nx
import pandas as pd
import spacy
import streamlit as st
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
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #2f1b4d 0%, #47295f 45%, #6b3f8f 100%);
}

.topnav {
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1.25rem;
}
.topnav-brand { font-weight: 700; font-size: 1.05rem; color: #ffffff; letter-spacing: 0.01em; }
.topnav-actions { display: flex; align-items: center; gap: 1rem; }
.topnav-actions a { text-decoration: none; }
.nav-link { color: rgba(255,255,255,0.82); font-size: 0.88rem; }
.pill-btn-outline {
    color: #ffffff; font-size: 0.82rem; font-weight: 600;
    padding: 0.4rem 1rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.35);
}

.hero {
    background: linear-gradient(135deg, #fdf3ea 0%, #fbdfe6 50%, #ffd7bd 100%);
    padding: 2.25rem 2.25rem;
    border-radius: 22px;
    color: #241b2e;
    margin-bottom: 1rem;
    box-shadow: 0 16px 40px rgba(20, 10, 35, 0.35);
}
.hero h1 { margin: 0; font-size: clamp(1.6rem, 3.4vw, 2.5rem); line-height: 1.15; font-weight: 800; color: #221830; }
.hero p.subhead { margin: 0.75rem 0 0 0; opacity: 0.85; font-size: clamp(0.92rem, 1.6vw, 1.05rem); max-width: 46rem; color: #362a44; }

.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0 1.6rem 0; }
.trust-badge {
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.22);
    color: #fdf6ff;
    padding: 0.32rem 0.85rem; border-radius: 999px; font-size: 0.78rem; white-space: nowrap;
}

.card-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 0.9rem; margin-bottom: 1.6rem;
}
.glass-card {
    background: rgba(255,255,255,0.08); backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.16); border-radius: 16px;
    padding: 1.1rem 1.2rem;
}
.glass-card .card-kicker { font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #e7d3f0; opacity: 0.8; }
.glass-card h4 { margin: 0.3rem 0 0.5rem 0; font-size: 1.02rem; color: #ffffff; }
.glass-card p { margin: 0; font-size: 0.87rem; color: #f1e9f6; opacity: 0.92; line-height: 1.5; }

.stat-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem; margin: 0.5rem 0 1.2rem 0;
}
.stat-box {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255, 175, 130, 0.35);
    border-radius: 14px; padding: 0.85rem 1rem;
}
.stat-box .stat-num { font-size: 1.55rem; font-weight: 700; line-height: 1.1; color: #ffb385; }
.stat-box .stat-label { font-size: 0.76rem; color: #f1e9f6; opacity: 0.8; margin-top: 0.2rem; }

.legend-chip {
    display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
    color: white; font-size: 0.78rem; margin-right: 0.4rem; margin-bottom: 0.3rem;
}
.disclaimer {
    font-size: 0.8rem; color: #d9cde3; border-top: 1px solid rgba(255,255,255,0.15);
    padding-top: 0.75rem; margin-top: 1.5rem;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06); padding: 0.75rem 1rem; border-radius: 10px;
}

.stButton > button { border-radius: 999px; }
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #ff8a5c, #ff6a88);
    border: none; font-weight: 600;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.06); }

@media (max-width: 640px) {
    .hero { padding: 1.5rem; }
    .card-grid, .stat-strip { grid-template-columns: 1fr; }
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


def format_count_plus(n):
    """Format a count as a conservative, always-true 'X,000+' style figure
    by rounding DOWN, so the number on screen never overstates what's
    actually behind it."""
    if n >= 1000:
        return f"{(n // 1000) * 1000:,}+"
    if n >= 100:
        return f"{(n // 100) * 100}+"
    return str(n)


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

    html = net.generate_html()
    st.iframe(html, height=680)


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
        if st.button(label, key=preset_query, width="stretch"):
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
        "<div class='disclaimer'>Research exploration tool only. Not medical "
        "advice. Relationships are extracted computationally and may contain "
        "NLP errors; always verify against the source abstract.</div>",
        unsafe_allow_html=True,
    )

# ---- Top nav --------------------------------------------------------------
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">NeuroGraph</div>
  <div class="topnav-actions">
    <a href="#methodology-sources" class="nav-link">Methodology</a>
    <a href="https://github.com/lmarshall-boop/Knowledge-Graph" target="_blank" class="pill-btn-outline">View source</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Header: what this is, in one glance --------------------------------
st.markdown("""
<div class="hero">
  <h1>See what decades of neurodegenerative disease research actually say</h1>
  <p class="subhead">Pick a disease. This tool pulls real PubMed literature and active
  clinical trials, extracts the relationships researchers have reported, and lays them
  out as one connected, explorable graph, not hundreds of separate abstracts.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
  <span class="trust-badge">Live NCBI PubMed data</span>
  <span class="trust-badge">ClinicalTrials.gov integrated</span>
  <span class="trust-badge">Every relationship cites its source</span>
  <span class="trust-badge">Open method, not a black box</span>
</div>
""", unsafe_allow_html=True)

# ---- Why this exists ------------------------------------------------------
st.markdown("""
<div class="card-grid">
  <div class="glass-card">
    <div class="card-kicker">Why it matters</div>
    <h4>Research is scattered</h4>
    <p>Findings on a disease like Alzheimer's or ALS are spread across thousands of
    separate papers. Most people don't have time to read all of them. This tool makes
    that literature easier to navigate.</p>
  </div>
  <div class="glass-card">
    <div class="card-kicker">Built on primary sources</div>
    <h4>Every claim is traceable</h4>
    <p>Abstracts come from NCBI's PubMed and trial data comes from ClinicalTrials.gov.
    Every relationship in the graph links back to the PMID it came from, so you can
    check it against the original source.</p>
  </div>
  <div class="glass-card">
    <div class="card-kicker">Data driven, transparent</div>
    <h4>The method is visible</h4>
    <p>Relationships are extracted with spaCy's dependency parser and typed against a
    curated biomedical lexicon. It's a documented process, not a proprietary model.
    Negated claims, like "X does <em>not</em> cause Y," are flagged instead of hidden.</p>
  </div>
</div>
""", unsafe_allow_html=True)

build_clicked = st.button("🔎 Build knowledge graph", type="primary")

if build_clicked:
    with st.spinner("Fetching abstracts, trials, and context..."):
        pubmed_result = cached_pubmed(query, max_results)
        trials_result = cached_trials(query) if include_trials else {"trials": [], "total_count": 0}
        wiki = cached_wiki(query) if include_wiki else None

    # Defensive: Streamlit Cloud's hot-reload can occasionally keep serving a
    # cache entry computed under a previous version of these functions (back
    # when they returned a plain list instead of a dict). Normalize instead
    # of crashing if that ever happens again.
    if isinstance(pubmed_result, list):
        pubmed_result = {"abstracts": pubmed_result, "total_count": len(pubmed_result)}
    if isinstance(trials_result, list):
        trials_result = {"trials": trials_result, "total_count": len(trials_result)}

    st.session_state["last_result"] = {
        "pubmed_result": pubmed_result, "trials_result": trials_result,
        "wiki": wiki, "query": query,
    }

result = st.session_state.get("last_result")

if not result:
    st.info("Pick a preset in the sidebar or enter your own search term, then build the graph.")
else:
    abstracts = result["pubmed_result"]["abstracts"]
    pubmed_total = result["pubmed_result"]["total_count"]
    trials = result["trials_result"]["trials"]
    trials_total = result["trials_result"]["total_count"]
    wiki = result["wiki"]

    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-box"><div class="stat-num">{format_count_plus(pubmed_total)}</div>
        <div class="stat-label">PubMed records matching "{result['query']}"</div></div>
      <div class="stat-box"><div class="stat-num">{format_count_plus(trials_total)}</div>
        <div class="stat-label">Registered clinical trials matching this search</div></div>
      <div class="stat-box"><div class="stat-num">{len(abstracts)}</div>
        <div class="stat-label">Abstracts analyzed in depth for this graph</div></div>
    </div>
    """, unsafe_allow_html=True)

    if wiki:
        with st.container():
            st.markdown(f"**{wiki['title']}.** {wiki['extract']}  \n[Read more]({wiki['url']})")

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
        m4.metric("Active trials shown", len(trials))

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
            st.dataframe(df, width="stretch", hide_index=True)

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
                st.caption(f"Showing {len(trials)} of {format_count_plus(trials_total)} matching registered studies.")
                st.dataframe(pd.DataFrame(trials), width="stretch", hide_index=True,
                             column_config={"url": st.column_config.LinkColumn("Link")})

        with tab_sources:
            for a in abstracts:
                with st.expander(f"{a['title']} (PMID {a['pmid']})"):
                    st.write(a["abstract"])
                    if a["pmid"]:
                        st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)")

# ---- Footer: methodology & sources ---------------------------------------
st.markdown("---")
with st.expander("📎 Methodology & sources", expanded=False):
    st.markdown("""
- **Literature:** [NCBI PubMed](https://pubmed.ncbi.nlm.nih.gov/) via the Entrez E-utilities API (`Bio.Entrez`, `Bio.Medline`)
- **Clinical trials:** [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api)
- **Plain-language context:** [Wikipedia REST Summary API](https://en.wikipedia.org/api/rest_v1/)
- **Relationship extraction:** spaCy `en_core_web_sm` dependency parser, subject-verb-object pattern matching
- **Entity typing:** a curated neurodegenerative-disease lexicon (disease, gene-protein, drug-chemical, anatomy, process)
- **"57 million"** figure on dementia prevalence, when shown, is from the [WHO dementia fact sheet](https://www.who.int/news-room/fact-sheets/detail/dementia)

Record counts above are pulled live from each API for your exact search term, not
hard-coded. They reflect the full matching corpus, while the graph itself analyzes
only the sample of abstracts set in **Advanced settings**.
    """)
