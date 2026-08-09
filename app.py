"""
NeuroLoom: Neurodegenerative Disease Knowledge Graph Explorer
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

# Simple line-art icon paths (24x24 viewBox), reused across trust badges and
# home cards so the whole app draws from one small, consistent icon set
# instead of mismatched emoji.
ICON_BOOK = '<path d="M4 4.5c2-1 5-1 7 0v15c-2-1-5-1-7 0z"/><path d="M20 4.5c-2-1-5-1-7 0v15c2-1 5-1 7 0z"/>'
ICON_FLASK = '<path d="M9 3h6"/><path d="M10 3v6l-5.3 9.2A1.8 1.8 0 0 0 6.3 21h11.4a1.8 1.8 0 0 0 1.6-2.8L14 9V3"/>'
ICON_LINK = '<path d="M9 15l6-6"/><path d="M11 6l.9-.9a3.5 3.5 0 0 1 5 5l-1 1"/><path d="M13 18l-.9.9a3.5 3.5 0 0 1-5-5l1-1"/>'
ICON_EYE = '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>'
ICON_SEARCH = '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>'


def svg_icon(paths, css_class="badge-icon"):
    return (f'<svg class="{css_class}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')


TRUST_BADGES = [
    (ICON_BOOK, "Straight from PubMed"),
    (ICON_FLASK, "Trials, not just papers"),
    (ICON_LINK, "Every edge cites its source"),
    (ICON_EYE, "See exactly how it works"),
]

WHY_CARDS = [
    (ICON_SEARCH, "Why it matters", "Research is scattered",
     "Findings on a disease like Alzheimer's or ALS live in thousands of separate papers, "
     "and nobody has time to read them all. NeuroLoom pulls the threads together so you can "
     "see how they connect."),
    (ICON_LINK, "Built on primary sources", "Every claim is traceable",
     "Every abstract comes straight from PubMed, every trial from ClinicalTrials.gov. Click "
     "through any relationship in the graph and you land on the actual PMID it came from, "
     "not a black box."),
    (ICON_EYE, "Data driven, transparent", "The method is visible",
     "spaCy reads the grammar of each sentence to spot who's doing what to whom, then checks "
     "it against a biomedical glossary. If a paper reports X does <em>not</em> cause Y, that "
     "gets flagged, not smoothed over. Full pipeline is on the <strong>Components</strong> page."),
]

PIPELINE_STEPS = [
    ("1", "Pick a topic", "Choose a preset in the sidebar, or type your own PubMed search. Your call."),
    ("2", "Pull the sources", "Requests go out to PubMed, ClinicalTrials.gov, and Wikipedia for that exact term, all in one pass."),
    ("3", "Parse each abstract", "spaCy reads every sentence and works out who's the subject, what's the verb, and what's on the receiving end."),
    ("4", "Type and normalize", "Each entity gets matched against a biomedical glossary; near-duplicates like \"AD\" and \"Alzheimer's disease\" collapse into one node."),
    ("5", "Flag negation", "Catches sentences like \"X does not cause Y\" and marks them, instead of quietly treating them like a positive claim."),
    ("6", "Build and render", "NetworkX wires up the nodes and edges, and pyvis turns that into something you can actually drag around."),
]

TECH_STACK = [
    ("Streamlit", "Runs the whole interface, hosted free on Streamlit Community Cloud."),
    ("spaCy", "Does the actual grammar parsing: subject, verb, object, for every sentence."),
    ("Biopython", "Speaks NCBI's Entrez API so we can search and pull PubMed abstracts."),
    ("NetworkX", "Holds the graph in memory: nodes for entities, directed edges for relationships."),
    ("pyvis", "Turns that graph into something you can drag, zoom, and click around."),
    ("pandas", "Behind the relationships table, the CSV export, and the trials table."),
    ("Requests", "The plain HTTP library making the ClinicalTrials.gov and Wikipedia calls."),
    ("Python", "Everything above is written in it."),
]

DATA_SOURCES = [
    ("NCBI PubMed", "Millions of biomedical abstracts, pulled through the Entrez E-utilities API "
                     "(Bio.Entrez, Bio.Medline). Every relationship in the graph traces back to "
                     "something published here.",
     "https://pubmed.ncbi.nlm.nih.gov/"),
    ("ClinicalTrials.gov", "Live status of registered studies, recruiting, active, or wrapped up, "
                            "for whatever condition you search, via ClinicalTrials.gov API v2.",
     "https://clinicaltrials.gov/data-api/api"),
    ("Wikipedia", "A quick, plain-language overview so you're not dropped into jargon cold, via "
                  "the Wikipedia REST Summary API.",
     "https://en.wikipedia.org/api/rest_v1/"),
]

# ---- Page setup & styling --------------------------------------------
st.set_page_config(
    page_title="NeuroLoom",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&family=Roboto+Condensed:wght@400;500;600;700&family=Roboto+Slab:wght@700;800&family=Baloo+2:wght@600;700;800&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #2f1b4d 0%, #47295f 45%, #6b3f8f 100%);
}

.st-key-navbar {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 999px;
    padding: 0.4rem 1.25rem;
    margin-bottom: 1.5rem;
}
.topnav-brand {
    font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: 1.35rem;
    color: #ffffff;
}

.hero {
    background: linear-gradient(135deg, #fdf3ea 0%, #fbdfe6 50%, #ffd7bd 100%);
    padding: 2.25rem 2.25rem;
    border-radius: 22px;
    color: #241b2e;
    margin-bottom: 1rem;
    box-shadow: 0 16px 40px rgba(20, 10, 35, 0.35);
}
.hero h1 {
    margin: 0; font-size: clamp(1.6rem, 3.4vw, 2.5rem); line-height: 1.15;
    font-family: 'Roboto Slab', serif; font-weight: 800; color: #221830;
}
.hero p.subhead { margin: 0.75rem 0 0 0; opacity: 0.85; font-size: clamp(0.92rem, 1.6vw, 1.05rem); max-width: 46rem; color: #362a44; }

.page-title {
    font-family: 'Roboto Slab', serif; font-weight: 800; color: #ffffff;
    font-size: clamp(1.4rem, 3vw, 2.05rem); margin: 0.25rem 0 0.35rem 0;
}
.page-subtitle { color: #e9def0; opacity: 0.85; font-size: 0.95rem; max-width: 46rem; margin-bottom: 1.5rem; }
.section-kicker {
    font-family: 'Roboto Condensed', sans-serif; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.6rem;
}

.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0 1.6rem 0; }
.trust-badge {
    font-family: 'Roboto Condensed', sans-serif;
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.22);
    color: #fdf6ff;
    padding: 0.32rem 0.85rem 0.32rem 0.65rem; border-radius: 999px; font-size: 0.8rem; white-space: nowrap;
}
.badge-icon { width: 15px; height: 15px; flex-shrink: 0; }

.card-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 0.9rem; margin-bottom: 1.6rem;
}
.glass-card {
    background: rgba(255,255,255,0.08); backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.16); border-radius: 16px;
    padding: 1.1rem 1.2rem;
}
.card-icon {
    width: 38px; height: 38px; border-radius: 50%;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; font-family: 'Roboto Condensed', sans-serif; font-weight: 700;
    color: #ffd7bd; margin-bottom: 0.6rem;
}
.card-icon.teal { color: #8fe9dc; background: rgba(143,233,220,0.14); border-color: rgba(143,233,220,0.32); }
.card-icon svg { width: 18px; height: 18px; }
.glass-card .card-kicker { font-family: 'Roboto Condensed', sans-serif; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #e7d3f0; opacity: 0.8; }
.glass-card h4 { font-family: 'Poppins', sans-serif; margin: 0.3rem 0 0.5rem 0; font-size: 1.02rem; font-weight: 600; color: #ffffff; }
.glass-card p { margin: 0; font-size: 0.87rem; color: #f1e9f6; opacity: 0.92; line-height: 1.5; }
.glass-card a { color: #ffd7bd; }

.tech-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.75rem; margin-bottom: 1.6rem;
}
.tech-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14);
    border-left: 3px solid #ffd7bd;
    border-radius: 12px; padding: 0.8rem 1rem;
}
.tech-card.teal { border-left-color: #8fe9dc; }
.tech-card .tech-name { font-family: 'Poppins', sans-serif; font-weight: 600; color: #ffffff; font-size: 0.92rem; }
.tech-card .tech-role { font-size: 0.8rem; color: #e9def0; opacity: 0.85; margin-top: 0.2rem; line-height: 1.4; }

.stat-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem; margin: 0.5rem 0 1.2rem 0;
}
.stat-box {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255, 175, 130, 0.35);
    border-radius: 14px; padding: 0.85rem 1rem;
}
.stat-box .stat-num { font-family: 'Roboto Slab', serif; font-size: 1.55rem; font-weight: 700; line-height: 1.1; color: #ffb385; }
.stat-box .stat-label { font-family: 'Roboto Condensed', sans-serif; font-size: 0.78rem; color: #f1e9f6; opacity: 0.8; margin-top: 0.2rem; }

.fun-badge {
    width: 78px; height: 78px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, #8ff0e3, #34b3a5);
    color: #072421; display: flex; flex-direction: column; align-items: center; justify-content: center;
    font-family: 'Baloo 2', sans-serif; font-weight: 700; font-size: 0.72rem; text-align: center; line-height: 1.3;
    transform: rotate(-7deg);
    box-shadow: 0 10px 22px rgba(52,179,165,0.35);
}
.fun-badge .fun-badge-big { font-size: 1.1rem; }

.legend-chip {
    font-family: 'Roboto Condensed', sans-serif;
    display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
    color: white; font-size: 0.8rem; margin-right: 0.4rem; margin-bottom: 0.3rem;
}
.disclaimer {
    font-size: 0.8rem; color: #d9cde3; border-top: 1px solid rgba(255,255,255,0.15);
    padding-top: 0.75rem; margin-top: 1.5rem;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06); padding: 0.75rem 1rem; border-radius: 10px;
}

.stButton > button, .stLinkButton > a { border-radius: 999px; font-family: 'Roboto Condensed', sans-serif; font-weight: 600; }
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #ff8a5c, #ff6a88);
    border: none;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.06); }

@media (max-width: 640px) {
    .hero { padding: 1.5rem; }
    .card-grid, .stat-strip, .tech-grid { grid-template-columns: 1fr; }
    /* st.columns stacks vertically below this width, so the navbar becomes a
       tall single column; a 999px pill radius on something that tall reads as
       a broken oval, so use a normal rounded-rectangle instead. */
    .st-key-navbar { border-radius: 20px; padding: 0.9rem 1rem; }
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
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown("### 🧠 Explore a disease")
    for label, preset_query in DISEASE_PRESETS:
        if st.button(label, key=preset_query, width="stretch"):
            st.session_state.query = preset_query
            st.session_state.page = "Home"
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
with st.container(key="navbar"):
    nav_brand, nav_home, nav_components, nav_spacer, nav_source = st.columns(
        [2.4, 0.9, 1.3, 2.4, 1.5], vertical_alignment="center"
    )
    with nav_brand:
        st.markdown('<div class="topnav-brand">NeuroLoom</div>', unsafe_allow_html=True)
    with nav_home:
        if st.button("Home", key="nav_home",
                      type="primary" if st.session_state.page == "Home" else "secondary",
                      width="stretch"):
            st.session_state.page = "Home"
            st.rerun()
    with nav_components:
        if st.button("Components", key="nav_components",
                      type="primary" if st.session_state.page == "Components" else "secondary",
                      width="stretch"):
            st.session_state.page = "Components"
            st.rerun()
    with nav_source:
        st.link_button("View source", "https://github.com/lmarshall-boop/Knowledge-Graph", width="stretch")

# =====================================================================
# HOME PAGE
# =====================================================================
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero">
      <h1>See what decades of neurodegenerative disease research actually say</h1>
      <p class="subhead">Pick a disease and watch years of scattered research turn into one
      graph you can actually explore, built straight from PubMed abstracts and active
      clinical trials instead of a pile of browser tabs you'll never finish reading.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="badge-row">' + "".join(
            f'<span class="trust-badge">{svg_icon(path)}{label}</span>'
            for path, label in TRUST_BADGES
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-grid">' + "".join(
            f'<div class="glass-card"><div class="card-icon">{svg_icon(path, "")}</div>'
            f'<div class="card-kicker">{kicker}</div><h4>{title}</h4><p>{body}</p></div>'
            for path, kicker, title, body in WHY_CARDS
        ) + '</div>',
        unsafe_allow_html=True,
    )

    build_col, badge_col = st.columns([3, 1], vertical_alignment="center")
    with build_col:
        build_clicked = st.button("🕸️ Build the graph", type="primary")
    with badge_col:
        st.markdown(
            '<div class="fun-badge"><span class="fun-badge-big">⚡</span>usually<br>~10 sec</div>',
            unsafe_allow_html=True,
        )

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
        st.info("Pick a disease from the sidebar, or type your own search, then hit build above.")
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

# =====================================================================
# COMPONENTS PAGE
# =====================================================================
else:
    st.markdown('<div class="page-title">What\'s actually happening under the hood</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">No magic, just a pipeline, a handful of tools, and '
        'sources you can click through and check yourself.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker" style="color:#ffb385;">The pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-grid">' + "".join(
            f'<div class="glass-card"><div class="card-icon{" teal" if i % 2 else ""}">{num}</div>'
            f'<h4>{title}</h4><p>{desc}</p></div>'
            for i, (num, title, desc) in enumerate(PIPELINE_STEPS)
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker" style="color:#8fe9dc;">Built with</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tech-grid">' + "".join(
            f'<div class="tech-card{" teal" if i % 2 else ""}"><div class="tech-name">{name}</div>'
            f'<div class="tech-role">{role}</div></div>'
            for i, (name, role) in enumerate(TECH_STACK)
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker" style="color:#ffb385;">Where the data comes from</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-grid">' + "".join(
            f'<div class="glass-card"><h4>{name}</h4><p>{desc}</p>'
            f'<p style="margin-top:0.5rem;"><a href="{url}" target="_blank">{url}</a></p></div>'
            for name, desc, url in DATA_SOURCES
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='disclaimer'>The \"57 million\" figure on dementia prevalence, when it "
        "shows up on the Home page, comes from the "
        "<a href=\"https://www.who.int/news-room/fact-sheets/detail/dementia\" target=\"_blank\">"
        "WHO dementia fact sheet</a>. Every record count is pulled live for your exact search "
        "term, never hard-coded. And to be clear: this is a research exploration tool, not "
        "medical advice. NLP makes mistakes, so always check the source abstract before "
        "trusting a claim.</div>",
        unsafe_allow_html=True,
    )
