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

TRUST_BADGES = [
    "Straight from PubMed",
    "Trials, not just papers",
    "Every edge cites its source",
    "See exactly how it works",
]

WHY_CARDS = [
    ("Why it matters", "Research is scattered",
     "Findings on a disease like Alzheimer's or ALS live in thousands of separate papers, "
     "and nobody has time to read them all. NeuroLoom pulls the threads together so you can "
     "see how they connect."),
    ("Built on primary sources", "Every claim is traceable",
     "Every abstract comes straight from PubMed, every trial from ClinicalTrials.gov. Click "
     "through any relationship in the graph and you land on the actual PMID it came from, "
     "not a black box."),
    ("Data driven, transparent", "The method is visible",
     "spaCy reads the grammar of each sentence to spot who's doing what to whom, then checks "
     "it against a biomedical glossary. If a paper reports X does <em>not</em> cause Y, that "
     "gets flagged, not smoothed over."),
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

# ---- Decorative assets (ported/adapted from NeuroLoom Redesign.dc.html) ----

LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 34 34">'
    '<defs><linearGradient id="logoGrad" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#c9558f"/><stop offset="1" stop-color="#7a3f8f"/></linearGradient></defs>'
    '<path d="M9 9 L17 15 L25 8" stroke="url(#logoGrad)" stroke-width="1.6" fill="none"/>'
    '<path d="M9 25 L17 19 L25 26" stroke="url(#logoGrad)" stroke-width="1.6" fill="none"/>'
    '<path d="M17 15 L17 19" stroke="url(#logoGrad)" stroke-width="1.6" fill="none"/>'
    '<circle cx="9" cy="9" r="3" fill="url(#logoGrad)"/><circle cx="25" cy="8" r="2.4" fill="url(#logoGrad)"/>'
    '<circle cx="17" cy="15" r="3.4" fill="url(#logoGrad)"/><circle cx="17" cy="19" r="3.4" fill="url(#logoGrad)"/>'
    '<circle cx="9" cy="25" r="2.4" fill="url(#logoGrad)"/><circle cx="25" cy="26" r="3" fill="url(#logoGrad)"/>'
    '</svg>'
)

INSTAGRAM_SVG = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
    '<rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="12" cy="12" r="4"/>'
    '<circle cx="17.2" cy="6.8" r="1"/></svg>'
)

HERO_GRAPH_SVG = """<svg viewBox="0 0 400 250" style="width:100%;display:block">
  <line x1="70" y1="60" x2="180" y2="120" stroke="#a89ab0" stroke-width="1.5"/>
  <line x1="180" y1="120" x2="120" y2="205" stroke="#a89ab0" stroke-width="1.5"/>
  <line x1="260" y1="45" x2="180" y2="120" stroke="#e8746b" stroke-width="1.5" stroke-dasharray="4 4"/>
  <line x1="180" y1="120" x2="300" y2="150" stroke="#a89ab0" stroke-width="1.5"/>
  <line x1="300" y1="150" x2="340" y2="215" stroke="#a89ab0" stroke-width="1.5"/>
  <line x1="70" y1="60" x2="260" y2="45" stroke="#a89ab0" stroke-width="1.5"/>
  <circle cx="70" cy="60" r="9" fill="#6ba5e8"/>
  <circle cx="260" cy="45" r="9" fill="#e8746b"/>
  <circle cx="180" cy="120" r="11" fill="#b38ce8"/>
  <circle cx="300" cy="150" r="8" fill="#d9b45c"/>
  <circle cx="120" cy="205" r="8" fill="#b38ce8"/>
  <circle cx="340" cy="215" r="8" fill="#7fd18f"/>
  <text x="70" y="42" fill="#EBE4EC" font-size="10" font-family="Roboto Condensed" text-anchor="middle">amyloid-beta</text>
  <text x="260" y="30" fill="#EBE4EC" font-size="10" font-family="Roboto Condensed" text-anchor="middle">alzheimer's disease</text>
  <text x="180" y="140" fill="#EBE4EC" font-size="10" font-family="Roboto Condensed" text-anchor="middle">neuroinflammation</text>
  <text x="300" y="135" fill="#EBE4EC" font-size="10" font-family="Roboto Condensed" text-anchor="middle">hippocampus</text>
  <text x="120" y="222" fill="#EBE4EC" font-size="10" font-family="Roboto Condensed" text-anchor="middle">apoptosis</text>
  <text x="340" y="232" fill="#EBE4EC" font-size="10" font-family="Roboto Condensed" text-anchor="middle">donepezil</text>
</svg>"""


def _mulberry32(seed):
    """Port of the mockup's JS PRNG (identical algorithm, 32-bit wraparound
    arithmetic) so the decorative neuron field matches the design file
    exactly instead of just approximately."""
    state = seed & 0xFFFFFFFF

    def next_val():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t1 = ((state ^ (state >> 15)) * (1 | state)) & 0xFFFFFFFF
        inner = ((t1 ^ (t1 >> 7)) * (61 | t1)) & 0xFFFFFFFF
        t2 = ((t1 + inner) & 0xFFFFFFFF) ^ t1
        result = (t2 ^ (t2 >> 14)) & 0xFFFFFFFF
        return result / 4294967296

    return next_val


def _build_neuron_field():
    rand = _mulberry32(7)
    node_colors = ["#7a3f8f", "#a4487f", "#E88977", "#c9558f"]
    nodes = []
    for _ in range(28):
        nodes.append({
            "x": round(rand() * 1200), "y": round(rand() * 1400),
            "r": round(1.6 + rand() * 2.4, 1),
            "color": node_colors[int(rand() * len(node_colors))],
        })
    lines = []
    for n in nodes:
        link_count = 1 + int(rand() * 2)
        for _ in range(link_count):
            other = nodes[int(rand() * len(nodes))]
            if other is not n:
                lines.append({"x1": n["x"], "y1": n["y"], "x2": other["x"], "y2": other["y"], "color": n["color"]})
    return nodes, lines


def _render_background_decoration():
    nodes, lines = _build_neuron_field()
    lines_svg = "".join(
        f'<line x1="{l["x1"]}" y1="{l["y1"]}" x2="{l["x2"]}" y2="{l["y2"]}" stroke="{l["color"]}" stroke-width="1.3"/>'
        for l in lines
    )
    nodes_svg = "".join(
        f'<circle cx="{n["x"]}" cy="{n["y"]}" r="{n["r"]}" fill="{n["color"]}"/>'
        for n in nodes
    )
    loop_radii = [60, 105, 150, 195, 240, 285, 330, 375, 420]
    rings_svg = "".join(
        f'<circle cx="60" cy="580" r="{r}" fill="none" stroke="url(#loopGrad)" '
        f'stroke-width="22" opacity="{max(0.95 - i * 0.07, 0.05):.2f}"/>'
        for i, r in enumerate(loop_radii)
    )

    # Deliberately NOT anchored via .stApp{{position:relative}}: Streamlit relies on
    # its own positioning for .stApp's viewport-fill sizing, and overriding it
    # collapses the whole app to zero height (confirmed by testing). Anchoring to
    # the initial containing block instead means this covers a fixed span of the
    # page (~2400px) rather than dynamically matching exact content height, which
    # is an acceptable tradeoff for a low-opacity decorative layer.
    #
    # No overflow:hidden here: Streamlit's own root already clips horizontal
    # overflow, and adding a second, narrower clip boundary on this wrapper cut
    # the intentionally-oversized blob/spiral off with a visible hard edge
    # partway across the page instead of letting them fade out at the true edge.
    return f"""
    <div style="position:absolute;top:0;left:0;width:100%;height:2400px;pointer-events:none;z-index:0">
      <svg style="position:absolute;inset:0;width:100%;height:100%;opacity:0.16"
           viewBox="0 0 1200 1400" preserveAspectRatio="xMidYMid slice">{lines_svg}{nodes_svg}</svg>
      <div style="position:absolute;top:-220px;right:-220px;width:620px;height:620px;border-radius:50%;
           background:radial-gradient(circle,rgba(232,137,119,0.35),transparent 70%);filter:blur(18px);
           animation:blobDrift 14s ease-in-out infinite"></div>
      <svg style="position:absolute;bottom:-160px;left:-160px;width:640px;height:640px" viewBox="0 0 640 640">
        <defs><linearGradient id="loopGrad" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stop-color="#150F18"/><stop offset="0.4" stop-color="#5c3357"/>
          <stop offset="0.75" stop-color="#c9558f"/><stop offset="1" stop-color="#E88977"/>
        </linearGradient></defs>
        {rings_svg}
        <path d="M60 580 L60 480 M60 580 L140 520 M60 580 L20 490" stroke="#EBE4EC" stroke-width="1.4" opacity="0.5" fill="none"/>
        <circle cx="60" cy="580" r="10" fill="#EBE4EC" opacity="0.8"/>
      </svg>
    </div>
    """


# ---- Page setup & styling --------------------------------------------
st.set_page_config(
    page_title="NeuroLoom",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&family=Roboto+Condensed:wght@400;500;600;700&display=swap');

:root {
    /* Deep Aubergine palette -- swap these six values to retheme the whole app. */
    --bg: #150F18;
    --panel: #221826;       /* sidebar, nav */
    --card: #312336;        /* cards, graph box */
    --card-nested: #3d2b42; /* chips/stat-boxes sitting inside a dark-panel */
    --text-primary: #EBE4EC;
    --text-secondary: #A394A6;
    --accent: #E88977;
    --accent-deep: #c96a58;
    --border: rgba(235,228,236,0.14);
    --border-soft: rgba(235,228,236,0.12);
}

html, body, .stApp { font-family: 'Inter', sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { color: #f0a494; text-decoration: underline; }

@keyframes blobDrift {
    0% { transform: translate(0,0) scale(1); }
    50% { transform: translate(18px,-14px) scale(1.05); }
    100% { transform: translate(0,0) scale(1); }
}

.stApp {
    background: linear-gradient(160deg, var(--bg) 0%, #1c1420 100%);
}

[data-testid="stSidebar"] { background: var(--panel); }
[data-testid="stSidebar"] * { color: var(--text-primary); }
[data-testid="stSidebar"] .disclaimer { color: var(--text-secondary) !important; }

/* Native text inputs don't automatically follow this page's custom palette --
   without this, they fall back to the browser/Streamlit-theme default, which
   can render as illegible dark-on-dark if the viewer's system is already in
   dark mode. Forced explicitly so it never depends on config.toml alone. */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background-color: var(--panel) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-secondary) !important; }

.st-key-navbar {
    background: rgba(34,24,38,0.75); backdrop-filter: blur(8px);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 0.5rem 0.9rem; margin: 0.5rem 0 1.75rem;
}
.topnav-brand { font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.35rem; letter-spacing: -0.02em; }
.topnav-brand .neuro { color: var(--text-primary); }
.topnav-brand .loom {
    background: linear-gradient(120deg, var(--accent), var(--accent-deep));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}

.hero-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 2rem; align-items: center; margin: 1rem 0 2rem; }
.hero-grid h1 {
    margin: 0; font-family: 'Inter', sans-serif; font-weight: 800;
    font-size: clamp(1.9rem, 3.6vw, 2.7rem); line-height: 1.14; letter-spacing: -0.02em; color: var(--text-primary);
}
.hero-grid p.subhead { margin: 1.1rem 0 0; font-size: 1.02rem; line-height: 1.62; color: var(--text-secondary); max-width: 36rem; }

.dark-panel {
    background: var(--card); border: 1px solid var(--border); border-radius: 22px;
    padding: 1.4rem; box-shadow: 0 10px 18px rgba(0,0,0,0.35);
}
.dark-panel .panel-kicker {
    font-family: 'Roboto Condensed', sans-serif; font-size: 0.75rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.7rem;
}

.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.8rem 0 1.6rem; }
.trust-badge {
    font-family: 'Roboto Condensed', sans-serif;
    background: rgba(49,35,54,0.6); border: 1px solid var(--border-soft);
    color: var(--text-primary); padding: 0.32rem 0.85rem; border-radius: 999px; font-size: 0.8rem; white-space: nowrap;
}

.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; margin-bottom: 1.8rem; }
.glass-card {
    background: rgba(49,35,54,0.6); backdrop-filter: blur(6px);
    border: 1px solid var(--border-soft); border-radius: 16px; padding: 1.2rem 1.3rem;
}
.card-num {
    width: 38px; height: 38px; border-radius: 50%; background: rgba(232,137,119,0.14);
    border: 1px solid rgba(232,137,119,0.28); display: flex; align-items: center; justify-content: center;
    font-family: 'Roboto Condensed', sans-serif; font-weight: 700; font-size: 0.95rem; color: var(--accent);
    margin-bottom: 0.7rem;
}
.card-num.cool { background: rgba(163,148,166,0.14); border-color: rgba(163,148,166,0.28); color: var(--text-secondary); }
.card-num.warm { background: rgba(232,137,119,0.14); border-color: rgba(232,137,119,0.28); color: var(--accent); }
.glass-card .card-kicker { font-family: 'Poppins', sans-serif; font-size: 0.76rem; font-weight: 600; color: var(--accent); }
.glass-card h4 { font-family: 'Poppins', sans-serif; margin: 0.35rem 0 0.5rem; font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }
.glass-card p { margin: 0; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55; }
.glass-card a { color: var(--accent); }

.preset-chip {
    font-family: 'Roboto Condensed', sans-serif; font-size: 0.76rem;
    background: var(--card-nested); border: 1px solid var(--border);
    padding: 0.3rem 0.75rem; color: var(--text-primary);
}
.stat-box { background: var(--card-nested); border: 1px solid var(--border); border-radius: 14px; padding: 0.85rem 1rem; }
.stat-box .stat-num { font-family: 'Poppins', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.stat-box .stat-label { font-family: 'Roboto Condensed', sans-serif; font-size: 0.76rem; color: var(--text-primary); opacity: 0.82; margin-top: 0.2rem; }
.stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.75rem; margin-bottom: 1.2rem; }

.legend-chip {
    font-family: 'Roboto Condensed', sans-serif; font-weight: 600;
    display: inline-block; padding: 0.18rem 0.7rem; border-radius: 999px;
    color: #1a1420; font-size: 0.78rem; margin-right: 0.4rem; margin-bottom: 0.3rem;
}
.mono-note {
    background: var(--bg); border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem 1.1rem;
    font-family: ui-monospace, monospace; font-size: 0.78rem; color: var(--text-secondary);
}
.dark-panel .wiki-blurb { color: var(--text-primary); font-size: 0.92rem; line-height: 1.55; margin-bottom: 1.1rem; }
.dark-panel .wiki-blurb a { color: var(--accent); }

.tech-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.75rem; margin-bottom: 2rem; }
.tech-card {
    background: rgba(49,35,54,0.6); border: 1px solid var(--border-soft); border-radius: 12px;
    padding: 0.8rem 1rem;
}
.tech-card.warm { border-left: 3px solid var(--accent); }
.tech-card.cool { border-left: 3px solid var(--text-secondary); }
.tech-card .tech-name { font-family: 'Poppins', sans-serif; font-weight: 700; color: var(--text-primary); font-size: 0.92rem; }
.tech-card .tech-role { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem; line-height: 1.4; }

.page-title { font-family: 'Poppins', sans-serif; font-weight: 800; color: var(--text-primary); font-size: clamp(1.6rem, 3vw, 2.2rem); margin: 1rem 0 0.4rem; letter-spacing: -0.02em; }
.page-subtitle { color: var(--text-secondary); font-size: 0.95rem; max-width: 46rem; margin-bottom: 1.8rem; }
.section-kicker { font-family: 'Poppins', sans-serif; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em; margin-bottom: 0.7rem; }

.pipeline-card { opacity: 1; transform: translateY(0); }
@supports (animation-timeline: view()) {
    .pipeline-card {
        animation: pipelineReveal linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 25%;
    }
}
@keyframes pipelineReveal {
    from { opacity: 0; transform: translateY(28px); }
    to { opacity: 1; transform: translateY(0); }
}

.disclaimer { font-size: 0.8rem; color: var(--text-secondary); border-top: 1px solid var(--border); padding-top: 0.9rem; margin-top: 1.5rem; }

.site-footer {
    margin-top: 3rem; padding-top: 1.6rem; border-top: 1px solid var(--border);
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem;
}
.site-footer .footer-links { display: flex; gap: 1.4rem; flex-wrap: wrap; font-family: 'Roboto Condensed', sans-serif; font-size: 0.85rem; }
.site-footer .footer-meta { display: flex; align-items: center; gap: 0.9rem; font-family: 'Roboto Condensed', sans-serif; font-size: 0.85rem; color: var(--text-secondary); flex-wrap: wrap; }
.site-footer .footer-meta a { display: inline-flex; align-items: center; gap: 0.4rem; }

[data-testid="stMetric"] { background: rgba(235,228,236,0.06); padding: 0.75rem 1rem; border-radius: 10px; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }

.stButton > button, .stLinkButton > a { border-radius: 999px; font-family: 'Roboto Condensed', sans-serif; font-weight: 600; }
.stButton > button[kind="primary"] { background: var(--accent); color: var(--bg); border: none; }
.stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(232,137,119,0.35); }
.stButton > button[kind="secondary"], .stLinkButton > a[kind="secondary"] {
    background: transparent; color: var(--text-primary); border: 1px solid var(--border);
}
.stButton > button[kind="secondary"]:hover, .stLinkButton > a[kind="secondary"]:hover { background: rgba(235,228,236,0.08); }

.st-key-preset_chips .stButton > button {
    background: var(--card-nested); border: 1px solid var(--border); color: var(--text-primary);
    font-size: 0.78rem; padding: 0.3rem 0.7rem;
}
.st-key-preset_chips .stButton > button:hover { background: #4a3550; }

[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] {
    font-family: 'Roboto Condensed', sans-serif !important; font-weight: 600 !important;
    border-radius: 999px !important; background: var(--card-nested) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; padding: 0.4rem 0.95rem !important;
}
[data-baseweb="tab"][aria-selected="true"] { background: var(--accent) !important; border-color: var(--accent) !important; color: var(--bg) !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

@media (max-width: 768px) {
    .hero-grid { grid-template-columns: 1fr; }
    .card-grid, .stat-strip, .tech-grid { grid-template-columns: 1fr; }
    .st-key-navbar { border-radius: 20px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(_render_background_decoration(), unsafe_allow_html=True)


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
    net = Network(height="650px", width="100%", directed=True, notebook=False, bgcolor="#312336", font_color="white")

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


def render_preset_chips(disabled=False):
    """Quick-access preset row (mirrors the sidebar list) so the choice is
    visible without opening the sidebar, especially on mobile."""
    with st.container(key="preset_chips"):
        cols = st.columns(len(DISEASE_PRESETS))
        for col, (label, preset_query) in zip(cols, DISEASE_PRESETS):
            with col:
                if st.button(label, key=f"chip_{preset_query}", width="stretch", disabled=disabled):
                    st.session_state.query = preset_query
                    st.session_state.page = "Home"
                    st.rerun()


# ---- Sidebar ------------------------------------------------------------
if "query" not in st.session_state:
    st.session_state.query = DISEASE_PRESETS[0][1]
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown("### Explore a disease")
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
    nav_logo, nav_brand, nav_home, nav_components, nav_spacer, nav_source = st.columns(
        [0.4, 1.6, 0.9, 1.3, 2.0, 1.5], vertical_alignment="center"
    )
    with nav_logo:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
    with nav_brand:
        st.markdown('<div class="topnav-brand"><span class="neuro">Neuro</span><span class="loom">Loom</span></div>',
                    unsafe_allow_html=True)
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
    hero_text, hero_preview = st.columns([1.15, 1], gap="large")
    with hero_text:
        st.markdown("""
        <div class="hero-grid" style="grid-template-columns:1fr;margin:0">
          <div>
            <h1>See what decades of neurodegenerative disease research actually say</h1>
            <p class="subhead">Pick a disease and watch years of scattered research turn into one
            graph you can actually explore, built straight from PubMed abstracts and active
            clinical trials instead of a pile of browser tabs you'll never finish reading.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            build_clicked = st.button("Build the graph", type="primary", width="stretch")
        with b2:
            if st.button("Learn more", type="secondary", width="stretch"):
                st.session_state.page = "Components"
                st.rerun()
    with hero_preview:
        st.markdown(f"""
        <div class="dark-panel">
          <div style="font-family:'Roboto Condensed',sans-serif;font-size:0.72rem;letter-spacing:0.08em;
               text-transform:uppercase;color:#EBE4EC;opacity:0.8;margin-bottom:0.6rem">Alzheimer's disease &middot; live graph</div>
          {HERO_GRAPH_SVG}
          <div style="font-family:ui-monospace,monospace;font-size:0.7rem;color:#A394A6;margin-top:0.4rem">
          example relationships &mdash; click Build below for the real thing</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="badge-row">' + "".join(
            f'<span class="trust-badge">{label}</span>' for label in TRUST_BADGES
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-grid">' + "".join(
            f'<div class="glass-card"><div class="card-num">{i + 1:02d}</div>'
            f'<div class="card-kicker">{kicker}</div><h4>{title}</h4><p>{body}</p></div>'
            for i, (kicker, title, body) in enumerate(WHY_CARDS)
        ) + '</div>',
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

    with st.container(key="explorer_panel"):
        st.markdown('<div class="dark-panel">', unsafe_allow_html=True)
        kicker = "Your results" if result else "Explorer preview"
        st.markdown(f'<div class="panel-kicker">{kicker}</div>', unsafe_allow_html=True)

        render_preset_chips()

        if not result:
            st.markdown(
                '<div class="mono-note" style="margin-top:1rem">Pick a disease above or in the '
                'sidebar, then click Build the graph. Live PubMed counts, extracted relationships, '
                'clinical trials, and source abstracts will render here.</div>',
                unsafe_allow_html=True,
            )
        else:
            abstracts = result["pubmed_result"]["abstracts"]
            pubmed_total = result["pubmed_result"]["total_count"]
            trials = result["trials_result"]["trials"]
            trials_total = result["trials_result"]["total_count"]
            wiki = result["wiki"]

            st.markdown(f"""
            <div class="stat-strip" style="margin-top:1rem">
              <div class="stat-box"><div class="stat-num">{format_count_plus(pubmed_total)}</div>
                <div class="stat-label">PubMed records matching "{result['query']}"</div></div>
              <div class="stat-box"><div class="stat-num">{format_count_plus(trials_total)}</div>
                <div class="stat-label">Registered clinical trials matching this search</div></div>
              <div class="stat-box"><div class="stat-num">{len(abstracts)}</div>
                <div class="stat-label">Abstracts analyzed in depth for this graph</div></div>
            </div>
            """, unsafe_allow_html=True)

            if wiki:
                st.markdown(
                    f'<div class="wiki-blurb"><strong>{wiki["title"]}.</strong> {wiki["extract"]} '
                    f'<a href="{wiki["url"]}" target="_blank">Read more</a></div>',
                    unsafe_allow_html=True,
                )

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
                    ["Graph", "Relationships", "Clinical trials", "Sources"]
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

        st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-kicker" style="color:#E88977;">The pipeline &mdash; scroll to reveal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-grid">' + "".join(
            f'<div class="glass-card pipeline-card"><div class="card-num {"cool" if i % 2 else "warm"}">{num}</div>'
            f'<h4>{title}</h4><p>{desc}</p></div>'
            for i, (num, title, desc) in enumerate(PIPELINE_STEPS)
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker" style="color:#A394A6;">Built with</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tech-grid">' + "".join(
            f'<div class="tech-card {"cool" if i % 2 else "warm"}"><div class="tech-name">{name}</div>'
            f'<div class="tech-role">{role}</div></div>'
            for i, (name, role) in enumerate(TECH_STACK)
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker" style="color:#E88977;">Where the data comes from</div>', unsafe_allow_html=True)
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

# ---- Footer (both pages) --------------------------------------------------
st.markdown(f"""
<div class="site-footer">
  <div class="footer-links">
    <a href="https://github.com/lmarshall-boop/Knowledge-Graph" target="_blank">GitHub</a>
    <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">PubMed</a>
    <a href="https://clinicaltrials.gov/" target="_blank">ClinicalTrials.gov</a>
    <a href="https://en.wikipedia.org/api/rest_v1/" target="_blank">Wikipedia API</a>
  </div>
  <div class="footer-meta">
    <span>&copy; 2026 NeuroLoom &middot; research exploration, not medical advice</span>
    <a href="https://instagram.com/lianali4na" target="_blank">{INSTAGRAM_SVG}@lianali4na</a>
  </div>
</div>
""", unsafe_allow_html=True)
