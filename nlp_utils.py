"""
Biomedical NLP layer for the Knowledge Graph Explorer.
--------------------------------------------------------
Adds three things the original SVO extractor didn't have:

1. Entity typing — classifies each extracted span as DISEASE, GENE_PROTEIN,
   DRUG_CHEMICAL, ANATOMY, or PROCESS using a curated neurodegenerative-disease
   lexicon, so the graph can color-code nodes instead of drawing every node
   the same color.
2. Synonym normalization — collapses "AD", "Alzheimer disease", and
   "Alzheimer's disease" into one canonical node instead of three.
3. Negation detection — flags relationships like "X does not activate Y" so
   they can be rendered differently from positive claims instead of being
   silently treated the same way.
"""

import re

import spacy

# ---- Entity lexicon --------------------------------------------------
# Longest-match-first term lists per category. This is intentionally a
# lightweight, dependency-free alternative to a full biomedical NER model
# (e.g. scispaCy) so the app keeps deploying cleanly on Streamlit Community
# Cloud. Swap in scispaCy's en_ner_bc5cdr_md if you want model-based typing.
ENTITY_LEXICON = {
    "DISEASE": [
        "alzheimer's disease", "alzheimer disease", "parkinson's disease",
        "parkinson disease", "amyotrophic lateral sclerosis",
        "huntington's disease", "huntington disease",
        "frontotemporal dementia", "lewy body dementia", "multiple sclerosis",
        "motor neuron disease", "parkinsonism", "dementia", "als",
    ],
    "GENE_PROTEIN": [
        "amyloid-beta", "amyloid beta", "amyloid precursor protein", "tau",
        "alpha-synuclein", "a-synuclein", "huntingtin", "apoe", "app",
        "psen1", "psen2", "lrrk2", "gba", "c9orf72", "sod1", "tdp-43",
        "parkin", "pink1", "prion protein",
    ],
    "DRUG_CHEMICAL": [
        "levodopa", "donepezil", "memantine", "riluzole", "edaravone",
        "aducanumab", "lecanemab", "dopamine", "acetylcholine", "glutamate",
        "rivastigmine", "galantamine",
    ],
    "ANATOMY": [
        "substantia nigra", "hippocampus", "cortex", "basal ganglia",
        "striatum", "neuron", "microglia", "synapse", "brainstem",
        "cerebellum", "axon",
    ],
    "PROCESS": [
        "neuroinflammation", "aggregation", "phosphorylation", "apoptosis",
        "autophagy", "oxidative stress", "neurodegeneration",
        "mitochondrial dysfunction", "protein misfolding",
    ],
}

# Sorted longest-first so "amyloid precursor protein" matches before "amyloid".
_LEXICON_FLAT = sorted(
    ((term, category) for category, terms in ENTITY_LEXICON.items() for term in terms),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

ENTITY_COLORS = {
    "DISEASE": "#e8746b",
    "GENE_PROTEIN": "#6ba5e8",
    "DRUG_CHEMICAL": "#7fd18f",
    "ANATOMY": "#d9b45c",
    "PROCESS": "#b38ce8",
    "OTHER": "#b5b8c2",
}

SYNONYM_MAP = {
    "ad": "alzheimer's disease",
    "alzheimer disease": "alzheimer's disease",
    "pd": "parkinson's disease",
    "parkinson disease": "parkinson's disease",
    "als": "amyotrophic lateral sclerosis",
    "hd": "huntington's disease",
    "huntington disease": "huntington's disease",
    "ftd": "frontotemporal dementia",
    "lbd": "lewy body dementia",
    "ms": "multiple sclerosis",
    "amyloid beta": "amyloid-beta",
    "amyloid precursor protein": "app",
    "a-synuclein": "alpha-synuclein",
}

NEGATION_CUES = {"no", "not", "n't", "without", "never", "fail", "lack", "absent"}


def classify_entity(text):
    """Return the lexicon category for a span, or 'OTHER' if unmatched."""
    lowered = text.lower()
    for term, category in _LEXICON_FLAT:
        if term in lowered:
            return category
    return "OTHER"


def normalize_text(text):
    """Lowercase, strip leading articles, and canonicalize known synonyms
    so 'the plaque phenotype' / 'plaque phenotype' and 'AD' / 'Alzheimer's
    disease' collapse into the same graph node."""
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", text)
    return SYNONYM_MAP.get(text, text)


def get_span_for_token(token):
    for chunk in token.doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            return chunk.text
    return token.text


def is_negated(verb_token):
    """Check the verb's direct children for a dependency-parsed negation
    (handles 'does not activate'), then fall back to scanning the surface
    text of the sentence for a negation cue word (catches 'fails to
    activate', 'lack of activation', which the neg dep tag often misses)."""
    for child in verb_token.children:
        if child.dep_ == "neg":
            return True
    sent_text = verb_token.sent.text.lower()
    return any(cue in sent_text for cue in NEGATION_CUES)


def extract_triples(text, nlp, source_id=None):
    """Extract (subject, verb, object) relationships from text, enriched
    with entity type, negation, and provenance (source_id + sentence) so
    the UI can show *why* an edge exists, not just that it exists."""
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
            subj_text = normalize_text(get_span_for_token(subj))
            obj_text = normalize_text(get_span_for_token(obj))
            if subj_text == obj_text:
                continue
            triples.append({
                "subject": subj_text,
                "subject_type": classify_entity(subj_text),
                "verb": verb.lemma_,
                "object": obj_text,
                "object_type": classify_entity(obj_text),
                "negated": is_negated(verb),
                "sentence": sent.text.strip(),
                "source_id": source_id,
            })
    return triples
