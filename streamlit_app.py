"""
Flood Adaptation ABM — Interactive Dashboard

Two modes:
  - Research Mode: abstract/synthetic settings with noise toggles
  - Case Study Mode: upload location data and flood scenarios

Erfan Amini, Center for Climate Systems Research (CCSR), Columbia University
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import sys
import os
import time

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flood Adaptation ABM",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for aesthetics
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Color palette ---------- */
:root {
    --primary: #1B4F72;
    --primary-light: #2E86C1;
    --accent: #E67E22;
    --accent-light: #F5CBA7;
    --bg-dark: #1B2631;
    --bg-card: #F8F9F9;
    --text-dark: #1C2833;
    --text-muted: #5D6D7E;
    --border: #D5D8DC;
    --success: #27AE60;
}

/* ---------- Header banner ---------- */
.main-banner {
    background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 60%, #3498DB 100%);
    padding: 2.2rem 2.5rem 1.8rem 2.5rem;
    border-radius: 0 0 18px 18px;
    margin: -1rem -1rem 1.8rem -1rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 30%, rgba(255,255,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.main-banner h1 {
    margin: 0 0 0.25rem 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: white !important;
}
.main-banner p {
    margin: 0;
    font-size: 1.05rem;
    opacity: 0.92;
    color: #D6EAF8 !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2631 0%, #212F3D 100%);
}
section[data-testid="stSidebar"] * {
    color: #EBF5FB !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #EBF5FB !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #EBF5FB 0%, #D4E6F1 100%);
    border: 1px solid #AED6F1;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(27,79,114,0.08);
}
div[data-testid="stMetric"] label {
    color: var(--primary) !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--text-dark) !important;
    font-weight: 700 !important;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 1.22rem !important;
    padding: 0.7rem 1.4rem !important;
    letter-spacing: 0.2px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    padding: 1.5rem 1rem 1rem 1rem;
    margin-top: 3rem;
    border-top: 2px solid #D5D8DC;
    color: #5D6D7E;
    font-size: 0.88rem;
    line-height: 1.7;
}
.app-footer .author-name {
    font-weight: 700;
    color: #1B4F72;
    font-size: 0.95rem;
}
.app-footer .affiliation {
    color: #2E86C1;
}

/* ---------- Section headers in docs ---------- */
.doc-section-header {
    background: linear-gradient(90deg, #1B4F72 0%, #2E86C1 100%);
    color: white !important;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    margin: 1.5rem 0 0.8rem 0;
    font-size: 1.15rem;
    font-weight: 600;
}

/* ---------- Info cards in docs ---------- */
.info-card {
    background: #EBF5FB;
    border-left: 4px solid #2E86C1;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
}

/* ---------- Buttons ---------- */
button[kind="primary"] {
    background: linear-gradient(135deg, #1B4F72, #2E86C1) !important;
    border: none !important;
    font-weight: 600 !important;
}

/* ---------- License box ---------- */
.license-box {
    background: #F8F9F9;
    border: 1px solid #D5D8DC;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #5D6D7E;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Import model modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
import AAA_model_v12 as M
from FFF_neighborhood import identify_neighborhoods

# ---------------------------------------------------------------------------
# Matplotlib — LaTeX-style configuration
# ---------------------------------------------------------------------------
FS = 11
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Palatino Linotype", "Palatino",
                    "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "font.size": FS,
    "axes.titlesize": FS + 1,
    "axes.labelsize": FS,
    "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 2,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Title banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-banner">
    <h1>🌊 Social Learning and Flood Adaptation — Agent-Based Model</h1>
    <p>Bayesian Belief Updating for Household Retrofit Decisions under Flood Risk</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR — MODE + GENERAL SETTINGS
# ============================================================================

st.sidebar.title("🌊 Flood Adaptation ABM")
st.sidebar.caption("Bayesian Belief Updating Model")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    ["Research Mode", "Case Study Mode"],
    help=("**Research Mode**: Synthetic spatial layouts, GEV floods, "
          "noise toggles.\n\n"
          "**Case Study Mode**: Upload location CSV and flood data."),
)

st.sidebar.divider()
st.sidebar.header("⚙️ General Settings")

time_steps = st.sidebar.slider("Time Steps", 10, 500, 100, step=10)
n_agents = st.sidebar.slider("Number of Agents", 20, 1000, 200, step=10)

st.sidebar.divider()
st.sidebar.header("📐 Bayesian Updating")

initial_belief = st.sidebar.slider(
    "Initial Belief  —  $P(H_1)$", 0.01, 0.50, 0.05, 0.01)
lambda_flood = st.sidebar.slider(
    "Flood Experience  —  $\\lambda_{\\mathrm{flood}}$",
    1.0, 3.0, 1.20, 0.01)
lambda_social = st.sidebar.slider(
    "Proximity Learning  —  $\\lambda_{\\mathrm{social}}$",
    1.0, 5.0, 1.50, 0.01)
lambda_similarity = st.sidebar.slider(
    "Similarity Learning  —  $\\lambda_{\\mathrm{similarity}}$",
    1.0, 10.0, 3.00, 0.1)

st.sidebar.divider()
st.sidebar.header("🎯 PMT Threshold")

pmt_mean = st.sidebar.slider("Threshold Mean", 0.10, 0.90, 0.50, 0.01)

if mode == "Research Mode":
    pmt_noise_on = st.sidebar.checkbox(
        "Threshold Heterogeneity", value=False,
        help="Draw individual thresholds from a truncated Normal.")
    if pmt_noise_on:
        pmt_std = st.sidebar.slider("Std Dev", 0.01, 0.20, 0.10, 0.01)
        pmt_low = st.sidebar.slider("Lower Bound", 0.10, pmt_mean, 0.45, 0.01)
        pmt_high = st.sidebar.slider("Upper Bound", pmt_mean, 0.95, 0.80, 0.01)
    else:
        pmt_std, pmt_low, pmt_high = 0.0, pmt_mean, pmt_mean
else:
    pmt_std = st.sidebar.slider("Std Dev", 0.00, 0.20, 0.00, 0.01)
    pmt_low = st.sidebar.slider("Lower Bound", 0.10, 0.90, 0.50, 0.01)
    pmt_high = st.sidebar.slider("Upper Bound", 0.10, 0.95, 0.50, 0.01)

# --- Case Study file uploads ---
uploaded_csv = None
uploaded_flood = None
if mode == "Case Study Mode":
    st.sidebar.divider()
    st.sidebar.header("📁 Upload Data")
    uploaded_csv = st.sidebar.file_uploader(
        "Location CSV (columns: x, y, z)", type=["csv"])
    uploaded_flood = st.sidebar.file_uploader(
        "Flood Time Series CSV (optional)", type=["csv"],
        help="Column `flood_level`, one value per time step.")


# ============================================================================
# MAIN TABS
# ============================================================================

tab_doc, tab_settings, tab_sim, tab_results = st.tabs(
    ["📖 Documentation", "🔧 Advanced Settings",
     "▶️ Run Simulation", "📊 Results"])


# ============================================================================
# TAB: DOCUMENTATION (comprehensive, matching the Word file)
# ============================================================================

with tab_doc:

    # --- 1. Motivation ---
    st.markdown('<div class="doc-section-header">1 &nbsp; Motivation</div>',
                unsafe_allow_html=True)
    st.markdown("""
Coastal communities face escalating flood risks from climate change.
Public investment in protection happens on long time frames, is highly
uncertain, and leaves residual risk. Mitigating rising recovery costs
in the short term requires action by individuals and communities, yet
household investment in adaptation remains remarkably low (Amini et al.,
2025).

Empirical evidence from household surveys along the U.S. Eastern Seaboard
shows that residents are aware of increasing flood risk — 61% expect
more frequent flooding and 71% expect rising expenses — but they lack
information about the benefits and costs of home retrofits. Only about
25% of residents have taken protective action despite flood experience
and awareness of risk (Amini et al., 2025). This gap between risk
awareness and protective action motivates the present model.
""")

    # --- 2. Overview ---
    st.markdown('<div class="doc-section-header">2 &nbsp; Overview</div>',
                unsafe_allow_html=True)
    st.markdown("""
This agent-based model simulates household flood-adaptation decisions
using Bayesian belief updating in odds form (Jaynes, 2003, Ch. 4; Kass &
Raftery, 1995). Agents maintain a probability $P(H_1)$ representing their
belief that their situation warrants retrofitting. This belief is
updated each year through three evidence channels: personal flood
experience, proximity-based social observation, and similarity-based
social learning. The model is built on the Mesa framework (Kazil et al.,
2020) for Python.

The core mechanism is binary hypothesis testing. Each agent considers
two subjective assessments: **$H_1$ = "my situation warrants
retrofitting"** and **$H_0$ = "my situation does not warrant
retrofitting."** Evidence shifts belief by multiplying the agent's odds
via Bayes factors. Each evidence channel contributes a separate Bayes
factor. When $P(H_1)$ exceeds a heterogeneous threshold drawn from a
truncated Normal distribution (Rogers, 1975), the agent retrofits
permanently.

The model examines what learning processes — experience-based versus
social — are consistent with observed limited investment in adaptation,
and how household heterogeneity influences the diffusion of adaptation
behavior through social networks (Amini et al., 2025).
""")


    # --- 3. Theoretical Framework ---
    st.markdown('<div class="doc-section-header">3 &nbsp; Theoretical Framework</div>',
                unsafe_allow_html=True)

    st.subheader("3.1 &nbsp; Bayesian Belief Updating — Full Formulation")
    st.markdown("""
Each agent $i$ maintains a subjective belief $P_i(H_1) \\in [0, 1]$,
where $H_1$ = "my situation warrants retrofitting" and $H_0$ = "my
situation does not warrant retrofitting." The updating mechanism
proceeds through three algebraic steps at each evidence event
(Jaynes, 2003, Ch. 4; Kass & Raftery, 1995):

**Step 1 — Convert probability to odds.** The odds form provides a
natural multiplicative framework for sequential evidence accumulation:
""")
    st.latex(r"O_i = \frac{P_i(H_1)}{1 - P_i(H_1)}")
    st.markdown("""
For example, an initial belief of $P_i(H_1) = 0.05$ corresponds to
odds $O_i = 0.05 / 0.95 \\approx 0.0526$, meaning the agent considers
$H_0$ roughly 19 times more likely than $H_1$.

**Step 2 — Multiply odds by the Bayes factor.** The Bayes factor
$\\lambda$ quantifies the evidential strength of a single observation,
defined as the likelihood ratio (Kass & Raftery, 1995):
""")
    st.latex(r"\lambda = \frac{P(\text{evidence} \mid H_1)}{P(\text{evidence} \mid H_0)}")
    st.markdown("""
The posterior odds are obtained by multiplying the prior odds by the
Bayes factor:
""")
    st.latex(r"O_i^{\,\text{posterior}} = O_i^{\,\text{prior}} \times \lambda")
    st.markdown("""
A Bayes factor $\\lambda > 1$ shifts belief toward $H_1$; $\\lambda = 1$
is uninformative; $\\lambda < 1$ would shift toward $H_0$ (not used in
this model, since all evidence channels are non-negative). This model
uses three Bayes factors — $\\lambda_{\\text{flood}}$,
$\\lambda_{\\text{social}}$, and $\\lambda_{\\text{similarity}}$ —
corresponding to the three evidence channels. When multiple channels
produce evidence in the same time step, their factors compose
multiplicatively:
""")
    st.latex(r"O_i^{\,\text{posterior}} = O_i^{\,\text{prior}} \times \lambda_{\text{flood}} \times \lambda_{\text{social}} \times \lambda_{\text{similarity}}^{\,S(i,j)}")

    st.markdown("""
**Step 3 — Convert posterior odds back to probability:**
""")
    st.latex(r"P_i^{\,\text{posterior}}(H_1) = \frac{O_i^{\,\text{posterior}}}{1 + O_i^{\,\text{posterior}}}")
    st.markdown("""
This three-step procedure is algebraically equivalent to the standard
form of Bayes' theorem. Its advantage is that each evidence source is
characterized by a single scalar (the Bayes factor), and sequential
updates reduce to repeated multiplication in the odds domain.
""")

    st.subheader("3.2 &nbsp; Channel 1: Personal Flood Experience")
    st.markdown("""
Each time step, a global flood level $f_t$ is drawn from a Generalized
Extreme Value distribution (Coles, 2001). Agent $i$ with elevation $z_i$
is flooded if $f_t > z_i$. The update rule is asymmetric, following the
availability heuristic (Tversky & Kahneman, 1974) — flood events are
psychologically salient while dry years are cognitively inert:
""")
    st.latex(r"""
\lambda_{\text{flood},i}^{(t)} =
\begin{cases}
\lambda_{\text{flood}} & \text{if } f_t > z_i \quad \text{(flooded)} \\
1 & \text{if } f_t \leq z_i \quad \text{(not flooded; no update)}
\end{cases}
""")
    st.markdown("""
With the default $\\lambda_{\\mathrm{flood}} = 1.20$, each flood
increases the odds by 20%. An agent must experience several floods
before belief approaches the decision threshold, consistent with
observed low adoption rates despite repeated flooding (Amini et al.,
2025).
""")

    st.subheader("3.3 &nbsp; Channel 2: Proximity-Based Social Learning")
    st.markdown("""
Network connections are binary: agents $i$ and $j$ are connected if
their Euclidean distance $d(i,j) \\leq$ `DISTANCE_THRESHOLD`, and
unconnected otherwise. No distance decay function is applied. Let
$\\mathcal{N}_i^{(t)}$ denote the set of connected neighbors of agent
$i$ that are **newly observed** as having retrofitted at time $t$ (each
neighbor is counted once — one-shot learning). For each
$j \\in \\mathcal{N}_i^{(t)}$:
""")
    st.latex(r"\lambda_{\text{social},j} = \lambda_{\text{social}}")
    st.markdown("""
This captures proximity-based social influence: living near someone who
has taken action provides a direct signal about the appropriateness of
retrofitting, regardless of demographic similarity (McPherson et al.,
2001).
""")

    st.subheader("3.4 &nbsp; Channel 3: Similarity-Based Social Learning")
    st.markdown("""
Spatial neighborhoods are identified using DBSCAN density-based
clustering (Ester et al., 1996) applied to agent positions with
$\\varepsilon$ = `DISTANCE_THRESHOLD`. Agents in the same cluster share a
neighborhood. Connector agents and isolated points (DBSCAN label $-1$)
are excluded from this channel.

For each newly retrofitted neighbor $j$ that shares the same
neighborhood as agent $i$, the Jaccard similarity $S(i,j)$ between their
categorical attribute vectors (Jaccard, 1912) modulates the Bayes
factor:
""")
    st.latex(r"""
S(i,j) = \frac{\left|\{k : a_{i,k} = a_{j,k}\}\right|}{K}
""")
    st.markdown("""
where $K$ is the total number of attributes and $a_{i,k}$ is agent $i$'s
value for attribute $k$. The similarity-scaled Bayes factor is:
""")
    st.latex(r"\lambda_{\text{similarity},j} = \lambda_{\text{similarity}}^{\,S(i,j)}")
    st.markdown("""
When $S = 1$ (identical attributes), the full factor applies. When
$S = 0$ (no shared attributes), $\\lambda = 1$ (no update). This captures
the homophily mechanism (McPherson et al., 2001): agents are more
influenced by those who share their characteristics. Social learning
rate depends on attribute similarity between agents, reflecting
empirical patterns where households adopt behaviors of similar neighbors
(Amini et al., 2025).

**Combined social update.** A neighbor who is both connected **and** in
the same neighborhood triggers both Channels 2 and 3 multiplicatively.
A neighbor who is connected but in a different neighborhood triggers
only Channel 2.
""")

    st.subheader("3.5 &nbsp; Combined Update — Single Time Step")
    st.markdown("""
At each time step $t$, agent $i$'s complete belief update combines all
three channels. Let $\\mathcal{N}_i^{(t)}$ be the set of newly observed
retrofitted neighbors, and let $\\mathcal{N}_i^{\\text{same},(t)}
\\subseteq \\mathcal{N}_i^{(t)}$ be those in the same DBSCAN
neighborhood. The full update is:
""")
    st.latex(r"""
O_i^{(t)} = O_i^{(t-1)}
\;\times\; \lambda_{\text{flood},i}^{(t)}
\;\times\; \prod_{j \in \mathcal{N}_i^{(t)}} \lambda_{\text{social}}
\;\times\; \prod_{j \in \mathcal{N}_i^{\text{same},(t)}}
\lambda_{\text{similarity}}^{\,S(i,j)}
""")
    st.markdown("""
followed by conversion back to probability:
$P_i^{(t)}(H_1) = O_i^{(t)} / (1 + O_i^{(t)})$.

Because all Bayes factors are $\\geq 1$ and non-flood years contribute a
factor of 1, belief is **monotonically non-decreasing** over time.
""")

    st.subheader("3.6 &nbsp; Decision Rule — Protection Motivation Theory")
    st.markdown("""
Each agent $i$ has an individual decision threshold
$\\theta_i$ drawn at initialization from a truncated Normal distribution:
""")
    st.latex(r"""
\theta_i \sim \mathcal{N}(\mu_\theta,\, \sigma_\theta)
\;\Big|_{\;\theta_{\min}}^{\;\theta_{\max}}
""")
    st.markdown("""
When $P_i(H_1) \\geq \\theta_i$, the agent retrofits permanently and
exits the learning loop. Retrofit is irreversible. Heterogeneous
thresholds create timing diversity among agents with identical exposure
histories, representing variation in self-efficacy and perceived
response costs (Rogers, 1975).
""")

    st.subheader("3.7 &nbsp; Flood Generation — GEV Distribution")
    st.markdown("""
Annual flood levels are sampled from a Generalized Extreme Value (GEV)
distribution with CDF:
""")
    st.latex(r"""
F(x) = \exp\!\left\{
-\left[1 + \xi\left(\frac{x - \mu}{\sigma}\right)\right]^{-1/\xi}
\right\}
""")
    st.markdown("""
where $\\mu$ is the location parameter, $\\sigma > 0$ the scale, and
$\\xi$ the shape. The parameters are fitted to user-defined return-period
/ flood-level pairs via constrained optimization (Hosking & Wallis,
1997). Each year, a single global flood level $f_t$ is drawn and clipped
to $[0, 1]$.
""")

    st.subheader("3.8 &nbsp; Edge Attributes — Jaccard Similarity")
    st.markdown("""
Each network edge stores the Jaccard similarity
$S(i,j) = |\\{k : a_{i,k} = a_{j,k}\\}| / K$. Connections are binary:
agents within `DISTANCE_THRESHOLD` are connected; agents beyond it are
not. No distance decay weighting is applied to edges.
""")

    st.subheader("3.9 &nbsp; Neighborhood Identification — DBSCAN")
    st.markdown("""
Spatial neighborhoods are identified using DBSCAN with
$\\varepsilon$ = `DISTANCE_THRESHOLD` and `min_samples` =
`DBSCAN_MIN_SAMPLES`. The clustering is computed once at initialization.
Agents in the same cluster share a neighborhood label. Agents classified
as noise (label = $-1$) have no neighborhood and do not participate in
similarity-based learning. These are typically connector agents bridging
two neighborhoods.
""")

    # --- 4. Model Workflow Flowchart ---
    st.markdown('<div class="doc-section-header">4 &nbsp; Model Workflow</div>',
                unsafe_allow_html=True)
    st.markdown("""
The following flowchart summarizes the initialization procedure and the
annual time-step loop executed by the model.
""")
    st.graphviz_chart(r'''
    digraph workflow {
        graph [
            rankdir=TB, fontname="Helvetica", fontsize=10,
            bgcolor="transparent", pad=0.2, nodesep=0.25, ranksep=0.35,
            size="6.5,7.5", ratio="compress"
        ];
        node [
            shape=box, style="rounded,filled", fontname="Helvetica",
            fontsize=8, fillcolor="#EBF5FB", color="#1B4F72",
            penwidth=1.0, margin="0.12,0.06"
        ];
        edge [fontname="Helvetica", fontsize=7, color="#2E86C1",
              penwidth=0.9, arrowsize=0.6];

        /* ---- Initialization ---- */
        subgraph cluster_init {
            label=<<B>Initialization (once at t = 0)</B>>;
            labeljust=l; fontname="Helvetica"; fontsize=9;
            style="dashed,rounded"; color="#1B4F72";
            bgcolor="#F8F9F9"; penwidth=1.0;

            S  [label="Generate Agent\nPositions & Elevations\n(FFF_spatial.py)"];
            A  [label="Generate Agent\nCategorical Attributes\n(FFF_attributes.py)"];
            N  [label="DBSCAN Neighborhood\nIdentification\n(FFF_neighborhood.py)"];
            G  [label="Build Binary Social Network\nEdges where d(i,j) ≤ threshold\n(FFF_network.py)"];
            AG [label="Create Agents\nSet P(H₁)₀ , draw θᵢ ~ N(μ,σ)"];
            FL [label="Fit GEV Distribution\nto Return-Period Pairs\n(FFF_flood.py)"];

            S -> A -> N -> G -> AG -> FL;
        }

        /* ---- Annual loop ---- */
        subgraph cluster_loop {
            label=<<B>Annual Time-Step Loop (t = 1, 2, …, T)</B>>;
            labeljust=l; fontname="Helvetica"; fontsize=9;
            style="dashed,rounded"; color="#1B4F72";
            bgcolor="#FDFEFE"; penwidth=1.0;

            F1 [label=<<B>Sample Flood Level</B><BR/>f<SUB>t</SUB> ~ GEV(μ, σ, ξ), clip to [0,1]>,
                fillcolor="#D6EAF8"];
            C1 [label=<<B>Channel 1 — Flood Experience</B><BR/>If f<SUB>t</SUB> &gt; z<SUB>i</SUB> : odds × λ<SUB>flood</SUB><BR/>Else: no update>,
                fillcolor="#FADBD8"];
            C2 [label=<<B>Channel 2 — Proximity Learning</B><BR/>For each new retrofitted neighbor j:<BR/>odds × λ<SUB>social</SUB>>,
                fillcolor="#D5F5E3"];
            C3 [label=<<B>Channel 3 — Similarity Learning</B><BR/>If same DBSCAN neighborhood:<BR/>odds × λ<SUB>similarity</SUB><SUP> S(i,j)</SUP>>,
                fillcolor="#FEF9E7"];
            CV [label=<<B>Convert Back to Probability</B><BR/>P(H₁) = O / (1 + O)>,
                fillcolor="#EBF5FB"];
            DEC [label=<<B>Decision Rule (PMT)</B><BR/>If P(H₁) ≥ θ<SUB>i</SUB> → Retrofit permanently>,
                 fillcolor="#E8DAEF"];
            DC [label="Collect Data\n(agent-level & model-level metrics)",
                fillcolor="#EBF5FB"];

            F1 -> C1 -> C2 -> C3 -> CV -> DEC -> DC;
        }

        /* ---- Connections ---- */
        FL  -> F1  [style=bold, label="  begin loop ", color="#E67E22",
                     penwidth=1.0];
        DC  -> F1  [style=dashed, label=" next t  ", color="#7F8C8D",
                     constraint=false];

        /* ---- Output ---- */
        OUT [label=<<B>Outputs</B><BR/>Adoption curves · Belief evolution<BR/>Network maps · Exported CSV>,
             fillcolor="#D4E6F1", shape=box, style="rounded,filled"];
        DC  -> OUT [style=bold, label="  t = T (end) ", color="#E67E22",
                     penwidth=1.0];
    }
    ''', use_container_width=False)

    # --- 5. Architecture ---
    st.markdown('<div class="doc-section-header">5 &nbsp; Architecture</div>',
                unsafe_allow_html=True)
    st.markdown("""
| File | Description |
|:---|:---|
| `AAA_model.py` | Main model: parameters, agents, three-channel Bayesian updating, simulation loop, visualization |
| `FFF_flood.py` | GEV flood generation (Coles, 2001), clipped to [0, 1] |
| `FFF_spatial.py` | Agent positions and elevations (grid, random, or CSV) |
| `FFF_attributes.py` | Agent attributes and Jaccard similarity (Jaccard, 1912) |
| `FFF_network.py` | Binary social network: edges within distance threshold |
| `FFF_neighborhood.py` | DBSCAN neighborhood identification (Ester et al., 1996) |

*All user-tunable parameters are defined only in `AAA_model.py`.
Helper modules have no module-level defaults; all values are passed from
the main file.*
""")

    # --- 6. How It Works ---
    st.markdown('<div class="doc-section-header">6 &nbsp; How It Works</div>',
                unsafe_allow_html=True)

    st.subheader("6.1 &nbsp; Initialization")
    st.markdown("""
1. Generate agent positions and elevations (`FFF_spatial.py`).
2. Generate agent attributes (`FFF_attributes.py`).
3. Identify neighborhoods via DBSCAN (`FFF_neighborhood.py`).
4. Build binary social network: edges between agents within
   `DISTANCE_THRESHOLD` (`FFF_network.py`). Each edge stores Jaccard similarity.
5. Create agents with initial belief $P(H_1)$ = `INITIAL_BELIEF`,
   individual PMT threshold from $\\mathcal{N}(\\mu, \\sigma)[\\text{low}, \\text{high}]$,
   and neighborhood label.
6. Fit GEV distribution to return period / flood level pairs (`FFF_flood.py`).
""")

    st.subheader("6.2 &nbsp; Each Time Step (Year)")
    st.markdown("""
1. Sample annual flood level from GEV (Coles, 2001), clipped to [0, 1].
2. **Channel 1 (personal):** Each non-retrofitted agent checks whether
   flooded. If flooded, odds × $\\lambda_{\\mathrm{flood}}$. If not flooded,
   no update (Tversky & Kahneman, 1974).
3. **Channels 2 and 3 (social):** Each non-retrofitted agent checks connected
   neighbors. For each neighbor newly observed as retrofitted:
   (a) odds × $\\lambda_{\\mathrm{social}}$ (proximity signal);
   (b) if both agents share the same DBSCAN neighborhood,
   odds × $\\lambda_{\\mathrm{similarity}}^{S(i,j)}$ (similarity signal).
   Each neighbor is counted once.
4. **Decision:** If $P(H_1) \\geq$ agent's PMT threshold, agent retrofits permanently.
5. Data collection: agent-level and model-level metrics recorded.
""")

    # --- 7. Understanding the Bayes Factors ---
    st.markdown('<div class="doc-section-header">7 &nbsp; Understanding the Bayes Factors</div>',
                unsafe_allow_html=True)
    st.markdown("""
Each evidence channel has a Bayes factor that measures how much the
evidence supports $H_1$ over $H_0$ (Kass & Raftery, 1995). Factors
greater than 1 push belief toward $H_1$. A factor of 1 is uninformative.

| Factor | Default | Interpretation |
|:---|:---:|:---|
| $\\lambda_{\\mathrm{flood}}$ | 1.20 | Each flood multiplies odds by 20%. Multiple floods needed to reach threshold. |
| $\\lambda_{\\mathrm{social}}$ | 1.50 | Each connected neighbor who retrofits multiplies odds by 50%. Proximity signal. |
| $\\lambda_{\\mathrm{similarity}}$ | 3.00 | At $S=1$: triples odds. At $S=0.5$: effective factor = $3.00^{0.5} \\approx 1.73$. At $S=0$: factor = 1 (no effect). Homophily signal. |

A neighbor who is both connected and in the same neighborhood triggers
both Channels 2 and 3. A neighbor who is connected but in a different
neighborhood (or a connector agent) triggers only Channel 2.
""")

    # --- 8. Key Properties ---
    st.markdown('<div class="doc-section-header">8 &nbsp; Key Properties</div>',
                unsafe_allow_html=True)
    st.markdown("""
**Three separable evidence channels.** Flood experience, proximity, and
similarity contribute independently. Each can be turned off by setting
its Bayes factor to 1.

**Binary connections.** Agents are either connected or not. No continuous
distance decay. This simplifies the network and makes the proximity
signal unambiguous.

**Similarity operates within neighborhoods.** The DBSCAN clustering step
identifies spatial communities. The similarity channel only applies
between agents in the same community, reflecting that homophily effects
are strongest among agents who share a local context.

**Belief is monotonically non-decreasing.** Safe years produce no update.
All three channels can only increase belief (all Bayes factors ≥ 1).

**Heterogeneous thresholds create timing diversity.** Agents with the
same exposure may adopt at different times due to individual PMT thresholds.
""")

    # --- 9. Parameters ---
    st.markdown('<div class="doc-section-header">9 &nbsp; Parameters</div>',
                unsafe_allow_html=True)

    st.subheader("9.1 &nbsp; Bayesian Belief Parameters")
    st.markdown("""
| Parameter | Default | Description |
|:---|:---:|:---|
| `INITIAL_BELIEF` | 0.05 | Prior $P(H_1)$. Agents begin with mild awareness. |
| `LAMBDA_FLOOD` | 1.20 | Bayes factor per flood event (personal experience). |
| `LAMBDA_SOCIAL` | 1.50 | Bayes factor per connected neighbor retrofit (proximity). |
| `LAMBDA_SIMILARITY` | 3.00 | Bayes factor at full similarity ($S=1$). Scaled: $\\lambda_{\\text{similarity}}^{S(i,j)}$. |
""")

    st.subheader("9.2 &nbsp; PMT Threshold Parameters")
    st.markdown("""
| Parameter | Default | Description |
|:---|:---:|:---|
| `PMT_THRESHOLD_MEAN` | 0.50 | Population mean threshold. |
| `PMT_THRESHOLD_STD` | 0.10 | Standard deviation. |
| `PMT_THRESHOLD_LOW` | 0.50 | Hard lower bound (truncation). |
| `PMT_THRESHOLD_HIGH` | 0.50 | Hard upper bound (truncation). |
""")

    st.subheader("9.3 &nbsp; Network, Neighborhood & Spatial Parameters")
    st.markdown("""
| Parameter | Default | Description |
|:---|:---:|:---|
| `DISTANCE_THRESHOLD` | 0.09 | Max distance for binary connections and DBSCAN $\\varepsilon$. |
| `DBSCAN_MIN_SAMPLES` | 4 | Min agents (incl. self) to form a DBSCAN core point. |
| `N_ATTRIBUTES` | 2 | Attributes per agent. Determines Jaccard resolution. |
| `N_CLASSES` | 3 | Categories per attribute. With 2 attrs × 3 classes: $S \\in \\{0, 0.5, 1\\}$. |
| `SPATIAL_MODE` | 2 | 0=CSV, 1=random, 2=grid with connectors. |
| `SLOPE` | 1.0 | Elevation gradient. $z = \\text{slope} \\times x$. |
""")

    # --- 10. Modes of Operation ---
    st.markdown('<div class="doc-section-header">10 &nbsp; Modes of Operation</div>',
                unsafe_allow_html=True)

    col_r, col_c = st.columns(2)
    with col_r:
        st.subheader("Research Mode")
        st.markdown("""
Designed for theoretical exploration. Uses synthetic layouts and
GEV-generated flood levels. Noise toggles allow turning on or off:
- **PMT threshold heterogeneity** (truncated Normal vs. homogeneous)
- **Elevation noise** (random perturbation of the elevation gradient)

Useful for sensitivity analysis and understanding how each model
component contributes to aggregate adoption patterns.
""")
    with col_c:
        st.subheader("Case Study Mode")
        st.markdown("""
Designed for application to a specific site. Upload:
- **Location CSV** (required): columns `x`, `y`, `z` with values in [0, 1].
- **Flood time series CSV** (optional): column `flood_level`, one value
  per time step. If not provided, GEV generation is used.
""")

    # --- 11. CSV Format ---
    st.markdown('<div class="doc-section-header">11 &nbsp; Case Study Mode — CSV Format</div>',
                unsafe_allow_html=True)

    col_loc, col_fld = st.columns(2)
    with col_loc:
        st.markdown("**Location file** (required):")
        st.code("x,y,z\n0.12,0.45,0.03\n0.15,0.47,0.04\n...", language="csv")
    with col_fld:
        st.markdown("**Flood time series** (optional):")
        st.code("flood_level\n0.02\n0.00\n0.08\n...", language="csv")

    # --- 12. References ---
    st.markdown('<div class="doc-section-header">12 &nbsp; References</div>',
                unsafe_allow_html=True)
    st.markdown("""
- Amini, E., Madajewicz, M., Orton, P., Srikrishnan, V., & Yanez Mena, P. (2025). Social Learning and Flood Adaptation via an Agent-Based Model. Poster presented at AGU Fall Meeting 2025, Washington, D.C.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*, 226–231.
- Hosking, J. R. M., & Wallis, J. R. (1997). *Regional Frequency Analysis*. Cambridge University Press.
- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37–50.
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press.
- Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430), 773–795.
- Kazil, J., Masad, D., & Crooks, A. (2020). Utilizing Python for Agent-Based Modeling: The Mesa Framework. Springer.
- McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a feather: Homophily in social networks. *Annual Review of Sociology*, 27, 415–444.
- Rogers, R. W. (1975). A protection motivation theory of fear appeals and attitude change. *Journal of Psychology*, 91(1), 93–114.
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124–1131.
""")


# ============================================================================
# TAB: ADVANCED SETTINGS
# ============================================================================

with tab_settings:
    st.title("🔧 Advanced Settings")
    st.markdown("Adjust spatial, network, attribute, and flood parameters below.")

    random_seed = st.number_input("Random Seed", 0, 9999, 42)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🗺️ Spatial Layout")
        if mode == "Research Mode":
            spatial_mode = st.selectbox(
                "Layout Mode",
                options=[2, 1],
                format_func=lambda x: {2: "Grid with Connectors",
                                        1: "Random Positions"}[x],
            )
            grid_rows = st.number_input("Grid Rows", 1, 10, 3)
            grid_cols = st.number_input("Grid Cols", 1, 10, 4)
            n_connectors = st.number_input(
                "Connectors Between Neighborhoods", 0, 10, 2)
            slope = st.slider("Elevation Slope", 0.0, 2.0, 0.20, 0.01)
            elev_noise_on = st.checkbox("Elevation Noise", value=True,
                help="Add random perturbation to elevation gradient.")
            noise_factor = 0.05 if elev_noise_on else 0.0
        else:
            spatial_mode = 0
            grid_rows, grid_cols, n_connectors = 3, 4, 2
            slope, noise_factor = 1.0, 0.0
            st.info("Spatial layout is determined by uploaded CSV "
                    "in Case Study Mode.")

        st.subheader("🧬 Agent Attributes")
        enable_het = st.checkbox(
            "Enable Attribute Heterogeneity", value=True)
        n_attributes = st.number_input("Attributes per Agent", 1, 10, 2)
        n_classes = st.number_input("Classes per Attribute", 1, 10, 3)

    with col_right:
        st.subheader("🔗 Network")
        dist_threshold = st.slider(
            "Distance Threshold", 0.01, 0.30, 0.09, 0.01,
            help="Max Euclidean distance for a binary connection.")
        dbscan_min = st.number_input(
            "DBSCAN Min Samples", 2, 10, 4,
            help="Minimum agents (incl. self) to form a neighborhood core.")

        st.subheader("🌊 Flood Generation (GEV)")
        if mode == "Research Mode":
            rp_str = st.text_input("Return Periods", "10, 20, 50, 100")
            fl_str = st.text_input("Flood Levels", "0.05, 0.10, 0.15, 0.30")
            return_periods = [int(x.strip()) for x in rp_str.split(",")]
            flood_levels = [float(x.strip()) for x in fl_str.split(",")]
        else:
            return_periods = [10, 20, 50, 100]
            flood_levels = [0.05, 0.10, 0.15, 0.30]
            st.info("Flood data from uploaded CSV or GEV defaults "
                    "in Case Study Mode.")

        st.subheader("📋 Observed Data (optional)")
        obs_bins_str = st.text_input("Observed Bins", "0, 1, 2-3, 4+")
        obs_rates_str = st.text_input("Observed Rates (%)", "13, 18, 27, 57")
        observed_bins = [x.strip() for x in obs_bins_str.split(",")]
        observed_rates = [int(x.strip()) for x in obs_rates_str.split(",")]


# ============================================================================
# HELPER: apply parameters
# ============================================================================

def apply_params():
    M.TIME_STEPS = time_steps
    M.RANDOM_SEED = random_seed
    M.N_AGENTS = n_agents
    M.SPATIAL_MODE = spatial_mode
    M.GRID_ROWS = grid_rows
    M.GRID_COLS = grid_cols
    M.N_CONNECTORS = n_connectors
    M.SLOPE = slope
    M.NOISE_FACTOR = noise_factor
    M.ENABLE_HETEROGENEITY = enable_het
    M.N_ATTRIBUTES = n_attributes
    M.N_CLASSES = n_classes
    M.DISTANCE_THRESHOLD = dist_threshold
    M.DBSCAN_MIN_SAMPLES = dbscan_min
    M.USER_EDGES_CSV = None
    M.CSV_PATH = None
    M.INITIAL_BELIEF = initial_belief
    M.LAMBDA_FLOOD = lambda_flood
    M.LAMBDA_SOCIAL = lambda_social
    M.LAMBDA_SIMILARITY = lambda_similarity
    M.PMT_THRESHOLD_MEAN = pmt_mean
    M.PMT_THRESHOLD_STD = pmt_std
    M.PMT_THRESHOLD_LOW = pmt_low
    M.PMT_THRESHOLD_HIGH = pmt_high
    M.RETURN_PERIODS = return_periods
    M.FLOOD_LEVELS = flood_levels
    M.OBSERVED_BINS = observed_bins
    M.OBSERVED_RATES = observed_rates
    M.FLOOD_BIN_RANGES = M.parse_flood_bins(observed_bins)


# ============================================================================
# HELPER: color palettes for figures
# ============================================================================

# Consistent color scheme across all plots
CLR_PRIMARY = "#1B4F72"
CLR_SECONDARY = "#2E86C1"
CLR_ACCENT = "#E67E22"
CLR_SUCCESS = "#27AE60"
CLR_DANGER = "#C0392B"
CLR_FLOOD_BAR = "#5DADE2"
CLR_MODEL_BAR = "#2E86C1"
CLR_OBS_BAR = "#E67E22"
CLR_RETRO = "#27AE60"
CLR_NOT_RETRO = "#D5D8DC"
CLR_BELIEF = "#1B4F72"
CLR_THRESHOLD = "#C0392B"
CLR_ELEV = ["#C0392B", "#E67E22", "#27AE60"]


# ============================================================================
# HELPER: generate figures
# ============================================================================

def make_figures(model):
    agents = list(model.agents)
    df_model = model.get_model_dataframe()
    df_agent = model.get_agent_dataframe()
    steps = df_model.index.values
    figs = {}

    # --- Figure 1: Adoption Curve + Flood History (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(df_model.index, df_model["pct_retrofitted"],
             color=CLR_PRIMARY, linewidth=1.8)
    ax1.fill_between(df_model.index, df_model["pct_retrofitted"],
                     alpha=0.12, color=CLR_SECONDARY)
    ax1.set(xlabel="Time Step", ylabel="Retrofitted (%)",
            title="(a) Retrofit Adoption Over Time",
            xlim=(1, model.time_steps), ylim=(0, 100))

    ax2.bar(range(1, len(model.flood_history)+1), model.flood_history,
            color=CLR_FLOOD_BAR, alpha=0.75, edgecolor="none", linewidth=0)
    m = np.mean(model.flood_history)
    ax2.axhline(y=m, color=CLR_DANGER, linestyle="--", linewidth=1.2,
                label=f"Mean = {m:.3f}")
    ax2.set(xlabel="Time Step", ylabel="Flood Level",
            title="(b) Flood Levels (GEV, Coles 2001)")
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["adoption_flood"] = fig

    # --- Figure 2: Elevation + Comparison (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    terciles = np.percentile([a.z for a in agents], [33.33, 66.67])
    groups = {"Low": [], "Medium": [], "High": []}
    for a in agents:
        if a.z <= terciles[0]: groups["Low"].append(a)
        elif a.z <= terciles[1]: groups["Medium"].append(a)
        else: groups["High"].append(a)
    rates_e, labels_e = [], []
    for label, group in groups.items():
        if group:
            rates_e.append(
                100*sum(1 for a in group if a.is_retrofitted)/len(group))
            labels_e.append(f"{label}\n($n$={len(group)})")
    bars = ax1.bar(labels_e, rates_e,
                   color=CLR_ELEV, edgecolor="black", linewidth=0.6)
    for bar, rate in zip(bars, rates_e):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=FS-1)
    ax1.set(xlabel="Elevation Tercile", ylabel="Retrofit Rate (%)",
            title="(a) Adoption by Elevation",
            ylim=(0, max(rates_e)*1.25 if rates_e else 100))

    bins_dict = {b: {"total": 0, "retrofitted": 0} for b in observed_bins}
    for a in agents:
        cat = M.categorize_flood_count(a.flood_count)
        bins_dict[cat]["total"] += 1
        if a.is_retrofitted: bins_dict[cat]["retrofitted"] += 1
    model_rates = [100*bins_dict[b]["retrofitted"]/bins_dict[b]["total"]
                   if bins_dict[b]["total"] > 0 else 0 for b in observed_bins]
    x_pos = np.arange(len(observed_bins)); width = 0.35
    ax2.bar(x_pos-width/2, model_rates, width, label="Model",
            color=CLR_MODEL_BAR, edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos+width/2, observed_rates, width, label="Observed",
            color=CLR_OBS_BAR, edgecolor="black", linewidth=0.5)
    for i, (mr, orr) in enumerate(zip(model_rates, observed_rates)):
        if mr > 0:
            ax2.text(i-width/2, mr+1, f"{mr:.0f}%", ha="center",
                     fontsize=FS-2, color=CLR_MODEL_BAR, fontweight="bold")
        if orr > 0:
            ax2.text(i+width/2, orr+1, f"{orr}%", ha="center",
                     fontsize=FS-2, color=CLR_OBS_BAR, fontweight="bold")
    ax2.set(xlabel="Flood Count Category", ylabel="Retrofit Rate (%)",
            title="(b) Model vs Observed")
    ax2.set_xticks(x_pos); ax2.set_xticklabels(observed_bins)
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["elev_comparison"] = fig

    # --- Figure 3: Belief Evolution (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    p10, p90 = [], []
    for s in steps:
        sb = df_agent.xs(s, level="Step")["belief"].values
        p10.append(np.percentile(sb, 10))
        p90.append(np.percentile(sb, 90))
    ax1.fill_between(steps, p10, p90, alpha=0.15, color=CLR_SECONDARY,
                     label="10th\u201390th pctl.")
    ax1.plot(steps, df_model["mean_belief"], color=CLR_PRIMARY,
             linewidth=1.8, label="Mean $P(H_1)$")
    final_data = df_agent.xs(steps[-1], level="Step")
    thr_vals = final_data["pmt_threshold"].values
    thr_mean = thr_vals.mean()
    if pmt_std > 0:
        tp10, tp90 = np.percentile(thr_vals, [10, 90])
        ax1.axhspan(tp10, tp90, color=CLR_THRESHOLD, alpha=0.06,
                    label=f"Threshold [{tp10:.2f}, {tp90:.2f}]")
    ax1.axhline(y=thr_mean, color=CLR_THRESHOLD, linestyle="--",
                linewidth=1.2, label=f"Threshold = {thr_mean:.2f}")
    ax1.axhline(y=initial_belief, color="gray", linestyle=":",
                linewidth=1, label=f"Prior = {initial_belief}")
    ax1.set(xlabel="Time Step", ylabel="$P(H_1)$",
            title="(a) Belief Evolution")
    ax1.legend(fontsize=FS-3, framealpha=0.9)

    final_beliefs = final_data["belief"].values
    ax2.hist(final_beliefs, bins=30, color=CLR_SECONDARY,
             edgecolor="white", linewidth=0.4, alpha=0.85)
    ax2.axvline(x=thr_mean, color=CLR_THRESHOLD, linestyle="--",
                linewidth=1.5, label=f"Threshold = {thr_mean:.2f}")
    ax2.set(xlabel="$P(H_1)$", ylabel="Number of Agents",
            title=f"(b) Final Belief Distribution (Step {steps[-1]})")
    ax2.legend(framealpha=0.9)

    fig.tight_layout(w_pad=3)
    figs["belief"] = fig

    # --- Figure 4: Network map ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    segs = []
    for u, v in model.G.edges():
        a_u, a_v = model.agents_by_node[u], model.agents_by_node[v]
        segs.append([(a_u.x, a_u.y), (a_v.x, a_v.y)])
    if segs:
        ax1.add_collection(LineCollection(
            segs, linewidths=0.4, colors="gray", alpha=0.2, zorder=1))
    adopted = [a for a in agents if a.is_retrofitted]
    not_adopted = [a for a in agents if not a.is_retrofitted]
    # Larger node size for readability
    node_size = max(100, min(450, 15000 // max(len(agents), 1)))
    font_size = max(5, min(7, 1400 // max(len(agents), 1)))
    if not_adopted:
        ax1.scatter([a.x for a in not_adopted], [a.y for a in not_adopted],
                    c=CLR_NOT_RETRO, s=node_size, edgecolor="black",
                    linewidth=0.6, zorder=2)
        for a in not_adopted:
            ax1.text(a.x, a.y, str(a.flood_count), ha="center",
                     va="center", fontsize=font_size,
                     fontweight="bold", zorder=3)
    if adopted:
        sc = ax1.scatter([a.x for a in adopted], [a.y for a in adopted],
            c=[a.retrofit_step for a in adopted], cmap="YlGn",
            s=node_size, edgecolor="black", linewidth=0.6, zorder=2,
            vmin=1, vmax=model.current_step)
        plt.colorbar(sc, ax=ax1, shrink=0.5, pad=0.02).set_label(
            "Retrofit Step", fontsize=FS-1)
        for a in adopted:
            ax1.text(a.x, a.y, str(a.flood_count), ha="center",
                     va="center", fontsize=font_size,
                     fontweight="bold", zorder=3)
    ax1.set(xlabel="$x$", ylabel="$y$",
            title="Social Network — Binary Connections within Threshold",
            xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax1.legend(handles=[
        Patch(facecolor=CLR_NOT_RETRO, edgecolor="black",
              label="Not Retrofitted"),
        Patch(facecolor=CLR_RETRO, edgecolor="black",
              label="Retrofitted")],
        loc="upper right", fontsize=FS-2, framealpha=0.9)
    fig.tight_layout()
    figs["network_large"] = fig

    # --- Figure 5: Similarity distribution (standalone) ---
    fig, ax2 = plt.subplots(figsize=(8, 4))
    sims = [d["similarity"] for _, _, d in model.G.edges(data=True)]
    ax2.hist(sims, bins=30, color=CLR_SECONDARY, edgecolor="white",
             linewidth=0.4, alpha=0.85)
    ax2.axvline(np.mean(sims), color=CLR_DANGER, linestyle="--",
                linewidth=1.2, label=f"Mean = {np.mean(sims):.3f}")
    ax2.set(xlabel="Jaccard Similarity $S(i,j)$", ylabel="Frequency",
            title="Attribute Similarity Distribution (Jaccard, 1912)")
    ax2.legend(framealpha=0.9)
    fig.tight_layout()
    figs["similarity"] = fig

    # --- Figure 6: Spatial map (compact) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    segs2 = []
    for u, v in model.G.edges():
        a_u, a_v = model.agents_by_node[u], model.agents_by_node[v]
        segs2.append([(a_u.x, a_u.y), (a_v.x, a_v.y)])
    if segs2:
        ax.add_collection(LineCollection(
            segs2, linewidths=0.3, colors="gray", alpha=0.15, zorder=1))
    small_node = max(40, min(180, 6000 // max(len(agents), 1)))
    sc = ax.scatter([a.x for a in agents], [a.y for a in agents],
                    c=[a.z for a in agents], cmap="terrain", s=small_node,
                    alpha=0.8, edgecolor="black", linewidth=0.3, zorder=2)
    for a in agents:
        if a.is_retrofitted:
            ax.scatter(a.x, a.y, c=CLR_RETRO, s=small_node,
                       edgecolor="black", linewidth=1.0, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02).set_label(
        "Elevation", fontsize=FS-2)
    ax.set(xlabel="$x$", ylabel="$y$",
           title="Elevation & Retrofit Status",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
    ax.legend(handles=[
        Patch(facecolor=CLR_RETRO, edgecolor="black", label="Retrofitted"),
        Patch(facecolor="lightgray", edgecolor="black",
              label="Not Retrofitted")],
        loc="upper right", fontsize=FS-3, framealpha=0.9)
    fig.tight_layout()
    figs["spatial"] = fig

    return figs


# ============================================================================
# TAB: RUN SIMULATION
# ============================================================================

with tab_sim:
    st.title("▶️ Run Simulation")

    # Parameter summary
    st.subheader("Current Configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", n_agents)
    c1.metric("Time Steps", time_steps)
    c2.metric("Flood Experience — $\\lambda_{\\mathrm{flood}}$",
              f"{lambda_flood:.2f}")
    c2.metric("Proximity Learning — $\\lambda_{\\mathrm{social}}$",
              f"{lambda_social:.2f}")
    c3.metric("Similarity Learning — $\\lambda_{\\mathrm{sim.}}$",
              f"{lambda_similarity:.2f}")
    c3.metric("PMT Threshold", f"{pmt_mean:.2f}")
    c4.metric("Initial Belief — $P(H_1)$", f"{initial_belief:.2f}")
    c4.metric("Mode", mode.replace(" Mode", ""))

    st.divider()

    if mode == "Case Study Mode" and uploaded_csv is None:
        st.warning(
            "Upload a location CSV in the sidebar to use Case Study Mode.")

    # Run button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        apply_params()
        log_area = st.empty()
        progress_bar = st.progress(0)
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            log_area.code("\n".join(log_lines), language="text")

        log("Initializing model...")

        # Handle Case Study CSV
        if mode == "Case Study Mode" and uploaded_csv is not None:
            uploaded_csv.seek(0)
            csv_df = pd.read_csv(uploaded_csv)
            csv_path = "/tmp/_abm_upload.csv"
            csv_df.to_csv(csv_path, index=False)
            M.N_AGENTS = len(csv_df)
            M.SPATIAL_MODE = 0
            M.CSV_PATH = csv_path
            log(f"  Loaded {len(csv_df)} agents from uploaded CSV")

        model = M.FloodAdaptationModel()
        log(f"  Agents: {model.n_agents}")
        log(f"  Edges: {model.G.number_of_edges()}")
        log(f"  Neighborhoods: {model.n_neighborhoods} "
            f"(isolated: {int(np.sum(model.neighborhood_labels == -1))})")
        log(f"  Mean degree: "
            f"{np.mean([d for _, d in model.G.degree()]):.1f}")
        progress_bar.progress(5)

        # Inject flood time series if provided
        if mode == "Case Study Mode" and uploaded_flood is not None:
            uploaded_flood.seek(0)
            flood_ts = pd.read_csv(uploaded_flood)["flood_level"].values
            class FixedFlood:
                def __init__(self, levels):
                    self._levels = levels; self._idx = 0
                def sample(self):
                    if self._idx < len(self._levels):
                        v = self._levels[self._idx]; self._idx += 1
                        return float(np.clip(v, 0, 1))
                    return 0.0
            model.flood_generator = FixedFlood(flood_ts)
            log(f"  Using uploaded flood time series "
                f"({len(flood_ts)} steps)")

        log("\nRunning simulation...")
        t0 = time.time()

        for i in range(model.time_steps):
            model.step()
            pct = int(5 + 85 * (i + 1) / model.time_steps)
            progress_bar.progress(pct)
            if (i + 1) % max(1, model.time_steps // 10) == 0:
                n_ret = sum(1 for a in model.agents if a.is_retrofitted)
                log(f"  Step {i+1:4d}: retrofitted = "
                    f"{n_ret}/{model.n_agents} "
                    f"({100*n_ret/model.n_agents:.1f}%)")

        elapsed = time.time() - t0
        agents = list(model.agents)
        n_ret = sum(1 for a in agents if a.is_retrofitted)

        log(f"\nSimulation complete in {elapsed:.1f}s")
        log(f"  Retrofitted: {n_ret}/{model.n_agents} "
            f"({100*n_ret/model.n_agents:.1f}%)")
        log(f"  Mean belief: "
            f"{np.mean([a.belief for a in agents]):.4f}")
        log(f"  Mean flood count: "
            f"{np.mean([a.flood_count for a in agents]):.1f}")

        progress_bar.progress(90, text="Generating figures...")
        log("\nGenerating figures...")

        st.session_state["model"] = model
        st.session_state["figs"] = make_figures(model)

        progress_bar.progress(100, text="Done.")
        log("Done. Go to the Results tab.")

        st.success(
            f"Retrofitted: {n_ret}/{model.n_agents} "
            f"({100*n_ret/model.n_agents:.1f}%) in {elapsed:.1f}s")


# ============================================================================
# TAB: RESULTS
# ============================================================================

with tab_results:
    st.title("📊 Results")

    if "figs" not in st.session_state:
        st.info("Run a simulation first to see results here.")
    else:
        figs = st.session_state["figs"]
        model = st.session_state["model"]

        st.header("Adoption and Flood History")
        st.pyplot(figs["adoption_flood"])

        st.header("Elevation and Observed Comparison")
        st.pyplot(figs["elev_comparison"])

        st.header("Belief Evolution")
        st.pyplot(figs["belief"])

        st.header("Social Network")
        st.pyplot(figs["network_large"])

        # Similarity + Spatial side by side (similarity bigger)
        col_sim, col_sp = st.columns([3, 2])
        with col_sim:
            st.header("Similarity Distribution")
            st.pyplot(figs["similarity"])
        with col_sp:
            st.header("Spatial Map")
            st.pyplot(figs["spatial"])

        # Export
        st.divider()
        st.header("📥 Export Data")
        col1, col2 = st.columns(2)
        with col1:
            agent_csv = model.get_agent_dataframe().to_csv()
            st.download_button("Download Agent Data (CSV)",
                               agent_csv, "agent_data.csv", "text/csv",
                               use_container_width=True)
        with col2:
            model_csv = model.get_model_dataframe().to_csv()
            st.download_button("Download Model Data (CSV)",
                               model_csv, "model_data.csv", "text/csv",
                               use_container_width=True)


# ============================================================================
# FOOTER — Author + License
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="app-footer">
    <span class="author-name">Erfan Amini</span><br>
    <span class="affiliation">Center for Climate Systems Research (CCSR),
    Columbia University</span><br><br>
    <div class="license-box">
        <strong>License — Creative Commons Attribution 4.0 International
        (CC BY 4.0)</strong><br><br>
        Copyright © 2025 Erfan Amini<br><br>
        This work is licensed under the Creative Commons Attribution 4.0
        International License. You are free to share, copy, redistribute,
        adapt, remix, transform, and build upon this material for any
        purpose, including commercial, under the following terms:<br><br>
        <strong>Attribution Required.</strong> You must give appropriate
        credit by citing the following reference in any publication,
        presentation, software, or derivative work that uses this
        model:<br><br>
        <em>Amini, E., Madajewicz, M., Orton, P., Srikrishnan, V.,
        &amp; Yanez Mena, P. (2025). Social Learning and Flood Adaptation
        via an Agent-Based Model. Poster presented at AGU Fall Meeting
        2025, Washington, D.C.</em><br><br>
        You must also indicate if changes were made. You may do so in any
        reasonable manner, but not in any way that suggests the licensor
        endorses you or your use.<br><br>
        Full license text:
        <a href="https://creativecommons.org/licenses/by/4.0/"
           target="_blank" style="color: #2E86C1;">
           https://creativecommons.org/licenses/by/4.0/</a><br><br>
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
    </div>
</div>
""", unsafe_allow_html=True)
