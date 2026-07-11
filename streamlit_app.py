"""
Flood Adaptation ABM v13 - Bayesian Belief Updating (Three Channels)
Single-file Streamlit application. All engine modules (flood/GEV, spatial,
attributes, network, neighborhoods, model) are inlined here; there are no
external FFF_*.py or AAA_model_*.py dependencies.

Binary hypothesis model:
  H1 = "I should retrofit my house"
  H0 = "I should not retrofit my house"
Belief P(H1) updates via Bayes' theorem in odds form
(Jaynes, 2003, Ch. 4; Kass & Raftery, 1995):
  posterior_odds = prior_odds x Bayes_factor

Three evidence channels, each a BASE Bayes factor times a CONDITIONAL
multiplier that equals 1 (no effect) whenever its trigger is absent:

  1. Personal flood experience
       base:       lambda_flood        (one factor per flood; Good, 1950)
       multiplier: lambda_severity     active for agents who expect rising
                   flood damage (perceived-severity appraisal; Rogers, 1975;
                   Floyd, Prentice-Dunn & Rogers, 2000). = 1 otherwise.
       Fires only in a flood year (flood_level > z). Availability heuristic:
       safe years produce no update (Tversky & Kahneman, 1974).

  2. Proximity-based social learning
       base:       lambda_social       per newly-retrofitted connected
                   neighbor (Granovetter, 1978).
       multiplier: lambda_similarity   active when that neighbor is similar,
                   i.e. Gower similarity S(i,j) >= SIM_THRESHOLD
                   (Gower, 1971; McPherson et al., 2001). = 1 otherwise.
       Fires only when a connected neighbor is retrofitted; with no
       retrofitted neighbor the whole channel is inert (factor 1).

  3. Trusted flood information
       base:       lambda_info         active for agents who have a trusted
                   flood-information source.
       multiplier: lambda_forecast     active when the agent also prepares
                   on the basis of flood forecasts. = 1 otherwise.
       Fires ONCE, at initialization (a static informational prior), because
       trusted-info and forecast-preparation are stable household traits.

Survey-anchored defaults (NYC Flood Vulnerability Survey, cleaned):
  lambda_flood      1.52  (owner per-flood odds ratio)
  lambda_severity   2.40  (expecting- vs not-expecting rising damage)
  lambda_forecast   3.20  (forecast-preparation odds ratio)
  P(expects rising damage) 0.69 ; P(trusted info) 0.48 ; P(forecast prep) 0.65
  Cumulative "at most k" retrofit targets: 18.0 / 22.3 / 27.4 %

References
----------
Jaynes (2003) Probability Theory, Ch. 4. Kass & Raftery (1995) Bayes Factors.
Good (1950) Probability and the Weighing of Evidence. Rogers (1975) PMT.
Floyd, Prentice-Dunn & Rogers (2000) Meta-analysis of PMT. Granovetter (1978).
Gower (1971). McPherson et al. (2001) Homophily. Ester et al. (1996) DBSCAN.
Tversky & Kahneman (1974). Coles (2001) Extreme values.
"""

import os
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection

import mesa
from mesa.space import NetworkGrid
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.stats import genextreme
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN


# ============================================================================
# DEFAULT PARAMETERS  (all overridable from the UI)
# ============================================================================

DEFAULTS = dict(
    # Simulation
    TIME_STEPS=100, RANDOM_SEED=42,
    # Spatial / population
    N_AGENTS=200, GRID_ROWS=3, GRID_COLS=4, N_CONNECTORS=2,
    SLOPE=0.2, NOISE_FACTOR=0.05,
    # Attributes
    ENABLE_HETEROGENEITY=True, N_ATTRIBUTES=2, N_CLASSES=3,
    # Network / neighborhoods
    DISTANCE_THRESHOLD=0.09, DBSCAN_MIN_SAMPLES=4,
    # Belief.  Opening default is a LOW baseline (0.08) consistent with the
    # baseline against which the survey odds ratios were estimated; the raw
    # survey point-estimates (e.g. b0 ~ 0.23) assume the other channels held
    # at baseline, so they cannot all be defaults at once without saturating.
    INITIAL_BELIEF=0.08,
    # Channel 1 - experience.  Survey anchors: LAMBDA_FLOOD 1.52 (owner per-flood
    # odds ratio), LAMBDA_SEVERITY 2.40 (expecting vs not-expecting rising
    # damage).  Opened slightly de-escalated so the model is non-saturated out
    # of the box; tune upward toward the anchors and watch the cumulative bars.
    LAMBDA_FLOOD=1.52, LAMBDA_SEVERITY=1.60,
    P_EXPECT_RISING_DAMAGE=0.69,
    # Channel 2 - proximity + similarity.  Survey/prior anchor for social 4.51;
    # opened low (1.30) because the dense network makes the social cascade the
    # main saturation driver.  Similarity is a binary amplifier (S >= threshold).
    LAMBDA_SOCIAL=1.30, LAMBDA_SIMILARITY=1.00, SIM_THRESHOLD=0.50,
    # Channel 3 - information.  Trusted-info alone is weak in the survey
    # (OR ~1.39, n.s.); forecast preparation is the stronger amplifier
    # (OR ~3.2).  Opened with LOW information factor and multiplier so the
    # one-time t=0 informational prior does not by itself push belief over the
    # threshold; tune upward toward the survey anchors as needed.
    LAMBDA_INFO=1.05, LAMBDA_FORECAST=1.15,
    P_TRUSTED_INFO=0.48, P_FORECAST_PREP=0.65,
    # PMT threshold
    PMT_THRESHOLD_MEAN=0.85, PMT_THRESHOLD_STD=0.10,
    PMT_THRESHOLD_LOW=0.75, PMT_THRESHOLD_HIGH=0.95,
    ENABLE_THRESHOLD_HET=True,
    # Flood (GEV)
    RETURN_PERIODS=[10, 20, 50, 100],
    FLOOD_LEVELS=[0.05, 0.10, 0.15, 0.30],
    # Observed cumulative "at most k" retrofit rates (%)
    OBSERVED_CUM_LABELS=["0", "\u22644", "5+"],
    OBSERVED_CUM_RATES=[18.0, 22.3, 27.4],
    OBSERVED_CUM_MAX=[0, 4, np.inf],   # upper flood-count bound for each cumulative bin
)


# ============================================================================
# BAYESIAN UPDATE  (odds form; Jaynes, 2003; Kass & Raftery, 1995)
# ============================================================================

def bayesian_update(belief, bayes_factor):
    """posterior_odds = prior_odds x bayes_factor, returned as probability."""
    odds = belief / (1.0 - belief)
    odds *= bayes_factor
    return odds / (1.0 + odds)


# ============================================================================
# ATTRIBUTES + SIMILARITY  (Gower, 1971)
# ============================================================================

def similarity_coefficient(attr_a, attr_b):
    """Fraction of attributes on which two agents agree (Gower, 1971)."""
    a = np.asarray(attr_a); b = np.asarray(attr_b)
    return float(np.sum(a == b) / len(a))


def generate_attributes(n_agents, n_attributes, n_classes, enable_het, rng):
    if enable_het:
        return rng.integers(0, n_classes, size=(n_agents, n_attributes))
    return np.zeros((n_agents, n_attributes), dtype=int)


# ============================================================================
# FLOOD GENERATOR  (GEV; Coles, 2001)
# ============================================================================

def _return_period_to_probability(T):
    return 1.0 - 1.0 / np.asarray(T)


def _gev_quantile(p, loc, scale, shape):
    p = np.asarray(p); y = -np.log(p)
    if np.abs(shape) < 1e-8:
        return loc - scale * np.log(y)
    return loc + (scale / shape) * (y ** (-shape) - 1)


def _fit_gev(return_periods, flood_levels):
    rp = np.asarray(return_periods); fl = np.asarray(flood_levels)
    probs = _return_period_to_probability(rp)
    x0 = [np.mean(fl), np.std(fl) * np.sqrt(6) / np.pi, 0.1]

    def obj(params):
        loc, scale, shape = params
        if scale <= 0:
            return 1e10
        pred = _gev_quantile(probs, loc, scale, shape)
        sse = np.sum((pred - fl) ** 2)
        p_low = _gev_quantile(0.01, loc, scale, shape)
        if p_low < 0:
            sse += 1000 * (p_low ** 2)
        return sse

    res = minimize(obj, x0=x0, method="Nelder-Mead", options={"maxiter": 10000})
    loc, scale, shape = res.x
    return loc, max(scale, 1e-6), shape


class FloodGenerator:
    """Stateless GEV flood sampler (Coles, 2001)."""

    def __init__(self, return_periods, flood_levels, rng):
        self.rng = rng
        self.loc, self.scale, self.shape = _fit_gev(return_periods, flood_levels)

    def sample(self):
        v = genextreme.rvs(c=-self.shape, loc=self.loc, scale=self.scale,
                           random_state=self.rng)
        return float(np.clip(v, 0.0, 1.0))


# ============================================================================
# SPATIAL LAYOUT  (grid neighborhoods with optional connectors)
# ============================================================================

_DIAGONAL_SAFETY = 0.999
_MIN_MARGIN = 0.02
_MAX_EXTENT = 1.0 - 2 * _MIN_MARGIN
_COORD_MIN, _COORD_MAX = 0.01, 0.99


def _connected_grid(n_agents, distance_threshold, grid_rows, grid_cols, n_connectors):
    n_neighborhoods = grid_rows * grid_cols
    n_h = grid_rows * (grid_cols - 1)
    n_v = (grid_rows - 1) * grid_cols
    total_connectors = (n_h + n_v) * n_connectors
    agents_in_grids = n_agents - total_connectors
    base = agents_in_grids // n_neighborhoods
    remainder = agents_in_grids % n_neighborhoods

    def dims(n):
        nr = int(np.ceil(np.sqrt(n)))
        if nr % 2 == 0:
            nr += 1
        nc = int(np.ceil(n / nr))
        return nr, nc

    max_per = base + (1 if remainder > 0 else 0)
    nh_rows, nh_cols = dims(max_per)
    spacing = _DIAGONAL_SAFETY * distance_threshold / np.sqrt(2)
    nh_w = (nh_cols - 1) * spacing
    nh_h = (nh_rows - 1) * spacing
    gap = (n_connectors + 1) * spacing
    total_w = grid_cols * nh_w + (grid_cols - 1) * gap
    total_h = grid_rows * nh_h + (grid_rows - 1) * gap

    if total_w > _MAX_EXTENT or total_h > _MAX_EXTENT:
        scale = min(_MAX_EXTENT / total_w, _MAX_EXTENT / total_h)
        spacing *= scale
        nh_w = (nh_cols - 1) * spacing
        nh_h = (nh_rows - 1) * spacing
        gap = (n_connectors + 1) * spacing
        total_w = grid_cols * nh_w + (grid_cols - 1) * gap
        total_h = grid_rows * nh_h + (grid_rows - 1) * gap

    margin_x = _MIN_MARGIN
    margin_y = max(_MIN_MARGIN, (1.0 - total_h) / 2)
    coords = []
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            nh_idx = gr * grid_cols + gc
            n_here = base + (1 if nh_idx < remainder else 0)
            ox = margin_x + gc * (nh_w + gap)
            oy = margin_y + gr * (nh_h + gap)
            c = 0
            for row in range(nh_rows):
                for col in range(nh_cols):
                    if c >= n_here:
                        break
                    coords.append([ox + col * spacing, oy + row * spacing])
                    c += 1
                if c >= n_here:
                    break
    for gr in range(grid_rows):
        for gc in range(grid_cols - 1):
            lox = margin_x + gc * (nh_w + gap)
            loy = margin_y + gr * (nh_h + gap)
            rex = lox + nh_w
            my = loy + nh_h / 2
            for c in range(n_connectors):
                coords.append([rex + (c + 1) * spacing, my])
    for gr in range(grid_rows - 1):
        for gc in range(grid_cols):
            box = margin_x + gc * (nh_w + gap)
            boy = margin_y + gr * (nh_h + gap)
            tey = boy + nh_h
            mx = box + nh_w / 2
            for c in range(n_connectors):
                coords.append([mx, tey + (c + 1) * spacing])
    coords = np.clip(np.array(coords), _COORD_MIN, _COORD_MAX)
    return coords[:, 0], coords[:, 1]


def _generate_elevation(x, slope, noise_factor, rng):
    z_base = slope * x
    nm = noise_factor * slope
    noise = rng.uniform(-nm, nm, len(x))
    return np.clip(z_base + noise, 0, slope)


def generate_spatial(n_agents, distance_threshold, grid_rows, grid_cols,
                     n_connectors, slope, noise_factor, rng):
    x, y = _connected_grid(n_agents, distance_threshold, grid_rows, grid_cols,
                           n_connectors)
    z = _generate_elevation(x, slope, noise_factor, rng)
    return np.column_stack([x, y]), z


# ============================================================================
# NETWORK  (binary connections; edge stores Gower similarity)
# ============================================================================

def build_network(positions, attributes, distance_threshold):
    G = nx.Graph()
    n = len(positions)
    G.add_nodes_from(range(n))
    dm = cdist(positions, positions)
    rows, cols = np.where((dm <= distance_threshold) & (dm > 0))
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    S_all = np.mean(attributes[rows] == attributes[cols], axis=1)
    for idx in range(len(rows)):
        G.add_edge(int(rows[idx]), int(cols[idx]),
                   distance=float(dm[rows[idx], cols[idx]]),
                   similarity=float(S_all[idx]))
    return G


def identify_neighborhoods(positions, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(np.asarray(positions))
    return labels, len(set(labels) - {-1})


# ============================================================================
# CUMULATIVE FLOOD-BIN HELPERS
# ============================================================================

def cumulative_model_rates(agents, cum_max):
    """Cumulative 'at most k floods' retrofit rate for each bound in cum_max."""
    counts = np.array([a.flood_count for a in agents])
    retro = np.array([1 if a.is_retrofitted else 0 for a in agents])
    rates = []
    for hi in cum_max:
        m = counts <= hi
        rates.append(100.0 * retro[m].sum() / m.sum() if m.sum() else 0.0)
    return rates


# ============================================================================
# HOUSEHOLD AGENT
# ============================================================================

class HouseholdAgent(mesa.Agent):
    """Bayesian household agent with three evidence channels (see module docstring)."""

    def __init__(self, model, x, y, z, attributes, initial_belief,
                 pmt_threshold, neighborhood_id,
                 expects_rising_damage, has_trusted_info, forecast_prep):
        super().__init__(model)
        self.x = x
        self.y = y
        self.z = z
        self.belief = initial_belief
        self.pmt_threshold = pmt_threshold
        self.neighborhood_id = neighborhood_id
        self.is_retrofitted = False
        self.retrofit_step = None
        self.flood_count = 0
        self.attributes = attributes
        # static household traits (assigned once at t=0, never changed)
        self.expects_rising_damage = expects_rising_damage
        self.has_trusted_info = has_trusted_info
        self.forecast_prep = forecast_prep
        self.observed_retrofitted = set()

    # -- Channel 1: personal flood experience --------------------------------
    def experience_flood(self, flood_level):
        """
        Fires only in a flood year (flood_level > z). Base per-flood factor
        lambda_flood, times the perceived-severity multiplier lambda_severity
        for agents who expect rising flood damage (= 1 otherwise).
        """
        if self.is_retrofitted:
            return False
        if flood_level > self.z:
            self.flood_count += 1
            m = self.model
            mult = m.LAMBDA_SEVERITY if self.expects_rising_damage else 1.0
            self.belief = bayesian_update(self.belief, m.LAMBDA_FLOOD * mult)
            return True
        return False

    # -- Channel 2: proximity + similarity -----------------------------------
    def social_learning(self):
        """
        For each connected neighbor newly observed as retrofitted, apply the
        base proximity factor lambda_social times the similarity multiplier
        lambda_similarity when that neighbor is similar (Gower S >= threshold).
        With no retrofitted neighbor the channel is inert.
        """
        if self.is_retrofitted:
            return
        m = self.model
        for neighbor in m.grid.get_neighbors(self.pos, include_center=False):
            if neighbor.is_retrofitted and neighbor.unique_id not in self.observed_retrofitted:
                self.observed_retrofitted.add(neighbor.unique_id)
                S = m.G.edges[self.pos, neighbor.pos]["similarity"]
                mult = m.LAMBDA_SIMILARITY if S >= m.SIM_THRESHOLD else 1.0
                self.belief = bayesian_update(self.belief, m.LAMBDA_SOCIAL * mult)

    # -- Channel 3: trusted information (applied once at init) ---------------
    def apply_information_prior(self):
        """
        One-time informational update at t=0. Base factor lambda_info for
        agents with a trusted flood-information source, times the forecast
        multiplier lambda_forecast for those who also prepare on forecasts.
        Agents without trusted information receive no update.
        """
        if not self.has_trusted_info:
            return
        m = self.model
        mult = m.LAMBDA_FORECAST if self.forecast_prep else 1.0
        self.belief = bayesian_update(self.belief, m.LAMBDA_INFO * mult)

    def make_decision(self):
        if not self.is_retrofitted and self.belief >= self.pmt_threshold:
            self.is_retrofitted = True
            self.retrofit_step = self.model.current_step

    def step(self):
        self.social_learning()
        self.make_decision()


# ============================================================================
# MODEL
# ============================================================================

class FloodAdaptationModel(mesa.Model):
    """Flood adaptation with three-channel Bayesian updating."""

    def __init__(self, params, seed=None):
        _seed = seed if seed is not None else params["RANDOM_SEED"]
        # Mesa 3.x accepts rng=; older builds only seed=. Try the modern
        # keyword first and fall back so the model works across versions.
        try:
            super().__init__(rng=np.random.default_rng(_seed))
        except TypeError:
            super().__init__(seed=_seed)
        # Guarantee a usable numpy Generator regardless of Mesa version.
        if not hasattr(self, "rng") or not hasattr(self.rng, "integers"):
            self.rng = np.random.default_rng(_seed)
        for k, v in params.items():
            setattr(self, k, v)
        self.n_agents = params["N_AGENTS"]
        self.time_steps = params["TIME_STEPS"]
        self.current_step = 0
        self.current_flood_level = 0.0
        self.flood_history = []
        self.agent_data = []
        self.model_data = []
        self._init_components()

    def _init_components(self):
        self.flood_generator = FloodGenerator(
            self.RETURN_PERIODS, self.FLOOD_LEVELS, self.rng)

        self.positions, self.elevations = generate_spatial(
            self.n_agents, self.DISTANCE_THRESHOLD, self.GRID_ROWS,
            self.GRID_COLS, self.N_CONNECTORS, self.SLOPE,
            self.NOISE_FACTOR, self.rng)
        self.n_agents = len(self.positions)   # grid may adjust count

        self.attributes = generate_attributes(
            self.n_agents, self.N_ATTRIBUTES, self.N_CLASSES,
            self.ENABLE_HETEROGENEITY, self.rng)

        self.neighborhood_labels, self.n_neighborhoods = identify_neighborhoods(
            self.positions, eps=self.DISTANCE_THRESHOLD,
            min_samples=self.DBSCAN_MIN_SAMPLES)

        self.G = build_network(self.positions, self.attributes, self.DISTANCE_THRESHOLD)
        self.grid = NetworkGrid(self.G)

        # static trait draws (independent Bernoulli, survey-anchored fractions)
        exp_flags = self.rng.random(self.n_agents) < self.P_EXPECT_RISING_DAMAGE
        info_flags = self.rng.random(self.n_agents) < self.P_TRUSTED_INFO
        fc_flags = self.rng.random(self.n_agents) < self.P_FORECAST_PREP

        self.agents_by_node = {}
        for i in range(self.n_agents):
            if self.ENABLE_THRESHOLD_HET and self.PMT_THRESHOLD_STD > 0:
                thr = float(np.clip(
                    self.rng.normal(self.PMT_THRESHOLD_MEAN, self.PMT_THRESHOLD_STD),
                    self.PMT_THRESHOLD_LOW, self.PMT_THRESHOLD_HIGH))
            else:
                thr = self.PMT_THRESHOLD_MEAN
            agent = HouseholdAgent(
                model=self, x=self.positions[i, 0], y=self.positions[i, 1],
                z=self.elevations[i], attributes=self.attributes[i],
                initial_belief=self.INITIAL_BELIEF, pmt_threshold=thr,
                neighborhood_id=int(self.neighborhood_labels[i]),
                expects_rising_damage=bool(exp_flags[i]),
                has_trusted_info=bool(info_flags[i]),
                forecast_prep=bool(fc_flags[i]))
            self.grid.place_agent(agent, i)
            self.agents_by_node[i] = agent

        # Channel 3 fires once, now, before the loop and any flood.
        for agent in list(self.agents):
            agent.apply_information_prior()
            agent.make_decision()   # some may already cross threshold at t=0

    def step(self):
        self.current_step += 1
        self.current_flood_level = self.flood_generator.sample()
        self.flood_history.append(self.current_flood_level)
        for agent in self.agents:
            agent.experience_flood(self.current_flood_level)
        self.agents.shuffle_do("step")
        self._collect_data()

    def _collect_data(self):
        for agent in self.agents:
            self.agent_data.append({
                "Step": self.current_step, "AgentID": agent.unique_id,
                "x": agent.x, "y": agent.y, "z": agent.z,
                "belief": agent.belief, "pmt_threshold": agent.pmt_threshold,
                "neighborhood_id": agent.neighborhood_id,
                "is_retrofitted": agent.is_retrofitted,
                "flood_count": agent.flood_count,
                "retrofit_step": agent.retrofit_step})
        n_ret = sum(1 for a in self.agents if a.is_retrofitted)
        self.model_data.append({
            "Step": self.current_step,
            "flood_level": self.current_flood_level,
            "n_retrofitted": n_ret,
            "pct_retrofitted": 100 * n_ret / self.n_agents,
            "mean_belief": np.mean([a.belief for a in self.agents])})

    def run(self):
        for _ in range(self.time_steps):
            self.step()

    def get_agent_dataframe(self):
        return pd.DataFrame(self.agent_data).set_index(["Step", "AgentID"])

    def get_model_dataframe(self):
        return pd.DataFrame(self.model_data).set_index("Step")


# ============================================================================
# STREAMLIT APPLICATION  (ADAPT-consistent styling; sidebar navigation rail)
# ============================================================================

# ---- palette (matches the ADAPT tool) --------------------------------------
CLR_INK      = "#0f172a"   # slate-900  (rail top, headings)
CLR_INK2     = "#1e293b"   # slate-800  (rail bottom)
CLR_SKY      = "#0ea5e9"   # sky-500    (primary accent / selected pill)
CLR_SKY_DK   = "#0284c7"   # sky-600
CLR_MUTED    = "#64748b"   # slate-500  (captions)
CLR_SLATE300 = "#cbd5e1"
CLR_MODEL    = "#0ea5e9"   # model bars
CLR_OBS      = "#f97316"   # observed bars (orange-500)
CLR_RETRO    = "#22c55e"   # retrofitted (green-500)
CLR_NOT      = "#e2e8f0"   # not retrofitted
APP_PASSWORD = "NY2026VA"


def _inject_css():
    import streamlit as st
    st.markdown(f"""<style>
      .block-container {{ padding-top: 1.2rem; }}
      h2 {{ font-size: 1.75rem !important; }}
      h3 {{ font-size: 1.4rem !important; }}
      /* ---- dark navigation rail (sidebar) ---- */
      section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, {CLR_INK} 0%, {CLR_INK2} 100%) !important;
          border-right: 1px solid rgba(148,163,184,0.18);
          min-width: 250px !important; max-width: 300px !important;
      }}
      section[data-testid="stSidebar"] > div,
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
      section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
          background: transparent !important;
      }}
      .rail-brand {{ display:flex; flex-direction:column; align-items:flex-start;
          padding: 0 0.35rem 0.8rem 0.35rem; margin: 0 0 0.6rem 0;
          border-bottom: 1px solid rgba(148,163,184,0.22); }}
      /* logo image at top of the rail */
      section[data-testid="stSidebar"] [data-testid="stImage"] {{
          margin: 0.2rem 0 0.6rem 0; }}
      section[data-testid="stSidebar"] [data-testid="stImage"] img {{
          border-radius: 8px; }}
      .rail-word {{ font-size: 1.5rem; font-weight: 800; letter-spacing: 0.4px;
          color: #38bdf8; line-height: 1.05; }}
      .rail-sub  {{ font-size: 0.62rem; color: {CLR_SLATE300}; font-weight: 500;
          margin-top: 4px; line-height: 1.25; }}
      /* nav radio -> pill list */
      section[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 5px; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label {{
          display:flex; align-items:center; gap:0; padding: 0.6rem 0.85rem; margin:0; width:100%;
          border-radius: 10px; cursor:pointer; background: transparent;
          transition: background-color .2s ease, transform .2s ease, box-shadow .2s ease; }}
      /* Hide the native radio input AND its circular ring across Streamlit
         builds. The ring is the label's first child div on older builds and
         a div that wraps the <input> on newer ones; cover both without
         touching the text container (which holds a stMarkdownContainer). */
      section[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {{ display:none !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) {{
          display:none !important; width:0 !important; margin:0 !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label [data-baseweb="radio"] > div:first-child {{
          display:none !important; }}
      /* Force the label text visible and white. */
      section[data-testid="stSidebar"] [role="radiogroup"] > label,
      section[data-testid="stSidebar"] [role="radiogroup"] > label div,
      section[data-testid="stSidebar"] [role="radiogroup"] > label p,
      section[data-testid="stSidebar"] [role="radiogroup"] > label span {{
          color:#fff !important; font-weight:600; font-size:0.95rem;
          visibility:visible !important; opacity:1 !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label [data-testid="stMarkdownContainer"] {{
          display:block !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
          background: rgba(148,163,184,0.16); transform: translateX(3px); }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
          background: linear-gradient(135deg, {CLR_SKY} 0%, {CLR_SKY_DK} 100%);
          box-shadow: 0 6px 16px rgba(14,165,233,0.40); transform: translateX(3px); }}
      /* run button in the rail */
      section[data-testid="stSidebar"] .stButton > button {{
          background: linear-gradient(135deg, {CLR_SKY} 0%, {CLR_SKY_DK} 100%);
          color:#fff !important; font-weight:700; border:0; border-radius:10px;
          padding: 0.55rem 0.8rem; width:100%;
          box-shadow: 0 6px 16px rgba(14,165,233,0.35); }}
      section[data-testid="stSidebar"] .stButton > button:hover {{ filter: brightness(1.07); }}
      section[data-testid="stSidebar"] .rail-label {{ color:{CLR_SLATE300};
          font-size:0.72rem; font-weight:700; letter-spacing:0.6px;
          text-transform:uppercase; margin: 0.2rem 0 0.3rem 0.15rem; }}
      section[data-testid="stSidebar"] hr {{ border-color: rgba(148,163,184,0.20); margin: 0.7rem 0; }}
      /* progress bar under run */
      section[data-testid="stSidebar"] [data-testid="stProgress"] > div > div {{
          background: {CLR_SKY} !important; }}
      /* config summary chips on results */
      .chips {{ display:flex; flex-wrap:wrap; gap:0.4rem; margin: 0.1rem 0 1.1rem 0; }}
      .chip {{ background:#f1f5f9; border:1px solid #e2e8f0; border-radius:999px;
          padding: 0.18rem 0.7rem; font-size:0.8rem; color:#334155; }}
      .chip b {{ color:{CLR_INK}; }}
      .cfg-card {{ background: linear-gradient(135deg,#EAF6FD,#D6ECF8);
          border-radius: 10px; padding: 0.75rem 1rem; }}
      .cfg-card .lab {{ color:{CLR_SKY_DK}; font-size:0.8rem; font-weight:600; }}
      .cfg-card .val {{ color:{CLR_INK}; font-size:1.5rem; font-weight:800; }}
      .tab-desc {{ font-size:1.02rem; color:{CLR_MUTED}; font-style:italic;
          margin-bottom:1rem; padding:0.5rem 0.7rem; background:#f8fafc;
          border-radius:0.25rem; border-left:3px solid {CLR_SKY}; }}
      .doc-h {{ font-size:1.15rem; font-weight:800; color:{CLR_INK};
          border-bottom:2px solid {CLR_SKY}; padding-bottom:0.25rem;
          margin: 1.6rem 0 0.7rem 0; }}
    </style>""", unsafe_allow_html=True)


def _check_password():
    """ADAPT-style gate. Returns True once the correct code is entered."""
    import streamlit as st

    def _entered():
        if st.session_state.get("password") == APP_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:2.5rem;font-weight:800;color:{CLR_SKY};'
                'text-align:center;">\U0001F512 Flood Adaptation ABM</p>',
                unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center;color:{CLR_MUTED};margin-bottom:2rem;">'
                'Bayesian Belief Updating \u2014 Household Flood-Retrofit Model</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.text_input("Enter password to access the tool:", type="password",
                      on_change=_entered, key="password")
        if st.session_state.get("password_correct") is False:
            st.error("\U0001F615 Incorrect password. Please try again.")
    return False


def _collect_params():
    """Read every parameter box (Settings page) into a params dict."""
    import streamlit as st
    D = DEFAULTS
    S = st.session_state

    def g(key, default):
        return S.get(f"p_{key}", default)

    params = dict(D)
    params.update(
        TIME_STEPS=int(g("TIME_STEPS", D["TIME_STEPS"])),
        RANDOM_SEED=int(g("RANDOM_SEED", D["RANDOM_SEED"])),
        N_AGENTS=int(g("N_AGENTS", D["N_AGENTS"])),
        GRID_ROWS=int(g("GRID_ROWS", D["GRID_ROWS"])),
        GRID_COLS=int(g("GRID_COLS", D["GRID_COLS"])),
        N_CONNECTORS=int(g("N_CONNECTORS", D["N_CONNECTORS"])),
        SLOPE=g("SLOPE", D["SLOPE"]), NOISE_FACTOR=g("NOISE_FACTOR", D["NOISE_FACTOR"]),
        ENABLE_HETEROGENEITY=g("ENABLE_HETEROGENEITY", D["ENABLE_HETEROGENEITY"]),
        N_ATTRIBUTES=int(g("N_ATTRIBUTES", D["N_ATTRIBUTES"])),
        N_CLASSES=int(g("N_CLASSES", D["N_CLASSES"])),
        DISTANCE_THRESHOLD=g("DISTANCE_THRESHOLD", D["DISTANCE_THRESHOLD"]),
        DBSCAN_MIN_SAMPLES=int(g("DBSCAN_MIN_SAMPLES", D["DBSCAN_MIN_SAMPLES"])),
        INITIAL_BELIEF=g("INITIAL_BELIEF", D["INITIAL_BELIEF"]),
        LAMBDA_FLOOD=g("LAMBDA_FLOOD", D["LAMBDA_FLOOD"]),
        LAMBDA_SEVERITY=g("LAMBDA_SEVERITY", D["LAMBDA_SEVERITY"]),
        P_EXPECT_RISING_DAMAGE=g("P_EXPECT_RISING_DAMAGE", D["P_EXPECT_RISING_DAMAGE"]),
        LAMBDA_SOCIAL=g("LAMBDA_SOCIAL", D["LAMBDA_SOCIAL"]),
        LAMBDA_SIMILARITY=g("LAMBDA_SIMILARITY", D["LAMBDA_SIMILARITY"]),
        SIM_THRESHOLD=g("SIM_THRESHOLD", D["SIM_THRESHOLD"]),
        LAMBDA_INFO=g("LAMBDA_INFO", D["LAMBDA_INFO"]),
        LAMBDA_FORECAST=g("LAMBDA_FORECAST", D["LAMBDA_FORECAST"]),
        P_TRUSTED_INFO=g("P_TRUSTED_INFO", D["P_TRUSTED_INFO"]),
        P_FORECAST_PREP=g("P_FORECAST_PREP", D["P_FORECAST_PREP"]),
        PMT_THRESHOLD_MEAN=g("PMT_THRESHOLD_MEAN", D["PMT_THRESHOLD_MEAN"]),
        PMT_THRESHOLD_STD=g("PMT_THRESHOLD_STD", D["PMT_THRESHOLD_STD"]),
        PMT_THRESHOLD_LOW=g("PMT_THRESHOLD_LOW", D["PMT_THRESHOLD_LOW"]),
        PMT_THRESHOLD_HIGH=g("PMT_THRESHOLD_HIGH", D["PMT_THRESHOLD_HIGH"]),
        ENABLE_THRESHOLD_HET=g("ENABLE_THRESHOLD_HET", D["ENABLE_THRESHOLD_HET"]),
    )
    try:
        params["RETURN_PERIODS"] = [int(x) for x in
                                    str(g("RETURN_PERIODS", "10,20,50,100")).split(",")]
        params["FLOOD_LEVELS"] = [float(x) for x in
                                  str(g("FLOOD_LEVELS", "0.05,0.10,0.15,0.30")).split(",")]
    except Exception:
        params["RETURN_PERIODS"] = D["RETURN_PERIODS"]
        params["FLOOD_LEVELS"] = D["FLOOD_LEVELS"]
    return params


def _run_with_progress(params, progress_slot):
    """Run the model step-by-step, updating a progress bar in the rail."""
    model = FloodAdaptationModel(params, seed=params["RANDOM_SEED"])
    total = params["TIME_STEPS"]
    bar = progress_slot.progress(0.0)
    for i in range(total):
        model.step()
        if (i + 1) % max(1, total // 100) == 0 or i + 1 == total:
            bar.progress((i + 1) / total)
    bar.progress(1.0)
    return model


# ---------------------------------------------------------------------------
# PAGE RENDERERS
# ---------------------------------------------------------------------------

def _sec(title, subtitle, color):
    import streamlit as st
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{color}14,{color}05);"
        f"border-left:5px solid {color};border-radius:8px;padding:0.55rem 0.9rem;"
        f"margin:0.4rem 0 0.9rem 0;'>"
        f"<div style='font-size:1.05rem;font-weight:800;color:{CLR_INK};'>{title}</div>"
        f"<div style='font-size:0.82rem;color:{CLR_MUTED};'>{subtitle}</div></div>",
        unsafe_allow_html=True)


def _page_settings():
    import streamlit as st
    D = DEFAULTS
    # section accent colors
    C_BELIEF = "#0284c7"   # sky
    C_CH1    = "#0ea5e9"   # flood - sky
    C_CH2    = "#22c55e"   # proximity - green
    C_CH3    = "#f97316"   # information - orange
    C_PMT    = "#7c3aed"   # threshold - violet
    C_MINOR  = "#64748b"   # structural - slate

    st.markdown("## \u2699\ufe0f Settings")
    st.markdown('<div class="tab-desc">Model parameters, grouped by role. The '
                'primary drivers \u2014 belief, the three evidence channels, and the '
                'decision threshold \u2014 come first; structural and environment '
                'settings are grouped at the bottom. Survey-anchored defaults '
                'appear in each field\u2019s tooltip. Set values, then press '
                '<b>Run Simulation</b> in the left rail.</div>',
                unsafe_allow_html=True)

    def nb(label, key, step, fmt="%.2f", minv=0.0, maxv=None, help=None):
        st.number_input(label, value=float(D[key]), min_value=float(minv),
                        max_value=(float(maxv) if maxv is not None else None),
                        step=float(step), format=fmt, key=f"p_{key}", help=help)

    def ni(label, key, minv, maxv, step=1, help=None):
        st.number_input(label, value=int(D[key]), min_value=int(minv),
                        max_value=int(maxv), step=int(step), key=f"p_{key}", help=help)

    # ===================== PRIMARY DRIVERS =====================
    st.markdown("### \U0001F3AF Core decision drivers")

    _sec("Belief & Decision Threshold",
         "Prior belief that a home should be retrofitted, and the PMT bar it must clear.",
         C_BELIEF)
    b1, b2 = st.columns(2)
    with b1:
        nb("Initial Belief  \u2014  P(H\u2081)", "INITIAL_BELIEF", 0.01, "%.2f", 0.01, 0.99,
           help="Prior probability that a household should retrofit.")
        nb("PMT Threshold Mean  (\u03b8)", "PMT_THRESHOLD_MEAN", 0.01, "%.2f", 0.01, 0.99,
           help="Belief level at which a household acts (Rogers, 1975).")
    with b2:
        st.checkbox("Threshold Heterogeneity", value=D["ENABLE_THRESHOLD_HET"],
                    key="p_ENABLE_THRESHOLD_HET",
                    help="Draw individual thresholds from a clipped Normal.")
        cc1, cc2, cc3 = st.columns(3)
        with cc1: nb("Std Dev", "PMT_THRESHOLD_STD", 0.01, "%.2f", 0.0, 0.5)
        with cc2: nb("Lower", "PMT_THRESHOLD_LOW", 0.01, "%.2f", 0.01, 0.99)
        with cc3: nb("Upper", "PMT_THRESHOLD_HIGH", 0.01, "%.2f", 0.01, 0.99)

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        _sec("Channel 1 \u00b7 Flood Experience",
             "Belief update per flood, amplified by perceived severity.", C_CH1)
        nb("\u03bb_flood  (base, per flood)", "LAMBDA_FLOOD", 0.01, "%.2f", 1.0, None,
           help="Bayes factor per flood. Survey anchor 1.52.")
        nb("\u03bb_severity  (\u00d7 rising damage)", "LAMBDA_SEVERITY", 0.1, "%.2f", 1.0, None,
           help="Perceived-severity multiplier on a flood, for agents expecting "
                "rising flood damage. Survey anchor 2.40 (Rogers, 1975).")
        nb("Fraction expecting rising damage", "P_EXPECT_RISING_DAMAGE",
           0.01, "%.2f", 0.0, 1.0, help="Assigned once at t=0. Survey: 0.69.")
    with ch2:
        _sec("Channel 2 \u00b7 Proximity",
             "Social learning from retrofitted neighbours, stronger if similar.", C_CH2)
        nb("\u03bb_social  (base, per neighbor)", "LAMBDA_SOCIAL", 0.01, "%.2f", 1.0, None,
           help="Bayes factor per newly-retrofitted connected neighbor (Granovetter, 1978).")
        nb("\u03bb_similarity  (\u00d7 if similar)", "LAMBDA_SIMILARITY", 0.1, "%.2f", 1.0, None,
           help="Applied when the retrofitted neighbor is similar "
                "(Gower S \u2265 threshold). 1.0 = no similarity effect.")
        nb("Similarity threshold  (S \u2265)", "SIM_THRESHOLD", 0.05, "%.2f", 0.0, 1.0,
           help="A neighbor counts as similar at or above this Gower similarity.")
    with ch3:
        _sec("Channel 3 \u00b7 Information",
             "One-time t=0 prior from trusted information and forecast use.", C_CH3)
        nb("\u03bb_info  (base, if trusted info)", "LAMBDA_INFO", 0.01, "%.2f", 1.0, None,
           help="One-time t=0 factor for agents with a trusted source. "
                "Survey: weak alone (OR ~1.39).")
        nb("\u03bb_forecast  (\u00d7 if forecast-prep)", "LAMBDA_FORECAST", 0.05, "%.2f", 1.0, None,
           help="Forecast-preparer amplifier. Survey ~3.2.")
        nb("Fraction with trusted information", "P_TRUSTED_INFO", 0.01, "%.2f", 0.0, 1.0,
           help="Survey (never-flooded): 0.48.")
        nb("Fraction preparing on forecasts", "P_FORECAST_PREP", 0.01, "%.2f", 0.0, 1.0,
           help="Survey (never-flooded): 0.65.")

    # ===================== SECONDARY / STRUCTURAL =====================
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown("### \U0001F527 Environment & run settings")
    st.caption("Population, geography, network, flood generator, and run controls. "
               "These shape the setting rather than the decision mechanism; the "
               "defaults are sensible for most experiments.")

    with st.expander("Population, spatial layout & network", expanded=False):
        s1, s2, s3 = st.columns(3)
        with s1:
            _sec("Population", "How many households, and how varied.", C_MINOR)
            ni("Number of Agents", "N_AGENTS", 10, 100000, 10)
            st.checkbox("Attribute Heterogeneity", value=D["ENABLE_HETEROGENEITY"],
                        key="p_ENABLE_HETEROGENEITY",
                        help="If off, all agents identical (S=1 for all pairs).")
            ni("Attributes per Agent", "N_ATTRIBUTES", 1, 10)
            ni("Classes per Attribute", "N_CLASSES", 1, 10)
        with s2:
            _sec("Spatial layout", "Neighbourhood grid and terrain.", C_MINOR)
            ni("Grid Rows", "GRID_ROWS", 1, 10)
            ni("Grid Cols", "GRID_COLS", 1, 10)
            ni("Connectors", "N_CONNECTORS", 0, 10)
            nb("Elevation Slope", "SLOPE", 0.01, "%.2f", 0.01, 2.0)
            nb("Elevation Noise", "NOISE_FACTOR", 0.01, "%.2f", 0.0, 1.0)
        with s3:
            _sec("Network", "Who is connected, and cluster detection.", C_MINOR)
            nb("Distance Threshold", "DISTANCE_THRESHOLD", 0.01, "%.2f", 0.01, 0.5,
               help="Households within this distance are network neighbors.")
            ni("DBSCAN Min Samples", "DBSCAN_MIN_SAMPLES", 2, 10)

    with st.expander("Flood generator & run controls", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            _sec("Flood (GEV)", "Extreme-value flood sampler inputs.", C_MINOR)
            st.text_input("Return Periods",
                          value=", ".join(str(x) for x in D["RETURN_PERIODS"]),
                          key="p_RETURN_PERIODS",
                          help="Comma-separated return periods (years).")
            st.text_input("Flood Levels",
                          value=", ".join(str(x) for x in D["FLOOD_LEVELS"]),
                          key="p_FLOOD_LEVELS",
                          help="Comma-separated flood levels matching the return periods.")
        with f2:
            _sec("Run controls", "Length and reproducibility of the simulation.", C_MINOR)
            ni("Time Steps", "TIME_STEPS", 10, 10000, 10)
            ni("Random Seed", "RANDOM_SEED", 0, 10_000_000, 1)


def _draw_edges(ax, model, alpha=0.25):
    segs = []
    for u, v in model.G.edges():
        au, av = model.agents_by_node[u], model.agents_by_node[v]
        segs.append([(au.x, au.y), (av.x, av.y)])
    if segs:
        ax.add_collection(LineCollection(segs, linewidths=0.5, colors="gray",
                                         alpha=alpha, zorder=1))


def _fig_adoption_flood(model):
    """Adoption curve with the annual flood series on a twin axis."""
    df = model.get_model_dataframe()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(df.index, df["pct_retrofitted"], color=CLR_SKY_DK, lw=2.2,
            label="Retrofitted (%)", zorder=3)
    ax.fill_between(df.index, df["pct_retrofitted"], color=CLR_SKY, alpha=0.15, zorder=2)
    ax.set(xlabel="Time step", ylabel="Retrofitted (%)",
           xlim=(1, model.time_steps), ylim=(0, 100))
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(range(1, len(model.flood_history) + 1), model.flood_history,
            color="#94a3b8", alpha=0.35, zorder=1)
    ax2.set_ylabel("Flood level", color="#64748b")
    ax2.tick_params(axis="y", labelcolor="#64748b")
    ax.set_title("Adoption over time & annual flood levels", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_elevation_comparison(model):
    """Retrofit rate by elevation tercile."""
    agents = list(model.agents)
    zs = np.array([a.z for a in agents])
    t1, t2 = np.percentile(zs, [33.33, 66.67])
    groups = {"Low": [], "Medium": [], "High": []}
    for a in agents:
        if a.z <= t1: groups["Low"].append(a)
        elif a.z <= t2: groups["Medium"].append(a)
        else: groups["High"].append(a)
    labels, rates = [], []
    for lab, g in groups.items():
        if g:
            rates.append(100 * sum(1 for a in g if a.is_retrofitted) / len(g))
            labels.append(f"{lab}\n(n={len(g)})")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(labels, rates, color=["#ef4444", "#f97316", "#22c55e"],
                  edgecolor="black")
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set(xlabel="Elevation tercile", ylabel="Retrofit rate (%)",
           ylim=(0, max(rates) * 1.2 if rates else 100))
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Retrofit adoption by elevation", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_comparison(model, cum, obs):
    """Model vs observed cumulative retrofit rate."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(3); w = 0.38
    bm = ax.bar(x - w / 2, cum, w, label="Model", color=CLR_MODEL, edgecolor="black")
    bo = ax.bar(x + w / 2, obs, w, label="Observed", color=CLR_OBS, edgecolor="black")
    for bars in (bm, bo):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set(xlabel="Flood experience (cumulative, at most k)",
           ylabel="Retrofit rate (%)",
           ylim=(0, max(max(cum), max(obs)) * 1.25 + 1))
    ax.set_xticks(x); ax.set_xticklabels(["0", "\u2264 4", "5+ (all)"])
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Model vs observed (cumulative)", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_belief_evolution(model):
    """Mean belief with 10-90 band, threshold band, and prior line."""
    dfm = model.get_model_dataframe()
    dfa = model.get_agent_dataframe()
    steps = dfm.index.values
    p10, p90 = [], []
    for s in steps:
        b = dfa.xs(s, level="Step")["belief"].values
        p10.append(np.percentile(b, 10)); p90.append(np.percentile(b, 90))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.fill_between(steps, p10, p90, alpha=0.2, color=CLR_SKY,
                    label="10th\u201390th percentile")
    ax.plot(steps, dfm["mean_belief"], color=CLR_SKY_DK, lw=2, label="Mean $P(H_1)$")
    final = dfa.xs(steps[-1], level="Step")
    thr = final["pmt_threshold"].values
    tp10, tp90 = np.percentile(thr, [10, 90]); tmean = thr.mean()
    ax.axhspan(tp10, tp90, color="#ef4444", alpha=0.08,
               label=f"Threshold 10\u201390th: [{tp10:.2f}, {tp90:.2f}]")
    ax.axhline(tmean, color="#ef4444", ls="--", label=f"Mean \u03b8 = {tmean:.2f}")
    ax.axhline(model.INITIAL_BELIEF, color="gray", ls=":",
               label=f"Prior = {model.INITIAL_BELIEF:.2f}")
    ax.set(xlabel="Time step", ylabel="$P(H_1)$")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Bayesian belief evolution", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_network(model):
    """Network with retrofit status; label = personal flood count."""
    fig, ax = plt.subplots(figsize=(7, 5.4))
    agents = list(model.agents)
    _draw_edges(ax, model, alpha=0.3)
    na = [a for a in agents if not a.is_retrofitted]
    ad = [a for a in agents if a.is_retrofitted]
    if na:
        ax.scatter([a.x for a in na], [a.y for a in na], c="#e2e8f0", s=120,
                   edgecolor="black", linewidth=0.8, zorder=2)
    if ad:
        sc = ax.scatter([a.x for a in ad], [a.y for a in ad],
                        c=[a.retrofit_step for a in ad], cmap="YlGn", s=130,
                        edgecolor="black", linewidth=0.8, zorder=3,
                        vmin=1, vmax=max(1, model.current_step))
        fig.colorbar(sc, ax=ax, shrink=0.7).set_label("Retrofit step")
    ax.set(xlabel="x", ylabel="y", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(handles=[Patch(facecolor="#e2e8f0", edgecolor="black", label="Not retrofitted"),
                       Patch(facecolor="#31a354", edgecolor="black", label="Retrofitted")],
              loc="upper right", fontsize=8)
    ax.set_title("Social network (color = retrofit step)", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig_spatial(model):
    """Spatial map: elevation colormap with retrofitted homes ringed green."""
    fig, ax = plt.subplots(figsize=(7, 5.4))
    agents = list(model.agents)
    _draw_edges(ax, model, alpha=0.2)
    sc = ax.scatter([a.x for a in agents], [a.y for a in agents],
                    c=[a.z for a in agents], cmap="terrain", s=110, alpha=0.75,
                    edgecolor="black", linewidth=0.5, zorder=2)
    for a in agents:
        if a.is_retrofitted:
            ax.scatter(a.x, a.y, facecolors="none", edgecolors=CLR_RETRO,
                       s=150, linewidths=2.0, zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.75).set_label("Elevation")
    ax.set(xlabel="x", ylabel="y", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal")
    ax.legend(handles=[Patch(facecolor="none", edgecolor=CLR_RETRO, label="Retrofitted"),
                       Patch(facecolor="#cbd5e1", edgecolor="black", label="Not retrofitted")],
              loc="upper right", fontsize=8)
    ax.set_title("Spatial: elevation & retrofit status", fontweight="bold")
    fig.tight_layout()
    return fig


def _config_chips(p):
    import streamlit as st
    chips = [
        ("Agents", p["N_AGENTS"]), ("Steps", p["TIME_STEPS"]),
        ("P(H\u2081)", f"{p['INITIAL_BELIEF']:.2f}"),
        ("\u03b8", f"{p['PMT_THRESHOLD_MEAN']:.2f}"),
        ("\u03bb_flood\u00d7\u03bb_sev", f"{p['LAMBDA_FLOOD']:.2f}\u00d7{p['LAMBDA_SEVERITY']:.2f}"),
        ("\u03bb_social\u00d7\u03bb_sim", f"{p['LAMBDA_SOCIAL']:.2f}\u00d7{p['LAMBDA_SIMILARITY']:.2f}"),
        ("\u03bb_info\u00d7\u03bb_fc", f"{p['LAMBDA_INFO']:.2f}\u00d7{p['LAMBDA_FORECAST']:.2f}"),
        ("Seed", p["RANDOM_SEED"]),
    ]
    html = '<div class="chips">' + "".join(
        f'<span class="chip"><b>{k}</b> {v}</span>' for k, v in chips) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _page_results():
    import streamlit as st
    st.markdown("## \U0001F4CA Results")
    if not st.session_state.get("has_run"):
        st.info("No results yet. Set parameters in **Settings**, then press "
                "**Run Simulation** in the left rail.")
        return

    p = st.session_state["run_params"]
    _config_chips(p)

    mdf = st.session_state["model_df"]
    cum = st.session_state["cum_rates"]
    obs = p["OBSERVED_CUM_RATES"]

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    for col, lab, val in [
        (r1c1, "Final retrofitted", f"{mdf['pct_retrofitted'].iloc[-1]:.1f}%"),
        (r1c2, "Mean belief (final)", f"{mdf['mean_belief'].iloc[-1]:.2f}"),
        (r1c3, "Agents", f"{p['N_AGENTS']}"),
        (r1c4, "Steps", f"{p['TIME_STEPS']}")]:
        col.markdown(f"<div class='cfg-card'><div class='lab'>{lab}</div>"
                     f"<div class='val'>{val}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    model = st.session_state.get("model_obj")
    if model is None:
        st.warning("Run the simulation again to regenerate the figures.")
        return

    # ---- six figures in a 3 x 2 grid ----
    figs = [
        ("(a)  Adoption over time & flood levels", _fig_adoption_flood(model)),
        ("(b)  Model vs observed (cumulative)",    _fig_comparison(model, cum, obs)),
        ("(c)  Retrofit adoption by elevation",    _fig_elevation_comparison(model)),
        ("(d)  Bayesian belief evolution",         _fig_belief_evolution(model)),
        ("(e)  Social network",                    _fig_network(model)),
        ("(f)  Spatial map \u2014 elevation & retrofit", _fig_spatial(model)),
    ]
    for i in range(0, 6, 2):
        col_l, col_r = st.columns(2)
        for col, (title, fig) in zip((col_l, col_r), figs[i:i + 2]):
            with col:
                st.markdown(f"#### {title}")
                st.pyplot(fig); plt.close(fig)

    st.markdown("#### Cumulative rates  (model vs survey)")
    st.dataframe(pd.DataFrame({
        "Bin (at most k floods)": ["0", "\u2264 4", "5+ (all)"],
        "Model (%)": [f"{r:.1f}" for r in cum],
        "Observed (%)": [f"{o:.1f}" for o in obs]}),
        hide_index=True, use_container_width=True)

    # ---- data export (as in the previous app) ----
    st.divider()
    st.markdown("#### \U0001F4E5 Export data")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("Download agent data (CSV)",
                           st.session_state["agent_df"].to_csv(),
                           "agent_data.csv", "text/csv", use_container_width=True)
    with e2:
        st.download_button("Download model data (CSV)",
                           mdf.to_csv(), "model_data.csv", "text/csv",
                           use_container_width=True)


def _page_documentation():
    import streamlit as st
    st.markdown("## \U0001F4D8 Documentation")
    st.markdown('<div class="tab-desc">A household flood-retrofit agent-based '
                'model built on Bayesian belief updating in odds form and '
                'Protection Motivation Theory, calibrated to the New York City '
                'Flood Vulnerability Index Survey.</div>', unsafe_allow_html=True)

    # ---- 1 Motivation ----
    st.markdown('<div class="doc-h">1 &nbsp; Motivation</div>', unsafe_allow_html=True)
    st.markdown(
        "Coastal households in New York face rising flood risk, yet only a "
        "minority retrofit their homes. This model asks **why** \u2014 what mix of "
        "personal flood experience, social influence, and trusted information "
        "leads a household to invest in structural protection \u2014 and lets the "
        "user explore how those forces combine to produce neighbourhood-scale "
        "adoption. Every mechanism is grounded in survey evidence, and every "
        "parameter is exposed on the Settings page for experimentation.")

    # ---- 2 Overview ----
    st.markdown('<div class="doc-h">2 &nbsp; Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "Each agent is a household at a fixed location with elevation $z_i$. It "
        "holds a subjective belief $P_i(H_1)\\in[0,1]$ that it should retrofit, "
        "and it retrofits at the first moment that belief reaches its personal "
        "Protection Motivation Theory (PMT) threshold $\\theta_i$. Belief is "
        "revised over time as evidence arrives through three channels. "
        "Retrofitting is **absorbing**: once a household retrofits it takes no "
        "further flood damage and makes no further decisions. Each step, a "
        "global flood level is drawn from a Generalized Extreme Value "
        "distribution fitted to user-specified return periods.")

    # ---- 3 Theoretical framework ----
    st.markdown('<div class="doc-h">3 &nbsp; Theoretical framework</div>',
                unsafe_allow_html=True)

    st.markdown("#### 3.1 &nbsp; Bayesian belief updating \u2014 full formulation")
    st.markdown(
        "Each agent maintains a belief $P_i(H_1)$, where $H_1$ = \u201cI should "
        "retrofit my house\u201d and $H_0$ = \u201cI should not.\u201d Every evidence event "
        "is processed in three algebraic steps (Jaynes, 2003, Ch. 4; Kass & "
        "Raftery, 1995).")
    st.markdown("**Step 1 \u2014 Convert probability to odds.** Odds give a natural "
                "multiplicative framework for accumulating evidence:")
    st.latex(r"O_i = \frac{P_i(H_1)}{1 - P_i(H_1)}")
    st.markdown("For example, a belief of $P_i(H_1)=0.08$ corresponds to odds "
                "$O_i = 0.08/0.92 \\approx 0.087$ \u2014 the agent considers $H_0$ "
                "about eleven times more likely than $H_1$.")
    st.markdown("**Step 2 \u2014 Multiply the odds by a Bayes factor.** A Bayes "
                "factor $\\lambda$ is the likelihood ratio of a single "
                "observation (Kass & Raftery, 1995):")
    st.latex(r"\lambda = \frac{P(\text{evidence}\mid H_1)}{P(\text{evidence}\mid H_0)}"
             r"\qquad O_i^{\text{post}} = O_i^{\text{prior}} \times \lambda")
    st.markdown("A factor $\\lambda>1$ shifts belief toward retrofitting; "
                "$\\lambda=1$ is uninformative. When several channels fire in "
                "the same step their factors compose multiplicatively, and "
                "because multiplication is commutative the update is "
                "order-independent (Good, 1950):")
    st.latex(r"O_i^{\text{post}} = O_i^{\text{prior}}\times "
             r"\lambda_{\text{exp}}\times\lambda_{\text{prox}}\times\lambda_{\text{info}}")
    st.markdown("**Step 3 \u2014 Convert the posterior odds back to a probability:**")
    st.latex(r"P_i^{\text{post}}(H_1) = \frac{O_i^{\text{post}}}{1 + O_i^{\text{post}}}")
    st.markdown("This is algebraically identical to the standard form of Bayes\u2019 "
                "theorem; the advantage is that each evidence source is a single "
                "scalar and sequential updates are just repeated multiplication.")

    st.markdown("#### 3.2 &nbsp; Channel structure: base factor \u00d7 conditional multiplier")
    st.markdown(
        "Each of the three channels contributes a **base factor** times a "
        "**conditional multiplier**. The multiplier equals 1 \u2014 no effect \u2014 "
        "whenever its trigger is absent, so a channel that does not fire leaves "
        "belief unchanged. This keeps every channel to at most two interpretable "
        "parameters and lets a channel switch cleanly on and off.")

    st.markdown("#### 3.3 &nbsp; Channel 1: personal flood experience")
    st.markdown(
        "Each step a global flood level $f_t$ is drawn from a GEV distribution "
        "(Coles, 2001). Agent $i$ is flooded when $f_t > z_i$. The update is "
        "asymmetric \u2014 flood years are psychologically salient while dry years "
        "are cognitively inert (availability heuristic; Tversky & Kahneman, "
        "1974). On a flood, the base per-flood factor $\\lambda_{flood}$ is "
        "multiplied by a **perceived-severity** multiplier $\\lambda_{severity}$ "
        "for households that expect flood damage to worsen (Rogers, 1975; "
        "Floyd, Prentice-Dunn & Rogers, 2000):")
    st.latex(r"""\lambda_{\text{exp},i}^{(t)} =
\begin{cases}
\lambda_{flood}\cdot\lambda_{severity} & f_t > z_i \ \text{and agent expects rising damage}\\
\lambda_{flood} & f_t > z_i \ \text{and agent does not}\\
1 & f_t \le z_i \quad(\text{not flooded; no update})
\end{cases}""")
    st.markdown("A single per-flood factor is used (no separate first-flood "
                "term). Survey anchors: $\\lambda_{flood}=1.52$ (owner per-flood "
                "odds ratio) and $\\lambda_{severity}=2.40$ (expecting- vs "
                "not-expecting rising damage). Because each flood contributes "
                "only a modest factor, an agent must experience several floods "
                "before belief nears the threshold \u2014 consistent with observed "
                "low adoption despite repeated flooding.")

    st.markdown("#### 3.4 &nbsp; Channel 2: proximity-based social learning")
    st.markdown(
        "Network ties are binary: agents $i$ and $j$ are connected when their "
        "Euclidean distance $d(i,j)\\le$ `DISTANCE_THRESHOLD`. Let "
        "$\\mathcal{N}_i^{(t)}$ be the connected neighbours **newly observed** "
        "to have retrofitted at step $t$ (each counted once \u2014 one-shot "
        "learning). For each such neighbour the base factor $\\lambda_{social}$ "
        "(Granovetter, 1978) is multiplied by a similarity multiplier "
        "$\\lambda_{similarity}$ when the neighbour is **similar**, i.e. their "
        "Gower similarity meets the threshold (Gower, 1971; McPherson et al., "
        "2001):")
    st.latex(r"""\lambda_{\text{prox},ij} =
\begin{cases}
\lambda_{social}\cdot\lambda_{similarity} & j\ \text{retrofitted and } S(i,j)\ge S^\ast\\
\lambda_{social} & j\ \text{retrofitted and } S(i,j)< S^\ast\\
1 & \text{no newly-retrofitted neighbour}
\end{cases}""")
    st.markdown("Similarity is the fraction of discrete attributes on which two "
                "agents agree,")
    st.latex(r"S(i,j) = \frac{1}{A}\sum_{a=1}^{A}\mathbb{1}\!\left[x_{ia}=x_{ja}\right]"
             r"\ \in[0,1]\quad(\text{Gower, 1971})")
    st.markdown("so a retrofitting neighbour who resembles the agent carries "
                "more weight than a dissimilar one, and with no retrofitted "
                "neighbour the channel is inert.")

    st.markdown("#### 3.5 &nbsp; Channel 3: trusted information")
    st.markdown(
        "Applied **once, at initialisation**, because trusted-information and "
        "forecast-preparation are stable household traits rather than repeated "
        "events. Households with a trusted flood-information source apply a base "
        "factor $\\lambda_{info}$, multiplied by a forecast multiplier "
        "$\\lambda_{forecast}$ for those who also prepare on the basis of flood "
        "forecasts:")
    st.latex(r"""\lambda_{\text{info},i} =
\begin{cases}
\lambda_{info}\cdot\lambda_{forecast} & \text{trusted info and forecast-preparer}\\
\lambda_{info} & \text{trusted info only}\\
1 & \text{no trusted information}
\end{cases}""")
    st.markdown("In the survey, having a trusted source alone is weak "
                "(odds ratio \u2248 1.39, n.s.) while forecast preparation is the "
                "stronger amplifier (odds ratio \u2248 3.2), which is why the base "
                "is the weaker term and the multiplier the stronger one. This "
                "channel represents an informational prior that lifts belief at "
                "$t=0$ for informed households.")

    # ---- 4 Static traits ----
    st.markdown('<div class="doc-h">4 &nbsp; Static household traits</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Three binary traits gate the conditional multipliers above: *expects "
        "rising damage*, *has trusted information*, and *prepares on forecasts*. "
        "Each is drawn once at $t=0$ as an independent Bernoulli variable using "
        "survey-anchored fractions among never-flooded households \u2014 0.69, 0.48, "
        "and 0.65 respectively \u2014 and is fixed for the life of the simulation.")

    # ---- 5 Decision rule ----
    st.markdown('<div class="doc-h">5 &nbsp; Decision rule (PMT threshold)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "A household retrofits at the first step its belief reaches its "
        "threshold, $P_i(H_1)\\ge\\theta_i$ (Rogers, 1975). Thresholds are "
        "heterogeneous, drawn from a Normal distribution clipped to a user-set "
        "band $[\\theta_{low},\\theta_{high}]$; belief is homogeneous at $t=0$. "
        "Because only the gap between belief and threshold governs behaviour, a "
        "single heterogeneous quantity (the threshold) is sufficient and "
        "identifiable \u2014 belief and threshold spreads are not separately "
        "estimated.")
    st.latex(r"\text{retrofit}_i \iff P_i(H_1)\ge\theta_i,\qquad "
             r"\theta_i\sim\mathcal{N}(\mu_\theta,\sigma_\theta)\ \text{clipped to}\ "
             r"[\theta_{low},\theta_{high}]")

    # ---- 6 Understanding the Bayes factors ----
    st.markdown('<div class="doc-h">6 &nbsp; Understanding the Bayes factors</div>',
                unsafe_allow_html=True)
    st.markdown(
        "A Bayes factor greater than 1 is evidence for retrofitting and "
        "multiplies the odds. Two floods contribute $\\lambda_{flood}^2$; a "
        "flood experienced by a household expecting rising damage contributes "
        "$\\lambda_{flood}\\,\\lambda_{severity}$. Working in odds means these "
        "combine by multiplication and the update is order-independent \u2014 the "
        "same evidence yields the same belief regardless of the order in which "
        "it arrives (Good, 1950).")

    # ---- 7 Key properties ----
    st.markdown('<div class="doc-h">7 &nbsp; Key properties</div>',
                unsafe_allow_html=True)
    st.markdown(
        "- **Absorbing retrofit.** Adaptation is permanent; retrofitted homes "
        "leave the risk pool.\n"
        "- **Experience dominates.** With weak social and information channels, "
        "adoption rises steeply with personal flood count \u2014 the empirical "
        "pattern in the survey.\n"
        "- **Parsimony.** Only the threshold is heterogeneous; each channel adds "
        "at most one base factor and one conditional multiplier.\n"
        "- **Identifiability.** Only belief minus threshold enters the decision, "
        "so their spreads are not separately estimated.\n"
        "- **Order independence.** Multiplicative odds updating makes belief "
        "path-independent for a given set of evidence.")

    # ---- 8 Parameters ----
    st.markdown('<div class="doc-h">8 &nbsp; Parameters</div>', unsafe_allow_html=True)
    st.markdown(
        "All parameters live on the **Settings** page, grouped into core "
        "decision drivers (belief, the three channels, the threshold) and "
        "structural environment settings. Survey-anchored starting values: "
        "$\\lambda_{flood}=1.52$, $\\lambda_{severity}=2.40$, "
        "$\\lambda_{forecast}\\approx3.2$; trait fractions 0.69 / 0.48 / 0.65. "
        "The opening defaults are deliberately de-escalated from these raw "
        "point estimates so the model is non-saturated out of the box: each "
        "survey odds ratio was estimated holding the other channels at "
        "baseline, so stacking them all at once over-drives belief. Tune each "
        "factor upward toward its survey anchor (shown in every field\u2019s "
        "tooltip) while watching the cumulative bars on the Results page.")

    # ---- 9 Calibration target ----
    st.markdown('<div class="doc-h">9 &nbsp; Calibration target</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The Results page compares the model\u2019s **cumulative** \u201cat most $k$ "
        "floods\u201d retrofit rate against the survey values **18.0 / 22.3 / "
        "27.4 %** for the 0 / $\\le 4$ / all bins. Cumulative targets are "
        "preferred over per-bin rates because they are more robust given the "
        "survey\u2019s modest sample size.")

    # ---- 10 Results plots ----
    st.markdown('<div class="doc-h">10 &nbsp; Reading the results</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The Results page shows six figures in a 3\u00d72 grid: "
        "**(a)** adoption over time with the annual flood series; "
        "**(b)** the model-vs-observed cumulative comparison; "
        "**(c)** retrofit rate by elevation tercile; "
        "**(d)** Bayesian belief evolution with the threshold band; "
        "**(e)** the social network coloured by retrofit step; and "
        "**(f)** the spatial map of elevation with retrofitted homes ringed. "
        "Agent- and model-level data can be downloaded as CSV beneath the "
        "figures.")

    # ---- 11 Workflow ----
    st.markdown('<div class="doc-h">11 &nbsp; Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        "1. Open **Settings** and set parameters (or keep the defaults).  \n"
        "2. Press **Run Simulation** in the left rail; the progress bar beneath "
        "the button tracks the run.  \n"
        "3. Open **Results** for the six figures, the cumulative comparison "
        "against survey data, and CSV export. A compact summary of the run\u2019s "
        "settings appears at the top of the page.")

    # ---- 12 References ----
    st.markdown('<div class="doc-h">12 &nbsp; References</div>', unsafe_allow_html=True)
    st.markdown(
        "Coles (2001) *An Introduction to Statistical Modeling of Extreme "
        "Values.* \u00b7 Ester, Kriegel, Sander & Xu (1996) *A density-based "
        "algorithm for discovering clusters* (DBSCAN). \u00b7 Floyd, Prentice-Dunn "
        "& Rogers (2000) *A meta-analysis of research on Protection Motivation "
        "Theory.* \u00b7 Good (1950) *Probability and the Weighing of Evidence.* "
        "\u00b7 Gower (1971) *A general coefficient of similarity and some of its "
        "properties.* \u00b7 Granovetter (1978) *Threshold models of collective "
        "behavior.* \u00b7 Jaynes (2003) *Probability Theory: The Logic of "
        "Science.* \u00b7 Kass & Raftery (1995) *Bayes factors.* \u00b7 McPherson, "
        "Smith-Lovin & Cook (2001) *Birds of a feather: homophily in social "
        "networks.* \u00b7 Rogers (1975) *A protection motivation theory of fear "
        "appeals and attitude change.* \u00b7 Tversky & Kahneman (1974) *Judgment "
        "under uncertainty: heuristics and biases.*")



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _run_app():
    import streamlit as st

    st.set_page_config(page_title="Flood Adaptation ABM",
                       page_icon="\U0001F30A", layout="wide",
                       initial_sidebar_state="expanded")
    _inject_css()

    if not _check_password():
        st.stop()

    ss = st.session_state
    # Apply any pending navigation request BEFORE the radio widget is created.
    # (Streamlit forbids mutating a widget's own key after instantiation, so
    # the run handler sets ss["pending_nav"] and we consume it here instead.)
    if ss.get("pending_nav"):
        ss["nav"] = ss.pop("pending_nav")
    if "nav" not in ss:
        ss["nav"] = "\U0001F4CA  Results" if ss.get("has_run") else "\u2699\ufe0f  Settings"

    # ---- navigation rail ----
    with st.sidebar:
        # Logo at the very top of the rail (as in the ADAPT tool). Falls back
        # to a text brand block if logo.png is not present in the app folder.
        import os as _os
        if _os.path.exists("logo.png"):
            try:
                st.image("logo.png", width="stretch")
            except TypeError:
                st.image("logo.png", use_container_width=True)
            st.markdown('<div class="rail-brand" style="border-top:0;padding-top:0;">'
                        '<span class="rail-sub">Bayesian Belief Updating &middot; v13'
                        '</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rail-brand"><span class="rail-word">'
                        '\U0001F30A Flood ABM</span><span class="rail-sub">'
                        'Bayesian Belief Updating &middot; v13</span></div>',
                        unsafe_allow_html=True)

        nav = st.radio("Navigation",
                       ["\U0001F4D8  Documentation", "\u2699\ufe0f  Settings",
                        "\U0001F4CA  Results"],
                       key="nav", label_visibility="collapsed")

        st.markdown('<hr/>', unsafe_allow_html=True)
        st.markdown('<div class="rail-label">Simulation</div>', unsafe_allow_html=True)
        run_clicked = st.button("\u25B6  Run Simulation", key="run_btn")
        progress_slot = st.empty()

    # ---- run action ----
    if run_clicked:
        params = _collect_params()
        try:
            model = _run_with_progress(params, progress_slot)
            ss["model_obj"] = model
            ss["model_df"] = model.get_model_dataframe()
            ss["agent_df"] = model.get_agent_dataframe()
            ss["cum_rates"] = cumulative_model_rates(
                list(model.agents), params["OBSERVED_CUM_MAX"])
            ss["final_positions"] = np.array(
                [[a.x, a.y, a.z, int(a.is_retrofitted), int(a.flood_count),
                  a.belief, a.pmt_threshold, int(a.neighborhood_id)]
                 for a in model.agents])
            ss["flood_history"] = list(model.flood_history)
            ss["run_params"] = params
            ss["has_run"] = True
            # Request navigation to Results on the next run (applied before the
            # widget is created), never by mutating the widget key here.
            ss["pending_nav"] = "\U0001F4CA  Results"
            st.rerun()
        except Exception as e:
            import traceback
            st.error("The simulation failed. Details below.")
            st.exception(e); st.code(traceback.format_exc())

    # ---- page routing ----
    if nav.strip().startswith("\U0001F4D8"):
        _page_documentation()
    elif nav.strip().startswith("\u2699"):
        _page_settings()
    else:
        _page_results()


def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    import streamlit as st
    try:
        _run_app()
    except Exception as _e:
        import traceback
        st.error("The app hit an error while starting. Full details below.")
        st.exception(_e); st.code(traceback.format_exc())
elif __name__ == "__main__":
    m = FloodAdaptationModel(dict(DEFAULTS), seed=42)
    m.run()
    rates = cumulative_model_rates(list(m.agents), DEFAULTS["OBSERVED_CUM_MAX"])
    print("cumulative model rates 0/<=4/5+:", ["%.1f" % r for r in rates])
    print("final pct retrofitted: %.1f%%" %
          m.get_model_dataframe()["pct_retrofitted"].iloc[-1])
