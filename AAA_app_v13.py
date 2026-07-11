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
        super().__init__(seed=seed if seed is not None else params["RANDOM_SEED"])
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
        for agent in self.agents:
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
# STREAMLIT APPLICATION
# ============================================================================

def _run_app():
    import streamlit as st

    st.set_page_config(page_title="Flood Adaptation ABM v13",
                       page_icon="\U0001F30A", layout="wide",
                       initial_sidebar_state="expanded")

    # ---- palette ----
    CLR_PRIMARY = "#1B4F72"; CLR_SECONDARY = "#2E86C1"; CLR_ACCENT = "#E67E22"
    CLR_MODEL = "#2E86C1"; CLR_OBS = "#E67E22"; CLR_RETRO = "#27AE60"
    CLR_NOT = "#D5D8DC"; CLR_BELIEF = "#1B4F72"; CLR_THRESH = "#C0392B"

    st.markdown("""<style>
      .block-container { padding-top: 2rem; }
      .cfg-card { background: linear-gradient(135deg,#EAF2F9,#D6EAF8);
        border-radius: 10px; padding: 0.8rem 1rem; }
      .cfg-card .lab { color:#1C6DAF; font-size:0.82rem; font-weight:600; }
      .cfg-card .val { color:#1B2A3A; font-size:1.6rem; font-weight:800; }
    </style>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ sidebar
    sb = st.sidebar
    sb.title("\U0001F30A Flood Adaptation ABM")
    sb.caption("Bayesian Belief Updating \u2014 v13")
    sb.divider()

    D = DEFAULTS

    def numbox(label, key, step, fmt="%.2f", minv=0.0, maxv=None, help=None):
        return sb.number_input(label, value=float(D[key]), min_value=float(minv),
                               max_value=(float(maxv) if maxv is not None else None),
                               step=float(step), format=fmt, help=help)

    sb.header("\u2699\ufe0f General")
    time_steps = int(sb.number_input("Time Steps", value=D["TIME_STEPS"],
                                     min_value=10, max_value=10000, step=10))
    n_agents = int(sb.number_input("Number of Agents", value=D["N_AGENTS"],
                                   min_value=10, max_value=100000, step=10))
    random_seed = int(sb.number_input("Random Seed", value=D["RANDOM_SEED"],
                                      min_value=0, max_value=10_000_000, step=1))

    sb.divider()
    sb.header("\U0001F4D0 Belief")
    initial_belief = numbox("Initial Belief  \u2014  P(H\u2081)", "INITIAL_BELIEF",
                            0.01, "%.2f", 0.01, 0.99,
                            help="Prior probability that a household should retrofit.")

    sb.divider()
    sb.header("\U0001F30A Channel 1 \u2014 Flood Experience")
    lambda_flood = numbox("\u03bb_flood  (base, per flood)", "LAMBDA_FLOOD", 0.01,
                          "%.2f", 1.0, None,
                          help="Bayes factor per flood. Survey anchor 1.52.")
    lambda_severity = numbox("\u03bb_severity  (\u00d7 if expects rising damage)",
                             "LAMBDA_SEVERITY", 0.1, "%.2f", 1.0, None,
                             help="Perceived-severity multiplier, applied on a flood "
                                  "only for agents who expect rising flood damage. "
                                  "Survey anchor 2.40 (Rogers, 1975).")
    p_expect = numbox("Fraction expecting rising damage", "P_EXPECT_RISING_DAMAGE",
                      0.01, "%.2f", 0.0, 1.0,
                      help="Assigned once at t=0. Survey (never-flooded): 0.69.")

    sb.divider()
    sb.header("\U0001F465 Channel 2 \u2014 Proximity")
    lambda_social = numbox("\u03bb_social  (base, per retrofitted neighbor)",
                           "LAMBDA_SOCIAL", 0.01, "%.2f", 1.0, None,
                           help="Bayes factor for each newly-retrofitted connected "
                                "neighbor (Granovetter, 1978).")
    lambda_similarity = numbox("\u03bb_similarity  (\u00d7 if neighbor is similar)",
                               "LAMBDA_SIMILARITY", 0.1, "%.2f", 1.0, None,
                               help="Multiplier applied when the retrofitted neighbor "
                                    "is similar, i.e. Gower S(i,j) \u2265 threshold "
                                    "(Gower, 1971). 1.0 = no similarity effect.")
    sim_threshold = numbox("Similarity threshold  (S \u2265)", "SIM_THRESHOLD",
                           0.05, "%.2f", 0.0, 1.0,
                           help="A neighbor counts as similar when the Gower "
                                "similarity coefficient meets this threshold.")

    sb.divider()
    sb.header("\u2139\ufe0f Channel 3 \u2014 Information")
    lambda_info = numbox("\u03bb_info  (base, if trusted information)",
                         "LAMBDA_INFO", 0.01, "%.2f", 1.0, None,
                         help="One-time t=0 factor for agents with a trusted flood-"
                              "information source. Survey: weak alone (OR ~1.39).")
    lambda_forecast = numbox("\u03bb_forecast  (\u00d7 if forecast-preparer)",
                             "LAMBDA_FORECAST", 0.05, "%.2f", 1.0, None,
                             help="Multiplier for agents who also prepare on flood "
                                  "forecasts. Survey anchor ~3.2.")
    p_info = numbox("Fraction with trusted information", "P_TRUSTED_INFO",
                    0.01, "%.2f", 0.0, 1.0, help="Survey (never-flooded): 0.48.")
    p_forecast = numbox("Fraction preparing on forecasts", "P_FORECAST_PREP",
                        0.01, "%.2f", 0.0, 1.0, help="Survey (never-flooded): 0.65.")

    sb.divider()
    sb.header("\U0001F3AF PMT Threshold")
    pmt_mean = numbox("Threshold Mean", "PMT_THRESHOLD_MEAN", 0.01, "%.2f", 0.01, 0.99)
    het_on = sb.checkbox("Threshold Heterogeneity", value=D["ENABLE_THRESHOLD_HET"],
                         help="Draw individual thresholds from a clipped Normal.")
    if het_on:
        pmt_std = numbox("Std Dev", "PMT_THRESHOLD_STD", 0.01, "%.2f", 0.0, 0.5)
        pmt_low = numbox("Lower Bound", "PMT_THRESHOLD_LOW", 0.01, "%.2f", 0.01, 0.99)
        pmt_high = numbox("Upper Bound", "PMT_THRESHOLD_HIGH", 0.01, "%.2f", 0.01, 0.99)
    else:
        pmt_std, pmt_low, pmt_high = 0.0, pmt_mean, pmt_mean

    sb.divider()
    sb.header("\U0001F9EC Agent Attributes")
    enable_het = sb.checkbox("Attribute Heterogeneity", value=D["ENABLE_HETEROGENEITY"],
                             help="If off, all agents are identical (S=1 for all pairs).")
    n_attributes = int(sb.number_input("Attributes per Agent", value=D["N_ATTRIBUTES"],
                                       min_value=1, max_value=10, step=1))
    n_classes = int(sb.number_input("Classes per Attribute", value=D["N_CLASSES"],
                                    min_value=1, max_value=10, step=1))

    sb.divider()
    sb.header("\U0001F5FA\ufe0f Spatial & Network")
    grid_rows = int(sb.number_input("Grid Rows", value=D["GRID_ROWS"], min_value=1, max_value=10, step=1))
    grid_cols = int(sb.number_input("Grid Cols", value=D["GRID_COLS"], min_value=1, max_value=10, step=1))
    n_connectors = int(sb.number_input("Connectors", value=D["N_CONNECTORS"], min_value=0, max_value=10, step=1))
    slope = numbox("Elevation Slope", "SLOPE", 0.01, "%.2f", 0.01, 2.0)
    noise_factor = numbox("Elevation Noise", "NOISE_FACTOR", 0.01, "%.2f", 0.0, 1.0)
    dist_threshold = numbox("Distance Threshold", "DISTANCE_THRESHOLD", 0.01, "%.2f", 0.01, 0.5)
    dbscan_min = int(sb.number_input("DBSCAN Min Samples", value=D["DBSCAN_MIN_SAMPLES"],
                                     min_value=2, max_value=10, step=1))

    sb.divider()
    sb.header("\U0001F30A Flood (GEV)")
    rp_str = sb.text_input("Return Periods", ", ".join(str(x) for x in D["RETURN_PERIODS"]))
    fl_str = sb.text_input("Flood Levels", ", ".join(str(x) for x in D["FLOOD_LEVELS"]))
    return_periods = [int(x.strip()) for x in rp_str.split(",")]
    flood_levels = [float(x.strip()) for x in fl_str.split(",")]

    # -------------------------------------------------------- assemble params
    params = dict(D)
    params.update(
        TIME_STEPS=time_steps, RANDOM_SEED=random_seed, N_AGENTS=n_agents,
        GRID_ROWS=grid_rows, GRID_COLS=grid_cols, N_CONNECTORS=n_connectors,
        SLOPE=slope, NOISE_FACTOR=noise_factor,
        ENABLE_HETEROGENEITY=enable_het, N_ATTRIBUTES=n_attributes, N_CLASSES=n_classes,
        DISTANCE_THRESHOLD=dist_threshold, DBSCAN_MIN_SAMPLES=dbscan_min,
        INITIAL_BELIEF=initial_belief,
        LAMBDA_FLOOD=lambda_flood, LAMBDA_SEVERITY=lambda_severity,
        P_EXPECT_RISING_DAMAGE=p_expect,
        LAMBDA_SOCIAL=lambda_social, LAMBDA_SIMILARITY=lambda_similarity,
        SIM_THRESHOLD=sim_threshold,
        LAMBDA_INFO=lambda_info, LAMBDA_FORECAST=lambda_forecast,
        P_TRUSTED_INFO=p_info, P_FORECAST_PREP=p_forecast,
        PMT_THRESHOLD_MEAN=pmt_mean, PMT_THRESHOLD_STD=pmt_std,
        PMT_THRESHOLD_LOW=pmt_low, PMT_THRESHOLD_HIGH=pmt_high,
        ENABLE_THRESHOLD_HET=het_on,
        RETURN_PERIODS=return_periods, FLOOD_LEVELS=flood_levels)

    # --------------------------------------------------------------- main tabs
    tab_run, tab_results, tab_doc = st.tabs(
        ["\u25B6 Run Simulation", "\U0001F4CA Results", "\U0001F4D8 Documentation"])

    # ---- Run tab: configuration cards ----
    with tab_run:
        st.markdown("## \u25B6 Run Simulation")
        st.markdown("### Current Configuration")

        def card(col, label, value):
            col.markdown(f"<div class='cfg-card'><div class='lab'>{label}</div>"
                         f"<div class='val'>{value}</div></div>", unsafe_allow_html=True)

        r1 = st.columns(4)
        card(r1[0], "Agents", n_agents)
        card(r1[1], "Time Steps", time_steps)
        card(r1[2], "Initial Belief  \u2014  P(H\u2081)", f"{initial_belief:.2f}")
        card(r1[3], "PMT Threshold", f"{pmt_mean:.2f}")

        st.markdown("#### Channel factors  (base  \u00d7  conditional multiplier)")
        r2 = st.columns(3)
        card(r2[0], "1 \u00b7 Flood Experience  \u2014  \u03bb_flood \u00d7 \u03bb_severity",
             f"{lambda_flood:.2f}  \u00d7  {lambda_severity:.2f}")
        card(r2[1], "2 \u00b7 Proximity  \u2014  \u03bb_social \u00d7 \u03bb_similarity",
             f"{lambda_social:.2f}  \u00d7  {lambda_similarity:.2f}")
        card(r2[2], "3 \u00b7 Information  \u2014  \u03bb_info \u00d7 \u03bb_forecast",
             f"{lambda_info:.2f}  \u00d7  {lambda_forecast:.2f}")

        st.caption("Each channel multiplies belief-odds by its base factor; the "
                   "conditional multiplier applies only when its trigger is present "
                   "(a flood hits and the agent expects rising damage; a similar "
                   "neighbor has retrofitted; the agent prepares on forecasts). "
                   "The multiplier is 1 \u2014 no effect \u2014 whenever the base trigger is "
                   "absent.")

        run_clicked = st.button("\u25B6  Run Simulation", type="primary")

        if run_clicked:
            with st.spinner("Running simulation\u2026"):
                model = FloodAdaptationModel(params, seed=random_seed)
                model.run()
            st.session_state["model_df"] = model.get_model_dataframe()
            st.session_state["agent_df"] = model.get_agent_dataframe()
            st.session_state["cum_rates"] = cumulative_model_rates(
                list(model.agents), params["OBSERVED_CUM_MAX"])
            st.session_state["final_positions"] = np.array(
                [[a.x, a.y, a.z, int(a.is_retrofitted)] for a in model.agents])
            st.session_state["has_run"] = True
            st.success("Simulation complete. Open the Results tab.")

    # ---- Results tab ----
    with tab_results:
        st.markdown("## \U0001F4CA Results")
        if not st.session_state.get("has_run"):
            st.info("Run a simulation first.")
        else:
            mdf = st.session_state["model_df"]
            cum = st.session_state["cum_rates"]
            obs = params["OBSERVED_CUM_RATES"]
            labels = params["OBSERVED_CUM_LABELS"]

            c1, c2 = st.columns(2)

            # (a) adoption curve
            with c1:
                st.markdown("#### (a)  Adoption over time")
                fig, ax = plt.subplots(figsize=(6, 4.2))
                ax.plot(mdf.index, mdf["pct_retrofitted"], color=CLR_PRIMARY, lw=2)
                ax.fill_between(mdf.index, mdf["pct_retrofitted"],
                                color=CLR_SECONDARY, alpha=0.15)
                ax.set(xlabel="Time step", ylabel="Retrofitted (%)",
                       xlim=(1, len(mdf)), ylim=(0, 100))
                ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)

            # (b) cumulative comparison vs observed
            with c2:
                st.markdown("#### (b)  Cumulative retrofit rate vs observed")
                fig, ax = plt.subplots(figsize=(6, 4.2))
                x = np.arange(len(labels)); w = 0.38
                bm = ax.bar(x - w / 2, cum, w, label="Model", color=CLR_MODEL,
                            edgecolor="black")
                bo = ax.bar(x + w / 2, obs, w, label="Observed", color=CLR_OBS,
                            edgecolor="black")
                for bars in (bm, bo):
                    for bar in bars:
                        h = bar.get_height()
                        if h > 0:
                            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                                    f"{h:.0f}", ha="center", va="bottom",
                                    fontsize=9, fontweight="bold")
                ax.set(xlabel="Flood experience (cumulative, at most k)",
                       ylabel="Retrofit rate (%)",
                       ylim=(0, max(max(cum), max(obs)) * 1.25 + 1))
                ax.set_xticks(x)
                ax.set_xticklabels(["0", "\u22644", "5+ (all)"])
                ax.legend(); ax.grid(alpha=0.3, axis="y")
                st.pyplot(fig); plt.close(fig)

            st.markdown("#### Cumulative rates")
            st.dataframe(pd.DataFrame({
                "Bin (at most k floods)": ["0", "\u2264 4", "5+ (all)"],
                "Model (%)": [f"{r:.1f}" for r in cum],
                "Observed (%)": [f"{o:.1f}" for o in obs]}),
                hide_index=True, use_container_width=True)

            # spatial retrofit map
            st.markdown("#### Spatial retrofit map (final step)")
            pos = st.session_state["final_positions"]
            fig, ax = plt.subplots(figsize=(7, 5.5))
            retro = pos[:, 3] == 1
            ax.scatter(pos[~retro, 0], pos[~retro, 1], c=CLR_NOT, s=30,
                       edgecolors="gray", linewidths=0.4, label="Not retrofitted")
            ax.scatter(pos[retro, 0], pos[retro, 1], c=CLR_RETRO, s=40,
                       edgecolors="black", linewidths=0.4, label="Retrofitted")
            ax.set(xlabel="x", ylabel="y (elevation \u2192)", xlim=(0, 1), ylim=(0, 1))
            ax.legend(loc="upper right"); ax.set_aspect("equal")
            st.pyplot(fig); plt.close(fig)

    # ---- Documentation tab ----
    with tab_doc:
        st.markdown("## \U0001F4D8 Documentation")
        st.markdown(r"""
This model represents household flood-retrofit decisions as Bayesian belief
updating in **odds form** combined with **Protection Motivation Theory**.
Each household holds a belief $P(H_1)$ that it should retrofit, updated by three
evidence channels, and retrofits once belief crosses its PMT threshold.
Retrofitting is **absorbing** (a retrofitted home takes no further damage).

**Belief update (odds form).** For every piece of evidence,
$\text{posterior odds} = \text{prior odds} \times \text{Bayes factor}$
(Jaynes, 2003; Kass & Raftery, 1995).

**Three channels \u2014 each a base factor \u00d7 a conditional multiplier** (the
multiplier is 1 whenever its trigger is absent):

1. **Flood experience.** On a flood year ($\text{flood level} > z$),
   $\text{odds} \times= \lambda_{\text{flood}} \cdot m$, where
   $m = \lambda_{\text{severity}}$ for agents who expect rising flood damage
   (perceived-severity appraisal; Rogers, 1975; Floyd et al., 2000) and $m=1$
   otherwise. Safe years produce no update (Tversky & Kahneman, 1974).

2. **Proximity.** For each newly-retrofitted connected neighbor,
   $\text{odds} \times= \lambda_{\text{social}} \cdot m$, where
   $m = \lambda_{\text{similarity}}$ when that neighbor is similar
   ($S(i,j) \ge$ threshold; Gower, 1971) and $m=1$ otherwise. With no
   retrofitted neighbor the channel is inert.

3. **Information.** Once at $t=0$, agents with a trusted flood-information
   source apply $\text{odds} \times= \lambda_{\text{info}} \cdot m$, where
   $m = \lambda_{\text{forecast}}$ for those who also prepare on flood
   forecasts and $m=1$ otherwise.

**Static traits** (expects-rising-damage, trusted-information,
forecast-preparation) are drawn once at $t=0$ from survey-anchored fractions
and never change.

**Decision.** A household retrofits when $P(H_1) \ge \theta_i$, its individual
PMT threshold (Rogers, 1975), heterogeneous via a clipped Normal.

**Comparison target.** Results plot (b) compares the model's **cumulative**
"at most $k$ floods" retrofit rate against the survey values
**18.0 / 22.3 / 27.4 %** for the 0 / $\le 4$ / all bins.
""")


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _srr
        _in_streamlit = _srr.get_script_run_ctx() is not None
    except Exception:
        _in_streamlit = False
    if _in_streamlit:
        _run_app()
    else:
        # headless smoke test
        m = FloodAdaptationModel(dict(DEFAULTS), seed=42)
        m.run()
        rates = cumulative_model_rates(list(m.agents), DEFAULTS["OBSERVED_CUM_MAX"])
        print("cumulative model rates 0/<=4/5+:", ["%.1f" % r for r in rates])
        print("final pct retrofitted: %.1f%%" %
              m.get_model_dataframe()["pct_retrofitted"].iloc[-1])
