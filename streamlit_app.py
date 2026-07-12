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
       multiplier: lambda_risk_perception  active for agents who expect
                   rising flood damage (risk-perception appraisal, a
                   forward-looking threat appraisal in the sense of
                   Rogers, 1975; Floyd, Prentice-Dunn & Rogers, 2000).
                   = 1 otherwise.
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
       multiplier: lambda_response      active when the agent also takes
                   precautionary action based on the forecast data.
                   = 1 otherwise.
       Fires ONCE, at initialization (a static informational prior), because
       trusted-info and forecast-preparation are stable household traits.

Survey-anchored defaults (NYC Flood Vulnerability Survey, cleaned):
  lambda_flood      1.52  (owner per-flood odds ratio)
  lambda_risk_perception  2.40  (expecting- vs not-expecting rising damage)
  lambda_response   3.20  (precautionary-action-on-forecast odds ratio)
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
    TIME_STEPS=75, RANDOM_SEED=42,
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
    # odds ratio), LAMBDA_RISK_PERCEPTION 2.40 (expecting vs not-expecting rising
    # damage).  Opened slightly de-escalated so the model is non-saturated out
    # of the box; tune upward toward the anchors and watch the cumulative bars.
    LAMBDA_FLOOD=1.52, LAMBDA_RISK_PERCEPTION=1.60,
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
    LAMBDA_INFO=1.05, LAMBDA_RESPONSE=1.15,
    P_TRUSTED_INFO=0.48, P_FORECAST_PREP=0.65,
    # PMT threshold
    # PMT threshold.  When heterogeneity is ON, individual thresholds are drawn
    # from Uniform(LOW, HIGH); when OFF, every household uses MEAN (a single
    # point value).
    PMT_THRESHOLD_MEAN=0.85,
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


class _FixedFloodSeries:
    """Replays a user-supplied flood series (Case Study Mode); 0 past the end."""

    def __init__(self, levels):
        self._levels = list(levels)
        self._i = 0

    def sample(self):
        if self._i < len(self._levels):
            v = self._levels[self._i]
            self._i += 1
            return float(np.clip(v, 0.0, 1.0))
        return 0.0


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
        lambda_flood, times the risk-perception multiplier
        lambda_risk_perception for agents who expect rising flood damage
        (= 1 otherwise).
        """
        if self.is_retrofitted:
            return False
        if flood_level > self.z:
            self.flood_count += 1
            m = self.model
            mult = m.LAMBDA_RISK_PERCEPTION if self.expects_rising_damage else 1.0
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
        agents with a trusted flood-information source, times the response
        multiplier lambda_response for those who also take precautionary
        action based on the forecast data.
        Agents without trusted information receive no update.
        """
        if not self.has_trusted_info:
            return
        m = self.model
        mult = m.LAMBDA_RESPONSE if self.forecast_prep else 1.0
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
        # Optional custom flood series (Case Study Mode): replaces the GEV
        # sampler with a fixed, user-supplied sequence of flood levels.
        custom_flood = getattr(self, "CUSTOM_FLOOD_SERIES", None)
        if custom_flood is not None and len(custom_flood) > 0:
            self.flood_generator = _FixedFloodSeries(custom_flood)
        else:
            self.flood_generator = FloodGenerator(
                self.RETURN_PERIODS, self.FLOOD_LEVELS, self.rng)

        # Positions & elevations. Case Study Mode supplies them directly from an
        # uploaded CSV (columns x, y, z); otherwise they are generated on a grid.
        custom_pos = getattr(self, "CUSTOM_POSITIONS", None)
        custom_elev = getattr(self, "CUSTOM_ELEVATIONS", None)
        if custom_pos is not None and custom_elev is not None:
            self.positions = np.asarray(custom_pos, dtype=float)
            self.elevations = np.asarray(custom_elev, dtype=float)
        else:
            self.positions, self.elevations = generate_spatial(
                self.n_agents, self.DISTANCE_THRESHOLD, self.GRID_ROWS,
                self.GRID_COLS, self.N_CONNECTORS, self.SLOPE,
                self.NOISE_FACTOR, self.rng)
        self.n_agents = len(self.positions)   # grid or CSV sets the final count

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
            if self.ENABLE_THRESHOLD_HET and self.PMT_THRESHOLD_HIGH > self.PMT_THRESHOLD_LOW:
                # Heterogeneous thresholds ~ Uniform(low, high).
                thr = float(self.rng.uniform(self.PMT_THRESHOLD_LOW,
                                             self.PMT_THRESHOLD_HIGH))
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
      .block-container {{ padding-top: 2.6rem; }}
      section.main > div {{ padding-top: 0.5rem; }}
      h2 {{ font-size: 1.75rem !important; line-height: 1.4 !important;
            padding-top: 0.35em !important; margin-top: 0.2rem !important; }}
      h3 {{ font-size: 1.4rem !important; line-height: 1.4 !important;
            padding-top: 0.2em !important; }}
      h2, h3 {{ overflow: visible !important; }}
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
      /* Hide the radio glyph/circle entirely, across all Streamlit builds,
         without ever hiding the label text. Strategy: the text always lives
         in a stMarkdownContainer; the glyph is everything else in the label
         (an <svg>, a baseweb radio wrapper, or a bare styled <div>). We hide
         the raw input, any baseweb radio, any svg, and any direct-child div
         of the label that does NOT contain the markdown text. */
      section[data-testid="stSidebar"] [role="radiogroup"] input {{ display:none !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] label svg {{ display:none !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] label [data-baseweb="radio"] {{
          display:none !important; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label > div:not(:has([data-testid="stMarkdownContainer"])) {{
          display:none !important; width:0 !important; height:0 !important;
          min-width:0 !important; margin:0 !important; padding:0 !important;
          border:0 !important; }}
      /* the first baseweb layer inside the label often carries a fixed-width
         slot for the glyph; collapse it but keep the text child visible. */
      section[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{
          min-width:0 !important; }}
      /* always keep the text container visible. */
      section[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {{
          display:block !important; visibility:visible !important; opacity:1 !important; }}
      /* Force the label text visible and white. */
      section[data-testid="stSidebar"] [role="radiogroup"] > label,
      section[data-testid="stSidebar"] [role="radiogroup"] > label div,
      section[data-testid="stSidebar"] [role="radiogroup"] > label p,
      section[data-testid="stSidebar"] [role="radiogroup"] > label span {{
          color:#fff !important; font-weight:600; font-size:0.95rem; }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
          background: rgba(148,163,184,0.16); transform: translateX(3px); }}
      section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
          background: linear-gradient(135deg, {CLR_SKY} 0%, {CLR_SKY_DK} 100%);
          box-shadow: 0 6px 16px rgba(14,165,233,0.40); transform: translateX(3px); }}
      /* sidebar buttons: PRIMARY (selected mode / run) = blue,
         SECONDARY (unselected mode) = muted dark slate. */
      section[data-testid="stSidebar"] .stButton > button {{
          color:#fff !important; font-weight:700; border:0; border-radius:10px;
          padding: 0.55rem 0.8rem; width:100%; }}
      section[data-testid="stSidebar"] .stButton > button[kind="primary"],
      section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {{
          background: linear-gradient(135deg, {CLR_SKY} 0%, {CLR_SKY_DK} 100%);
          box-shadow: 0 6px 16px rgba(14,165,233,0.35); }}
      section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
      section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"] {{
          background: rgba(148,163,184,0.16) !important;
          color:{CLR_SLATE300} !important; box-shadow:none; }}
      section[data-testid="stSidebar"] .stButton > button:hover {{ filter: brightness(1.1); }}
      section[data-testid="stSidebar"] .rail-label {{ color:{CLR_SLATE300};
          font-size:0.72rem; font-weight:700; letter-spacing:0.6px;
          text-transform:uppercase; margin: 0.2rem 0 0.3rem 0.15rem; }}
      section[data-testid="stSidebar"] hr {{ border-color: rgba(148,163,184,0.20); margin: 0.7rem 0; }}
      /* white labels for the case-study file uploaders */
      section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
      section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p,
      section[data-testid="stSidebar"] [data-testid="stFileUploader"] label span {{
          color:#ffffff !important; font-weight:600 !important; }}
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
                'Flood Mitigation Agent-based Model</p>',
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
        LAMBDA_RISK_PERCEPTION=g("LAMBDA_RISK_PERCEPTION", D["LAMBDA_RISK_PERCEPTION"]),
        P_EXPECT_RISING_DAMAGE=g("P_EXPECT_RISING_DAMAGE", D["P_EXPECT_RISING_DAMAGE"]),
        LAMBDA_SOCIAL=g("LAMBDA_SOCIAL", D["LAMBDA_SOCIAL"]),
        LAMBDA_SIMILARITY=g("LAMBDA_SIMILARITY", D["LAMBDA_SIMILARITY"]),
        SIM_THRESHOLD=g("SIM_THRESHOLD", D["SIM_THRESHOLD"]),
        LAMBDA_INFO=g("LAMBDA_INFO", D["LAMBDA_INFO"]),
        LAMBDA_RESPONSE=g("LAMBDA_RESPONSE", D["LAMBDA_RESPONSE"]),
        P_TRUSTED_INFO=g("P_TRUSTED_INFO", D["P_TRUSTED_INFO"]),
        P_FORECAST_PREP=g("P_FORECAST_PREP", D["P_FORECAST_PREP"]),
        PMT_THRESHOLD_MEAN=g("PMT_THRESHOLD_MEAN", D["PMT_THRESHOLD_MEAN"]),
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
    het_on = st.checkbox(
        "Threshold Heterogeneity  (draw individual \u03b8 from Uniform[min, max])",
        value=D["ENABLE_THRESHOLD_HET"], key="p_ENABLE_THRESHOLD_HET")
    if het_on:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            nb("Initial Belief  P(H\u2081)", "INITIAL_BELIEF", 0.01, "%.2f", 0.01, 0.99,
               help="Prior probability that a household should retrofit.")
        with bc2:
            nb("\u03b8 Minimum", "PMT_THRESHOLD_LOW", 0.01, "%.2f", 0.01, 0.99,
               help="Lower bound of the uniform threshold distribution.")
        with bc3:
            nb("\u03b8 Maximum", "PMT_THRESHOLD_HIGH", 0.01, "%.2f", 0.01, 0.99,
               help="Upper bound of the uniform threshold distribution.")
    else:
        bc1, bc2 = st.columns(2)
        with bc1:
            nb("Initial Belief  P(H\u2081)", "INITIAL_BELIEF", 0.01, "%.2f", 0.01, 0.99,
               help="Prior probability that a household should retrofit.")
        with bc2:
            nb("Decision Threshold  \u03b8", "PMT_THRESHOLD_MEAN", 0.01, "%.2f", 0.01, 0.99,
               help="Single belief level at which every household acts [14].")

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        _sec("Channel 1 \u00b7 Flood Experience",
             "Belief update per flood, amplified by expecting rising risk.", C_CH1)
        nb("\u03bb_flood  (base, per flood)", "LAMBDA_FLOOD", 0.01, "%.2f", 1.0, None,
           help="Bayes factor per flood. Survey anchor 1.52.")
        nb("\u03bb_risk_perception  (\u00d7 rising damage)", "LAMBDA_RISK_PERCEPTION", 0.1, "%.2f", 1.0, None,
           help="Risk-perception multiplier on a flood, for agents expecting "
                "rising flood damage. Survey anchor 2.40 [14].")
        nb("Fraction expecting rising damage", "P_EXPECT_RISING_DAMAGE",
           0.01, "%.2f", 0.0, 1.0, help="Assigned once at t=0. Survey: 0.69.")
    with ch2:
        _sec("Channel 2 \u00b7 Proximity",
             "Social learning from retrofitted neighbours, stronger if similar.", C_CH2)
        nb("\u03bb_social  (base, per neighbor)", "LAMBDA_SOCIAL", 0.01, "%.2f", 1.0, None,
           help="Bayes factor per newly-retrofitted connected neighbor [7].")
        nb("\u03bb_similarity  (\u00d7 if similar)", "LAMBDA_SIMILARITY", 0.1, "%.2f", 1.0, None,
           help="Applied when the retrofitted neighbor is similar "
                "(Gower S \u2265 threshold). 1.0 = no similarity effect.")
        nb("Similarity threshold  (S \u2265)", "SIM_THRESHOLD", 0.05, "%.2f", 0.0, 1.0,
           help="A neighbor counts as similar at or above this Gower similarity.")
    with ch3:
        _sec("Channel 3 \u00b7 Information",
             "One-time t=0 prior from trusted information and forecast info.", C_CH3)
        nb("\u03bb_info  (base, if trusted info)", "LAMBDA_INFO", 0.01, "%.2f", 1.0, None,
           help="One-time t=0 factor for agents with a trusted source. "
                "Survey: weak alone (OR ~1.39).")
        nb("\u03bb_response  (\u00d7 if forecast info)", "LAMBDA_RESPONSE", 0.05, "%.2f", 1.0, None,
           help="Amplifier for agents who take precautionary action based on the forecast data. Survey ~3.2.")
        nb("Fraction with trusted information", "P_TRUSTED_INFO", 0.01, "%.2f", 0.0, 1.0,
           help="Survey (never-flooded): 0.48.")
        nb("Fraction using forecast info", "P_FORECAST_PREP", 0.01, "%.2f", 0.0, 1.0,
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
        ax.scatter([a.x for a in na], [a.y for a in na], c="#e2e8f0", s=170,
                   edgecolor="black", linewidth=0.8, zorder=2)
    if ad:
        sc = ax.scatter([a.x for a in ad], [a.y for a in ad],
                        c=[a.retrofit_step for a in ad], cmap="YlGn", s=185,
                        edgecolor="black", linewidth=0.8, zorder=3,
                        vmin=1, vmax=max(1, model.current_step))
        fig.colorbar(sc, ax=ax, shrink=0.7).set_label("Retrofit step")
    # number inside each node = that household's personal flood count
    for a in agents:
        ax.text(a.x, a.y, str(a.flood_count), ha="center", va="center",
                fontsize=6, fontweight="bold", zorder=4)
    ax.set(xlabel="x", ylabel="y", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(handles=[Patch(facecolor="#e2e8f0", edgecolor="black", label="Not retrofitted"),
                       Patch(facecolor="#31a354", edgecolor="black", label="Retrofitted")],
              loc="upper right", fontsize=8)
    ax.text(0.0, -0.13, "Number in each node = household\u2019s personal flood count.",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="#64748b")
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
    fig.tight_layout()
    return fig


def _config_chips(p):
    import streamlit as st
    chips = [
        ("Agents", p["N_AGENTS"]), ("Steps", p["TIME_STEPS"]),
        ("P(H\u2081)", f"{p['INITIAL_BELIEF']:.2f}"),
        ("\u03b8", f"{p['PMT_THRESHOLD_MEAN']:.2f}"),
        ("\u03bb_flood\u00d7\u03bb_riskperc", f"{p['LAMBDA_FLOOD']:.2f}\u00d7{p['LAMBDA_RISK_PERCEPTION']:.2f}"),
        ("\u03bb_social\u00d7\u03bb_sim", f"{p['LAMBDA_SOCIAL']:.2f}\u00d7{p['LAMBDA_SIMILARITY']:.2f}"),
        ("\u03bb_info\u00d7\u03bb_resp", f"{p['LAMBDA_INFO']:.2f}\u00d7{p['LAMBDA_RESPONSE']:.2f}"),
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
        "is processed in three algebraic steps ([10]; Kass & "
        "Raftery, 1995).")
    st.markdown("**Step 1 \u2014 Convert probability to odds.** Odds give a natural "
                "multiplicative framework for accumulating evidence:")
    st.latex(r"O_i = \frac{P_i(H_1)}{1 - P_i(H_1)}")
    st.markdown("For example, a belief of $P_i(H_1)=0.08$ corresponds to odds "
                "$O_i = 0.08/0.92 \\approx 0.087$ \u2014 the agent considers $H_0$ "
                "about eleven times more likely than $H_1$.")
    st.markdown("**Step 2 \u2014 Multiply the odds by a Bayes factor.** A Bayes "
                "factor $\\lambda$ is the likelihood ratio of a single "
                "observation [11]:")
    st.latex(r"\lambda = \frac{P(\text{evidence}\mid H_1)}{P(\text{evidence}\mid H_0)}"
             r"\qquad O_i^{\text{post}} = O_i^{\text{prior}} \times \lambda")
    st.markdown("A factor $\\lambda>1$ shifts belief toward retrofitting; "
                "$\\lambda=1$ is uninformative. When several channels fire in "
                "the same step their factors compose multiplicatively, and "
                "because multiplication is commutative the update is "
                "order-independent [5]:")
    st.latex(r"O_i^{\text{post}} = O_i^{\text{prior}}\times "
             r"\lambda_{\text{exp}}\times\lambda_{\text{prox}}\times\lambda_{\text{info}}")
    st.markdown("**Step 3 \u2014 Convert the posterior odds back to a probability:**")
    st.latex(r"P_i^{\text{post}}(H_1) = \frac{O_i^{\text{post}}}{1 + O_i^{\text{post}}}")
    st.markdown("This is algebraically identical to the standard form of Bayes\u2019 "
                "theorem; the advantage is that each evidence source is a single "
                "scalar and sequential updates are just repeated multiplication.")

    st.markdown("**Worked example.** Take a prior belief $P(H_1)=0.08$, so the "
                "prior odds are $O = 0.08/0.92 = 0.087$. Suppose a household with "
                "a trusted information source ($\\lambda_{info}=1.05$) who also "
                "takes precautionary action on the forecast "
                "($\\lambda_{response}=1.15$) then "
                "experiences two floods, and expects rising damage "
                "($\\lambda_{flood}=1.52$, $\\lambda_{\\mathrm{risk\\,perc}}=1.60$). The "
                "information channel fires once at $t=0$ and each flood fires "
                "when it occurs:")
    st.latex(r"O_{\text{final}} = 0.087 \times "
             r"\underbrace{(1.05\times1.15)}_{\text{information}} \times "
             r"\underbrace{(1.52\times1.60)}_{\text{flood 1}} \times "
             r"\underbrace{(1.52\times1.60)}_{\text{flood 2}} = 0.62")
    st.markdown("Converting back, $P_{\\text{final}}(H_1) = 0.62/(1+0.62) = "
                "0.38$. Belief has risen from 0.08 to 0.38 \u2014 still below a "
                "threshold of $\\theta=0.85$, so this household would not yet "
                "retrofit. It illustrates why several strong signals are "
                "typically needed to cross the bar, matching the low observed "
                "adoption in the survey.")

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
        "[2]. Agent $i$ is flooded when $f_t > z_i$. The update is "
        "asymmetric \u2014 flood years are psychologically salient while dry years "
        "are cognitively inert (availability heuristic [15]). "
        "On a flood, the base per-flood factor $\\lambda_{flood}$ is "
        "multiplied by a **risk-perception** multiplier $\\lambda_{\\mathrm{risk\\,perc}}$ "
        "for households that expect flood damage to worsen [14], "
        "[4]:")
    st.latex(r"""\lambda_{\text{exp},i}^{(t)} =
\begin{cases}
\lambda_{flood}\cdot\lambda_{\mathrm{risk\,perc}} & f_t > z_i \ \text{and agent expects rising damage}\\
\lambda_{flood} & f_t > z_i \ \text{and agent does not}\\
1 & f_t \le z_i \quad(\text{not flooded; no update})
\end{cases}""")
    st.markdown("A single per-flood factor is used (no separate first-flood "
                "term). Survey anchors: $\\lambda_{flood}=1.52$ (owner per-flood "
                "odds ratio) and $\\lambda_{\\mathrm{risk\\,perc}}=2.40$ (expecting- vs "
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
        "[7] is multiplied by a similarity multiplier "
        "$\\lambda_{similarity}$ when the neighbour is **similar**, i.e. their "
        "Gower similarity meets the threshold [6], [12]:")
    st.latex(r"""\lambda_{\text{prox},ij} =
\begin{cases}
\lambda_{social}\cdot\lambda_{similarity} & j\ \text{retrofitted and } S(i,j)\ge S^\ast\\
\lambda_{social} & j\ \text{retrofitted and } S(i,j)< S^\ast\\
1 & \text{no newly-retrofitted neighbour}
\end{cases}""")
    st.markdown("This channel is grounded in empirical evidence that flood "
                "adaptation spreads through social proximity: households are "
                "more likely to adopt protective measures when neighbours and "
                "friends have already done so [1], [13], and neighbourhood "
                "proximity and social networks measurably shape property-level "
                "adaptation in both surveys and agent-based models [9], [13]. "
                "The similarity multiplier reflects homophily \u2014 the tendency for "
                "influence to run more strongly between similar households [12] "
                "\u2014 and parallels the neighbour effect documented for other "
                "protective investments such as residential solar adoption [8]. "
                "Similarity is the fraction of discrete attributes on which two "
                "agents agree,")
    st.latex(r"S(i,j) = \frac{1}{A}\sum_{a=1}^{A}\mathbb{1}\!\left[x_{ia}=x_{ja}\right]"
             r"\ \in[0,1]\quad(\text{[6]})")
    st.markdown("so a retrofitting neighbour who resembles the agent carries "
                "more weight than a dissimilar one, and with no retrofitted "
                "neighbour the channel is inert.")

    st.markdown("#### 3.5 &nbsp; Channel 3: trusted information")
    st.markdown(
        "Applied **once, at initialisation**, because trusted-information and "
        "forecast use are stable household traits rather than repeated "
        "events. Households with a trusted flood-information source apply a base "
        "factor $\\lambda_{info}$, multiplied by a response multiplier "
        "$\\lambda_{response}$ for those who also take **precautionary action "
        "based on the forecast data**:")
    st.latex(r"""\lambda_{\text{info},i} =
\begin{cases}
\lambda_{info}\cdot\lambda_{response} & \text{trusted info and precautionary action on forecast}\\
\lambda_{info} & \text{trusted info only}\\
1 & \text{no trusted information}
\end{cases}""")
    st.markdown("In the survey, having a trusted source alone is weak "
                "(odds ratio \u2248 1.39, n.s.) while taking precautionary action "
                "based on the forecast data is the stronger amplifier "
                "(odds ratio \u2248 3.2), which is why the base "
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
        "threshold, $P_i(H_1)\\ge\\theta_i$ [14]. When threshold heterogeneity "
        "is enabled, individual thresholds are drawn from a **uniform "
        "distribution** on a user-set band $[\\theta_{min},\\theta_{max}]$; "
        "otherwise every household shares a single threshold. Belief is "
        "homogeneous at $t=0$. Because only the gap between belief and threshold "
        "governs behaviour, a single heterogeneous quantity (the threshold) is "
        "sufficient and identifiable \u2014 belief and threshold spreads are not "
        "separately estimated.")
    st.latex(r"\text{retrofit}_i \iff P_i(H_1)\ge\theta_i,\qquad "
             r"\theta_i\sim\mathcal{U}(\theta_{min},\theta_{max})")

    # ---- 6 Understanding the Bayes factors ----
    st.markdown('<div class="doc-h">6 &nbsp; Understanding the Bayes factors</div>',
                unsafe_allow_html=True)
    st.markdown(
        "A Bayes factor greater than 1 is evidence for retrofitting and "
        "multiplies the odds. Two floods contribute $\\lambda_{flood}^2$; a "
        "flood experienced by a household expecting rising damage contributes "
        "$\\lambda_{flood}\\,\\lambda_{\\mathrm{risk\\,perc}}$. Working in odds means these "
        "combine by multiplication and the update is order-independent \u2014 the "
        "same evidence yields the same belief regardless of the order in which "
        "it arrives [5].")

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
        "$\\lambda_{flood}=1.52$, $\\lambda_{\\mathrm{risk\\,perc}}=2.40$, "
        "$\\lambda_{response}\\approx3.2$; trait fractions 0.69 / 0.48 / 0.65. "
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

    # ---- 12 Empirical grounding ----
    st.markdown('<div class="doc-h">12 &nbsp; Empirical grounding</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Every mechanism is anchored in the NYC Flood Vulnerability Index "
        "Survey (Rockaway peninsula). Key findings that shape the model:\n"
        "- **Flood experience is the dominant driver.** Retrofit rate rises "
        "with the number of floods a household has experienced (roughly 18% at "
        "zero floods to 27% among the most-flooded), while never-flooded "
        "households retrofit at low rates \u2014 evidence against social contagion "
        "as the primary mechanism.\n"
        "- **Risk perception matters.** Households expecting rising flood "
        "damage retrofit more, motivating the $\\lambda_{\\mathrm{risk\\,perc}}$ multiplier "
        "on the experience channel.\n"
        "- **Trusted information and precautionary action raise adoption.** "
        "Households with a trusted information source (35% vs 21%) and those "
        "taking precautionary action based on the forecast data (31% vs 17%) "
        "retrofit more, motivating the "
        "information channel; institutional/government sources show the "
        "strongest association.\n"
        "- **No income gradient.** Flood exposure and damage do not vary with "
        "household income, so income is deliberately excluded.\n"
        "- **Ownership gates action.** Almost all retrofitters are owners; "
        "renters act rarely and only after long delays.")

    # ---- 13 Assumptions & limitations ----
    st.markdown('<div class="doc-h">13 &nbsp; Assumptions & limitations</div>',
                unsafe_allow_html=True)
    st.markdown(
        "- **Homogeneous prior belief.** All households begin with the same "
        "$P(H_1)$; only the threshold is heterogeneous. This is a modelling "
        "choice for identifiability, not an empirical claim.\n"
        "- **Static traits.** Expects-rising-damage, trusted-information, and "
        "forecast-preparation are fixed at $t=0$; the model does not let "
        "information spread or beliefs about severity evolve endogenously.\n"
        "- **One-shot information channel.** Trusted information applies once, "
        "as a prior, rather than as repeated exposure.\n"
        "- **Absorbing retrofit.** Households never de-adapt or move.\n"
        "- **Cross-sectional calibration.** Survey odds ratios are associations "
        "from a single snapshot; the model uses them as anchors, not as "
        "identified causal effects, and per-channel factors are not separately "
        "identified from the survey alone.\n"
        "- **Synthetic geography in Research Mode.** Positions and elevations "
        "are generated on a grid; Case Study Mode replaces them with real "
        "uploaded coordinates and an observed flood series.")

    # ---- 14 Research vs Case Study mode ----
    st.markdown('<div class="doc-h">14 &nbsp; Research mode vs case-study mode</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**Research mode** generates a synthetic settlement: household "
        "positions on a connected grid, elevations from a linear coastal "
        "gradient with noise, and annual floods sampled from a GEV distribution "
        "fitted to the return periods and levels on the Settings page. Use it "
        "for controlled experiments and sensitivity analysis.\n\n"
        "**Case-study mode** replaces the synthetic setting with real data: "
        "upload a **location CSV** (columns `x, y, z` \u2014 one row per household, "
        "giving position and elevation) and a **flood-series CSV** (column "
        "`flood_level` \u2014 one value per time step, replayed in order). The agent "
        "count is taken from the location file; every behavioural parameter "
        "still comes from the Settings page, so the same calibrated mechanism "
        "can be driven by an observed landscape and flood history.")

    # ---- 15 References ----
    st.markdown('<div class="doc-h">15 &nbsp; References</div>', unsafe_allow_html=True)
    st.markdown(
        "References are numbered and cited as [X] throughout. Formatted in APA "
        "style.\n\n"
        "[1] Bubeck, P., Botzen, W. J. W., Kreibich, H., & Aerts, J. C. J. H. "
        "(2013). Detailed insights into the influence of flood-coping appraisals "
        "on mitigation behaviour. *Global Environmental Change, 23*(5), "
        "1327\u20131338. https://doi.org/10.1016/j.gloenvcha.2013.05.009\n\n"
        "[2] Coles, S. (2001). *An introduction to statistical modeling of "
        "extreme values.* Springer.\n\n"
        "[3] Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A "
        "density-based algorithm for discovering clusters in large spatial "
        "databases with noise. In *Proceedings of the Second International "
        "Conference on Knowledge Discovery and Data Mining (KDD-96)* "
        "(pp. 226\u2013231). AAAI Press.\n\n"
        "[4] Floyd, D. L., Prentice-Dunn, S., & Rogers, R. W. (2000). A "
        "meta-analysis of research on protection motivation theory. *Journal of "
        "Applied Social Psychology, 30*(2), 407\u2013429. "
        "https://doi.org/10.1111/j.1559-1816.2000.tb02323.x\n\n"
        "[5] Good, I. J. (1950). *Probability and the weighing of evidence.* "
        "Charles Griffin.\n\n"
        "[6] Gower, J. C. (1971). A general coefficient of similarity and some "
        "of its properties. *Biometrics, 27*(4), 857\u2013871. "
        "https://doi.org/10.2307/2528823\n\n"
        "[7] Granovetter, M. (1978). Threshold models of collective behavior. "
        "*American Journal of Sociology, 83*(6), 1420\u20131443. "
        "https://doi.org/10.1086/226707\n\n"
        "[8] Graziano, M., & Gillingham, K. (2015). Spatial patterns of solar "
        "photovoltaic system adoption: The influence of neighbors and the "
        "built environment. *Journal of Economic Geography, 15*(4), 815\u2013839. "
        "https://doi.org/10.1093/jeg/lbu036\n\n"
        "[9] Haer, T., Botzen, W. J. W., & Aerts, J. C. J. H. (2016). The "
        "effectiveness of flood risk communication strategies and the influence "
        "of social networks\u2014Insights from an agent-based model. *Environmental "
        "Science & Policy, 60*, 44\u201352. "
        "https://doi.org/10.1016/j.envsci.2016.03.006\n\n"
        "[10] Jaynes, E. T. (2003). *Probability theory: The logic of science.* "
        "Cambridge University Press.\n\n"
        "[11] Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of "
        "the American Statistical Association, 90*(430), 773\u2013795. "
        "https://doi.org/10.1080/01621459.1995.10476572\n\n"
        "[12] McPherson, M., Smith-Lovin, L., & Cook, J. M. (2001). Birds of a "
        "feather: Homophily in social networks. *Annual Review of Sociology, "
        "27*, 415\u2013444. https://doi.org/10.1146/annurev.soc.27.1.415\n\n"
        "[13] Poussin, J. K., Botzen, W. J. W., & Aerts, J. C. J. H. (2014). "
        "Factors of influence on flood damage mitigation behaviour by "
        "households. *Environmental Science & Policy, 40*, 69\u201377. "
        "https://doi.org/10.1016/j.envsci.2014.01.013\n\n"
        "[14] Rogers, R. W. (1975). A protection motivation theory of fear "
        "appeals and attitude change. *The Journal of Psychology, 91*(1), "
        "93\u2013114. https://doi.org/10.1080/00223980.1975.9915803\n\n"
        "[15] Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: "
        "Heuristics and biases. *Science, 185*(4157), 1124\u20131131. "
        "https://doi.org/10.1126/science.185.4157.1124")



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _workflow_svg():
    """Comprehensive SVG flowchart with professional edge-anchored arrows.

    Geometry system (like a real flowchart tool):
      * Boxes are named rectangles stored in B[name] = (x, y, w, h).
      * port(name, side) returns the CENTER of that box edge
        (side in 'top','bottom','left','right'), so every arrow starts and
        ends at the mid-point of a box side and stops exactly on the border.
      * Arrows are orthogonal polylines between two ports; the marker tip
        lands on the border with no overshoot and no gap.

    Layout invariants (do not change):
      * Channels 1, 2, 3 share one horizontal row.
      * Channels 1 and 2 are INSIDE the dashed 'each time step' loop;
        Channel 3 is OUTSIDE it, to the right.
    """
    C1, C2, C3 = "#0ea5e9", "#22c55e", "#f97316"   # ch1 sky, ch2 green, ch3 orange
    SKY, INK, MUT = "#0ea5e9", "#0f172a", "#64748b"
    GRN = "#22c55e"

    # ---- box registry -----------------------------------------------------
    B = {
        "init":    (240, 18, 260, 52),
        "assign":  (215, 102, 300, 56),
        "flood":   (255, 256, 230, 48),
        "ch1":     (70, 356, 180, 76),
        "ch2":     (300, 356, 180, 76),
        "ch3":     (620, 356, 190, 76),
        "belief":  (255, 488, 230, 48),
        "retrofit":(240, 670, 260, 52),
        "outputs": (620, 670, 200, 52),
    }
    # decision diamond: center + half-width/height
    DCX, DCY, DHW, DHH = 370, 602, 115, 36

    def port(name, side):
        x, y, w, h = B[name]
        if side == "top":    return (x + w / 2, y)
        if side == "bottom": return (x + w / 2, y + h)
        if side == "left":   return (x, y + h / 2)
        return (x + w, y + h / 2)   # right

    def dia_port(side):
        if side == "top":    return (DCX, DCY - DHH)
        if side == "bottom": return (DCX, DCY + DHH)
        if side == "left":   return (DCX - DHW, DCY)
        return (DCX + DHW, DCY)     # right

    # ---- drawing helpers --------------------------------------------------
    def box(name, fill, stroke, title, sub="", sub2="", tcol="#ffffff", fs=14,
            grad=False):
        x, y, w, h = B[name]
        f = "url(#gInk)" if grad else fill
        s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
             f'fill="{f}" stroke="{stroke}" stroke-width="1.5"/>')
        cx = x + w / 2
        if sub and sub2:
            s += f'<text x="{cx}" y="{y+h/2-13}" text-anchor="middle" dominant-baseline="middle" font-size="{fs}" font-weight="700" fill="{tcol}">{title}</text>'
            s += f'<text x="{cx}" y="{y+h/2+3}" text-anchor="middle" dominant-baseline="middle" font-size="10.5" fill="{tcol}" opacity="0.95">{sub}</text>'
            s += f'<text x="{cx}" y="{y+h/2+18}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="{tcol}" opacity="0.9" font-style="italic">{sub2}</text>'
        elif sub:
            s += f'<text x="{cx}" y="{y+h/2-7}" text-anchor="middle" dominant-baseline="middle" font-size="{fs}" font-weight="700" fill="{tcol}">{title}</text>'
            s += f'<text x="{cx}" y="{y+h/2+11}" text-anchor="middle" dominant-baseline="middle" font-size="10.5" fill="{tcol}" opacity="0.95">{sub}</text>'
        else:
            s += f'<text x="{cx}" y="{y+h/2}" text-anchor="middle" dominant-baseline="middle" font-size="{fs}" font-weight="700" fill="{tcol}">{title}</text>'
        return s

    def arrow(pts, color=MUT, dash="", marker="ah"):
        d = f'stroke-dasharray="{dash}"' if dash else ""
        pd = "M " + " L ".join(f"{px},{py}" for px, py in pts)
        return (f'<path d="{pd}" fill="none" stroke="{color}" stroke-width="2" '
                f'marker-end="url(#{marker})" {d}/>')

    def lbl(x, y, text, color=MUT, weight="700", size=12.5, anchor="middle", italic=False):
        it = ' font-style="italic"' if italic else ''
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
                f'font-weight="{weight}" fill="{color}"{it}>{text}</text>')

    def elbow_v(p_from, p_to, ymid):
        return [p_from, (p_from[0], ymid), (p_to[0], ymid), p_to]

    svg = f'''<svg viewBox="0 0 860 780" xmlns="http://www.w3.org/2000/svg"
      font-family="'Source Sans Pro',system-ui,sans-serif">
      <defs>
        <marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,3 L0,6 Z" fill="{MUT}"/></marker>
        <marker id="ahg" markerWidth="9" markerHeight="9" refX="8" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,3 L0,6 Z" fill="{GRN}"/></marker>
        <marker id="aho" markerWidth="9" markerHeight="9" refX="8" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,3 L0,6 Z" fill="{C3}"/></marker>
        <marker id="ah1" markerWidth="9" markerHeight="9" refX="8" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,3 L0,6 Z" fill="{C1}"/></marker>
        <marker id="ah2" markerWidth="9" markerHeight="9" refX="8" refY="3"
                orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,3 L0,6 Z" fill="{C2}"/></marker>
        <linearGradient id="gInk" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="{INK}"/><stop offset="1" stop-color="#1e293b"/>
        </linearGradient>
      </defs>

      <!-- dashed loop container (behind boxes) -->
      <rect x="40" y="220" width="530" height="410" rx="16" fill="none"
            stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="7 5"/>
      <text x="58" y="244" font-size="12" font-weight="800" fill="{MUT}"
            letter-spacing="0.6">EACH TIME STEP (YEAR)</text>

      <!-- boxes -->
      {box("init", "", INK, "Initialization", "households, elevation, network, ties", grad=True)}
      {box("assign", "#f1f5f9", "#cbd5e1", "Assign attributes &amp; prior belief", "risk perception, trusted info, forecast info", tcol=INK)}
      {box("flood", "#e0f2fe", SKY, "Annual flood level", "GEV sample f\u209c", tcol=INK, fs=14)}
      {box("ch1", C1, "#0369a1", "Channel 1", "Flood experience", "\u03bb_flood \u00d7 \u03bb_risk_perc", fs=14)}
      {box("ch2", C2, "#15803d", "Channel 2", "Proximity", "\u03bb_social \u00d7 \u03bb_sim", fs=14)}
      {box("ch3", C3, "#c2610c", "Channel 3", "Information (t=0)", "\u03bb_info \u00d7 \u03bb_response", fs=14)}
      {box("belief", "#f8fafc", "#cbd5e1", "Update belief P(H\u2081)", "posterior odds = prior odds \u00d7 Bayes factors", tcol=INK, fs=13.5)}
      {box("retrofit", "#dcfce7", GRN, "Retrofit  (absorbing)", "household leaves the risk pool permanently", tcol=INK, fs=14)}
      {box("outputs", "#eef2f7", "#cbd5e1", "Model outputs", "adoption curve, survey comparison", tcol=INK, fs=12.5)}

      <!-- decision diamond -->
      <polygon points="{DCX},{DCY-DHH} {DCX+DHW},{DCY} {DCX},{DCY+DHH} {DCX-DHW},{DCY}"
               fill="#fff7ed" stroke="{C3}" stroke-width="1.6"/>
      {lbl(DCX, DCY-4, "P(H\u2081) \u2265 \u03b8 ?", INK, size=14)}
      {lbl(DCX, DCY+15, "PMT threshold", MUT, weight="500", size=11.5)}

      <!-- ===== ARROWS (each anchored to an edge-centre port) ===== -->
      {arrow([port("init","bottom"), port("assign","top")])}
      {arrow([port("assign","bottom"), port("flood","top")])}
      {arrow(elbow_v(port("flood","bottom"), port("ch1","top"), 330), C1, marker="ah1")}
      {arrow(elbow_v(port("flood","bottom"), port("ch2","top"), 330), C2, marker="ah2")}
      {lbl(port("ch1","top")[0], 348, "if flooded", C1, size=13, italic=True)}
      {lbl(540, 342, "if a neighbour", C2, size=13, italic=True)}
      {lbl(540, 357, "retrofits", C2, size=13, italic=True)}
      {arrow([port("assign","right"), (715, port("assign","right")[1]), (715, port("ch3","top")[1])], C3, dash="6 4", marker="aho")}
      {lbl(712, 118, "once, at t=0", C3, size=13, anchor="end")}
      {arrow(elbow_v(port("ch1","bottom"), port("belief","top"), 462), C1, marker="ah1")}
      {arrow(elbow_v(port("ch2","bottom"), port("belief","top"), 462), C2, marker="ah2")}
      {arrow([port("ch3","bottom"), (port("ch3","bottom")[0], port("belief","right")[1]), port("belief","right")], C3, marker="aho")}
      {arrow([port("belief","bottom"), dia_port("top")])}
      {arrow([dia_port("left"), (25, dia_port("left")[1]), (25, port("flood","left")[1]), port("flood","left")])}
      {lbl(dia_port("left")[0]-45, dia_port("left")[1]-8, "no", MUT, size=13)}
      <rect x="6" y="372" width="22" height="150" fill="#ffffff" opacity="0.9"/>
      <text x="20" y="447" text-anchor="middle" font-size="12.5" fill="{MUT}"
            font-style="italic" transform="rotate(-90 20 447)">carry belief to next year</text>
      {arrow([dia_port("bottom"), port("retrofit","top")], GRN, marker="ahg")}
      {lbl(dia_port("bottom")[0]+26, dia_port("bottom")[1]+22, "yes", GRN, size=13, anchor="start")}
      {arrow([port("retrofit","right"), port("outputs","left")])}
    </svg>'''
    return svg


def _page_home():
    import streamlit as st

    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.markdown(
            f"<div style='font-size:2.0rem;font-weight:800;color:{CLR_INK};"
            "line-height:1.2;margin-bottom:0.2rem;'>Flood Mitigation "
            "Agent-based Model</div>"
            f"<div style='font-size:1.02rem;color:{CLR_MUTED};margin-bottom:1.0rem;'>"
            "A household decision model of flood-retrofit adoption, grounded in "
            "Bayesian belief updating and Protection Motivation Theory.</div>",
            unsafe_allow_html=True)

        st.markdown(
            "This tool simulates how households along a flood-exposed coastline "
            "decide whether to retrofit their homes. Each household holds a "
            "belief that it should retrofit and updates that belief as evidence "
            "arrives through three channels \u2014 **personal flood experience**, "
            "**social influence from retrofitted neighbours**, and **trusted "
            "information** \u2014 acting once its belief crosses a personal decision "
            "threshold.")
        st.markdown(
            "The model is calibrated to the NYC Flood Vulnerability Index Survey "
            "and supports both controlled experiments (**Research mode**) and "
            "real-data scenarios (**Case Study mode**), where you upload your "
            "own household locations and flood series.")

        st.markdown(
            f"<div style='margin-top:0.6rem;padding:0.8rem 1rem;background:#f8fafc;"
            f"border-left:3px solid {CLR_SKY};border-radius:6px;color:{CLR_MUTED};"
            "font-size:0.95rem;'>Use the left rail to get started: set "
            "parameters in <b>Settings</b>, press <b>Run Simulation</b>, and "
            "view the six result figures under <b>Results</b>. Full methodology "
            "and references are in <b>Documentation</b>.</div>",
            unsafe_allow_html=True)

    with right:
        # Show the workflow flowchart image (flowchart.png in the app folder).
        import os as _os
        if _os.path.exists("flowchart.png"):
            try:
                st.image("flowchart.png", width="stretch")
            except TypeError:
                st.image("flowchart.png", use_container_width=True)
        else:
            st.markdown(
                f"<div style='border:2px dashed {CLR_SLATE300};border-radius:10px;"
                f"padding:2.5rem 1rem;text-align:center;color:{CLR_MUTED};"
                "background:#f8fafc;'>Place <b>flowchart.png</b> in the app "
                "folder to display the model workflow diagram here.</div>",
                unsafe_allow_html=True)


def _run_app():
    import streamlit as st

    st.set_page_config(page_title="Flood Mitigation Agent-based Model",
                       page_icon="\U0001F30A", layout="wide",
                       initial_sidebar_state="expanded")
    _inject_css()

    if not _check_password():
        st.stop()

    ss = st.session_state
    # Apply any pending navigation request BEFORE the radio widget is created.
    # (Streamlit forbids mutating a widget's own key after instantiation, so
    # ---- navigation order: Home, Settings, Results, Documentation ----
    NAV_HOME = "\U0001F3E0  Home"
    NAV_SETTINGS = "\u2699\ufe0f  Settings"
    NAV_RESULTS = "\U0001F4CA  Results"
    NAV_DOC = "\U0001F4D8  Documentation"
    NAV_ORDER = [NAV_HOME, NAV_SETTINGS, NAV_RESULTS, NAV_DOC]

    # the run handler sets ss["pending_nav"] and we consume it here instead.)
    if ss.get("pending_nav"):
        ss["nav"] = ss.pop("pending_nav")
    # First load lands on Home (the intro page); no page auto-jumps.
    if "nav" not in ss or ss["nav"] not in NAV_ORDER:
        ss["nav"] = NAV_HOME
    if "mode" not in ss:
        ss["mode"] = "Research"

    # ---- navigation rail ----
    with st.sidebar:
        # Logo at the very top of the rail (as in the ADAPT tool). No text
        # brand beneath it. Falls back to a small emoji only if logo.png is
        # missing.
        import os as _os
        if _os.path.exists("logo.png"):
            try:
                st.image("logo.png", width="stretch")
            except TypeError:
                st.image("logo.png", use_container_width=True)
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rail-brand"><span class="rail-word">'
                        '\U0001F30A</span></div>', unsafe_allow_html=True)

        # ---- Mode selection (two buttons in one row; default Research) ----
        st.markdown('<div class="rail-label">Mode</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        if m1.button("Research", key="mode_research", use_container_width=True,
                     type=("primary" if ss["mode"] == "Research" else "secondary")):
            ss["mode"] = "Research"; st.rerun()
        if m2.button("Case Study", key="mode_case", use_container_width=True,
                     type=("primary" if ss["mode"] == "Case Study" else "secondary")):
            ss["mode"] = "Case Study"; st.rerun()

        # ---- Case Study uploads (only in Case Study Mode) ----
        uploaded_csv = uploaded_flood = None
        if ss["mode"] == "Case Study":
            st.markdown('<div class="rail-label">Upload data</div>',
                        unsafe_allow_html=True)
            uploaded_csv = st.file_uploader("Location CSV (x, y, z)", type=["csv"],
                                            key="csv_upl")
            uploaded_flood = st.file_uploader("Flood series CSV",
                                              type=["csv"], key="flood_upl",
                                              help="Column 'flood_level', one value per step.")

        st.markdown('<hr/>', unsafe_allow_html=True)

        nav = st.radio("Navigation", NAV_ORDER, key="nav",
                       label_visibility="collapsed")

        st.markdown('<hr/>', unsafe_allow_html=True)
        st.markdown('<div class="rail-label">Simulation</div>', unsafe_allow_html=True)
        run_clicked = st.button("\u25B6  Run Simulation", key="run_btn")
        progress_slot = st.empty()

    # ---- run action ----
    if run_clicked:
        params = _collect_params()
        params["MODE"] = ss["mode"]
        try:
            # Case Study Mode: load positions/elevations and optional flood series
            if ss["mode"] == "Case Study":
                if uploaded_csv is None:
                    st.warning("Upload a location CSV (columns x, y, z) to run "
                               "Case Study Mode.")
                    st.stop()
                uploaded_csv.seek(0)
                cdf = pd.read_csv(uploaded_csv)
                missing = {"x", "y", "z"} - set(cdf.columns)
                if missing:
                    st.error(f"Location CSV is missing column(s): "
                             f"{', '.join(sorted(missing))}.")
                    st.stop()
                params["CUSTOM_POSITIONS"] = cdf[["x", "y"]].to_numpy(float)
                params["CUSTOM_ELEVATIONS"] = cdf["z"].to_numpy(float)
                params["N_AGENTS"] = len(cdf)
                if uploaded_flood is None:
                    st.warning("Upload a flood series CSV (column 'flood_level') "
                               "to run Case Study Mode.")
                    st.stop()
                uploaded_flood.seek(0)
                fdf = pd.read_csv(uploaded_flood)
                if "flood_level" not in fdf.columns:
                    st.error("Flood series CSV needs a 'flood_level' column.")
                    st.stop()
                params["CUSTOM_FLOOD_SERIES"] = fdf["flood_level"].to_numpy(float)

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
            ss["pending_nav"] = NAV_RESULTS
            st.rerun()
        except Exception as e:
            import traceback
            st.error("The simulation failed. Details below.")
            st.exception(e); st.code(traceback.format_exc())

    # ---- page routing ----
    if nav == NAV_HOME:
        _page_home()
    elif nav == NAV_DOC:
        _page_documentation()
    elif nav == NAV_SETTINGS:
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
