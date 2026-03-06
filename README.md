# Flood Adaptation ABM v12

An agent-based model of household flood-retrofit adoption using Bayesian belief updating.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)

## What the model does

Agents decide whether to retrofit their property against flooding. Each agent holds a probability P(H1) representing their belief that their situation warrants retrofitting. This belief is updated each time step through three evidence channels, all using Bayes' theorem in odds form (Jaynes, 2003; Kass & Raftery, 1995).

**Channel 1 -- Flood experience.** When an agent is flooded, its odds are multiplied by a Bayes factor (LAMBDA_FLOOD). Safe time steps produce no update, following the availability heuristic (Tversky & Kahneman, 1974).

**Channel 2 -- Proximity.** When a connected neighbor retrofits, the agent's odds are multiplied by LAMBDA_SOCIAL. Connections are binary: agents within a distance threshold are connected; others are not.

**Channel 3 -- Similarity.** Within DBSCAN-identified neighborhoods (Ester et al., 1996), Jaccard attribute similarity between agents scales a third Bayes factor: LAMBDA_SIMILARITY raised to the power of the similarity score (Jaccard, 1912).

When P(H1) crosses the agent's individual threshold (Rogers, 1975), the agent retrofits permanently.

## Interactive app

The Streamlit app provides two modes:

- **Research Mode** -- Synthetic spatial layouts with GEV flood generation. Noise toggles for PMT threshold heterogeneity and elevation randomness.
- **Case Study Mode** -- Upload agent location data (CSV with x, y, z columns) and optionally a flood time series for a specific site.

All parameters are adjustable through the sidebar. Results include adoption curves, belief evolution, network visualizations, and comparison against observed data.

## Running locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or run the model directly (command line, no GUI):

```bash
python AAA_model_v12.py
```

Output is saved to a timestamped folder under `output/`.

## File structure

| File | Description |
|------|-------------|
| `streamlit_app.py` | Interactive web dashboard |
| `AAA_model_v12.py` | Main model: parameters, agents, Bayesian updating, visualization |
| `FFF_flood.py` | GEV flood generation (Coles, 2001) |
| `FFF_spatial.py` | Agent positions and elevations |
| `FFF_attributes.py` | Agent attributes and Jaccard similarity |
| `FFF_network.py` | Binary social network within distance threshold |
| `FFF_neighborhood.py` | DBSCAN neighborhood identification |
| `User_Manual_v12.docx` | Full documentation |
| `requirements.txt` | Python dependencies |

## Parameters

All parameters are set at the top of `AAA_model_v12.py` and can be adjusted in the Streamlit sidebar.

| Parameter | Default | Description |
|-----------|---------|-------------|
| INITIAL_BELIEF | 0.05 | Prior P(H1) |
| LAMBDA_FLOOD | 1.20 | Bayes factor per flood event |
| LAMBDA_SOCIAL | 1.50 | Bayes factor per connected neighbor retrofit |
| LAMBDA_SIMILARITY | 3.00 | Bayes factor at full Jaccard similarity |
| PMT_THRESHOLD_MEAN | 0.50 | Decision threshold |
| DISTANCE_THRESHOLD | 0.09 | Max distance for network connections |
| N_AGENTS | 200 | Number of agents |
| TIME_STEPS | 100 | Simulation length |

## Case Study Mode -- CSV format

**Location file** (required): CSV with columns `x`, `y`, `z`. Values in [0, 1].

```csv
x,y,z
0.12,0.45,0.03
0.15,0.47,0.04
...
```

**Flood time series** (optional): CSV with column `flood_level`. One value per time step.

```csv
flood_level
0.02
0.00
0.08
...
```

## References

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters. KDD-96, 226-231.
- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37-50.
- Jaynes, E.T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press.
- Kass, R.E. & Raftery, A.E. (1995). Bayes Factors. *JASA*, 90(430), 773-795.
- Kazil, J., Masad, D., & Crooks, A. (2020). Utilizing Python for Agent-Based Modeling: The Mesa Framework. Springer.
- McPherson, M., Smith-Lovin, L., & Cook, J.M. (2001). Birds of a feather: Homophily in social networks. *Annual Review of Sociology*, 27, 415-444.
- Rogers, R.W. (1975). A protection motivation theory of fear appeals. *Journal of Psychology*, 91(1), 93-114.
- Tversky, A. & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124-1131.

## License

MIT
