# Adaptive AI Agent for Fighting Games

Bachelor's thesis project — Universitatea "1 Decembrie 1918" din Alba Iulia, 2026.

Deep reinforcement learning agent for the FOOTSIES fighting game, trained via Proximal Policy Optimization (PPO) with potential-based reward shaping (PBRS).

## Stack

- Python 3.12.7
- Stable-Baselines3 2.8.0 (PPO with MultiInputPolicy)
- Gymnasium 1.2.3
- PyTorch 2.12.0
- FOOTSIES game built from Unity (martinhoT/Footsies-Gym)

## Layout

```
├── train.py                  # PPO + PBRS training
├── eval_league.py            # League evaluation vs. 3 baselines
├── eval_scenarios.py         # Structured tactical scenario eval (30 scenarios)
├── export_tb.py              # TensorBoard → CSV exporter
├── watch_agent.py            # Live render of trained agent
├── test_env.py               # Smoke test for env setup
├── scenarios.json            # 30 authored evaluation scenarios
├── footsies-env/             # Modified Unity game source (build output gitignored)
└── results/                  # Training curves, league results, scenario results (CSV/JSON)
```

## Reproducing

1. Install Unity Hub and Unity 6 (or 2022.3.10f1). Build the FOOTSIES project to `footsies-env/Build/FOOTSIES`.
2. `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. `python train.py --steps 1000000 --fast-forward-speed 20.0`
4. `python eval_league.py --matches 100 --speed 20`
5. `python eval_scenarios.py`

To watch a trained agent play (requires a display):

```bash
python watch_agent.py --model models/ppo_footsies_final.zip
```

## Key result

The trained agent achieves a positive win rate against the built-in CPU bot but exhibits positional overfitting: strong performance as P1, weak performance as P2. This empirically validates the literature's argument for symmetric training and self-play curricula.

## License

Academic / educational use. FOOTSIES game © Hi-Fight.
