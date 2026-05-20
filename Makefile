# Makefile — FOOTSIES PPO training pipeline
#
# Targets:
#   install  — create a virtualenv at ./venv and install all dependencies
#   train    — run a 1 M-step PPO training run (PBRS-shaped reward)
#   eval     — run league evaluation + tactical scenario evaluation
#   figures  — generate all thesis figures from pre-computed result files
#   all      — install → train → eval → figures in sequence
#
# Quick start:
#   make install
#   make all

PYTHON     := venv/bin/python
PIP        := venv/bin/pip
STEPS      := 1000000
SPEED      := 20.0
MODEL      := models/ppo_footsies_final.zip

.PHONY: install train eval figures all

## Create ./venv and install top-level requirements + footsies-gym in editable mode.
install:
	@echo "[install] Creating virtualenv and installing dependencies …"
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e footsies-gym/

## Train PPO for $(STEPS) steps with PBRS reward shaping; checkpoints → models/.
train:
	@echo "[train] Starting $(STEPS)-step PPO training run …"
	$(PYTHON) train.py --steps $(STEPS) --fast-forward-speed $(SPEED)

## Evaluate the final model: league (100 matches × 3 baselines) + 30 scenarios.
eval:
	@echo "[eval] Running league evaluation …"
	$(PYTHON) eval_league.py --model $(MODEL) --matches 100 --speed $(SPEED)
	@echo "[eval] Running scenario evaluation …"
	$(PYTHON) eval_scenarios.py --model $(MODEL)

## Generate all 7 thesis figures from results/ → figures/.
figures:
	@echo "[figures] Generating thesis figures …"
	$(PYTHON) generate_thesis_figures.py

## Run the full pipeline end-to-end: install → train → eval → figures.
all: install train eval figures
	@echo "[all] Pipeline complete."
