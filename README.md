<div align="center">

<h1>Flow Matching Loss for Offline Flow Policy Reinforcement Learning</h1>

</div>




## Installation

QGRQL builds on top of RQL and requires Python 3.9+ and is based on JAX. The main dependencies are
`jax >= 0.4.26`, `ogbench == 1.2.0`, and `gymnasium == 0.29.1`.
To install the full dependencies, simply run:
```bash
pip install -r requirements.txt
```


## Usage

The main implementation of QGRQL is in [agents/rql.py](agents/rql.py).
QGRQL is enabled with `--agent.opex=True`; with the flag unset the agent reduces to RQL.

Complete list of QGRQL commands for every environment: [hyperparameters.sh](hyperparameters.sh)

```bash
# QGRQL
python main.py \
    --agent=agents/rql.py \
    --env_name=humanoidmaze-large-navigate-singletask-v0 \
    --agent.alpha=0.3 \
    --agent.expectile=0.5 \
    --agent.ensemble_ct=10 \
    --agent.rho=0.0 \
    --agent.h=1 \
    --agent.discount=0.995 \
    --agent.opex=True \
    --agent.opex_steps=10 \
    --agent.opex_lr=0.1 \
    --agent.opex_max_radius=0.2 \
    --offline_steps=1000000 \
    --online_steps=0 \
    --agent.batch_size=256
```

```bash
# RQL baseline (same command without --agent.opex)
python main.py \
    --agent=agents/rql.py \
    --env_name=humanoidmaze-large-navigate-singletask-v0 \
    --agent.alpha=0.3 --agent.expectile=0.5 --agent.ensemble_ct=10 \
    --agent.rho=0.0 --agent.h=1 --agent.discount=0.995 \
    --offline_steps=1000000 --online_steps=0 --agent.batch_size=256
```


## Hyperparameters

Full hyperparameters are reported in the paper. Per-environment commands for all ten
environments are in [hyperparameters.sh](hyperparameters.sh).

QGRQL adds three hyperparameters on top of RQL:

| Hyperparameter | Flag | Value |
|---|---|---|
| Gradient ascent steps `k` | `--agent.opex_steps` | 10 |
| Ascent step size `β` | `--agent.opex_lr` | 0.1 |
| L2 ball radius `R` | `--agent.opex_max_radius` | 0.2 (default), 0.05 (`cube-quadruple-100M`, `puzzle-4x4-100M`) |


## Ablations

Each ablation is a single flag change on the command above.

```bash
# Trust region: sweep the L2 ball radius. R=0 removes the projection entirely
--agent.opex_max_radius=0.05 | 0.1 | 0.2 | 0.4 | 0
```

```bash
# Delayed start: train with dataset actions until this step, then switch on opex
--agent.opex_start_step=100000
```

```bash
# Action source: ascend from dataset actions (the method) or from policy samples (an ablation we tried)
--agent.opex_source=data | policy
```

```bash
# Where the optimized action is applied: to the intermediate state x_t, to the
# flow-matching target, or to both (the method).
--agent.opex_apply_to_xt=True  --agent.opex_apply_to_tgt=True    # QGRQL
--agent.opex_apply_to_xt=False --agent.opex_apply_to_tgt=True    # target only
--agent.opex_apply_to_xt=True  --agent.opex_apply_to_tgt=False   # intermediate state only
```

```bash
# Value-maximization term: 0.0 to drop it from the actor loss
--agent.actor_q_coef=0.0
```

## Using larger datasets

The paper uses 100m-sized datasets for the OGBench puzzle-4x4 & cube-quadruple environments.
These datasets can be downloaded with the following commands (see [this section of the OGBench repository](https://github.com/seohongpark/ogbench?tab=readme-ov-file#additional-features) for more diverse 100M-sized datasets available):
```bash
# cube-quadruple-play-100m (100 datasets * 1000 length-1000 trajectories).
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
# puzzle-4x4-play-100m (100 datasets * 1000 length-1000 trajectories).
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/puzzle-4x4-play-100m-v0/
```

Pass the download directory with `--ogbench_dataset_dir=<path>`.


## Acknowledgments

This codebase is built on top of the reference implementation of Reversal Q-Learning (RQL).
