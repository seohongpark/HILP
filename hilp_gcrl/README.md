# Foundation Policies with Hilbert Representations (Goal-Conditioned RL)

## Overview
This codebase contains the official implementation of **[Hilbert Foundation Policies](https://seohong.me/projects/hilp/)** (**HILPs**) for **goal-conditioned RL**.
The implementation is based on [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://github.com/seohongpark/HIQL).

## Requirements
* Python 3.8
* MuJoCo 2.1.0

## Installation
```
conda create --name hilp_gcrl python=3.8
conda activate hilp_gcrl
pip install -r requirements.txt --no-deps
pip install "jax[cuda11_cudnn82]==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Examples
```
# HILP on antmaze-large-diverse
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name antmaze-large-diverse-v2

# HILP-Plan on antmaze-large-diverse
python main.py --run_group EXP --agent_name hilp --algo_name hilp --planning_num_recursions 3 --seed 0 --env_name antmaze-large-diverse-v2

# HILP on antmaze-ultra-diverse
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name antmaze-ultra-diverse-v0

# HILP on kitchen-partial
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name kitchen-partial-v0 --train_steps 500000 --eval_interval 50000 --save_interval 500000 

# HILP on visual-kitchen-partial
mkdir -p data/d4rl_kitchen_rendered
python dataset_render.py --env_name kitchen-partial-v0
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name visual-kitchen-partial-v0 --train_steps 500000 --eval_interval 50000 --save_interval 500000 --expectile 0.7 --skill_expectile 0.7 --batch_size 256 --encoder impala_small --p_aug 0.5
```

## Issues

* (Added on 2025-09-29, reported by [@qortmdgh4141](https://github.com/qortmdgh4141))
The original implementation has an issue where the agent uses the same set of (batch-sized) random skills during training when JIT is enabled.
This issue has been fixed in the `master` branch.
However, the results in the paper were obtained using the original implementation, and we provide the original code in the `reproduce` branch for reproducibility.

## License

MIT