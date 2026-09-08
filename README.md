# LyTimeT: Towards Robust and Interpretable State-Variable Discovery

## Overview
This repo contains the reference implementation for **"LyTimeT: Towards Robust and Interpretable State-Variable Discovery"** by Kuai Yu, Crystal Su, Xiang Liu, Judah Goldfeder, Mingyuan Shao, and Hod Lipson ([arXiv:2510.19716](https://arxiv.org/html/2510.19716v1)).

LyTimeT is a two-phase framework for extracting robust, interpretable state variables from high-dimensional video of dynamical systems:

- **Phase 1 — Distraction-robust video prediction.** A spatio-temporal, TimeSformer-based encoder–decoder (`ViTTimeSformerEncoderDecoder` in `model_utils.py`, wired up as the `vit-tide` model in `models.py`) uses factorized spatial and temporal attention to focus on dynamically relevant regions of the frame while suppressing nuisance variation (background motion, occlusions, texture/lighting changes). The encoder mean-pools its tokens into a compact latent state $z_t$, a lightweight decoder reconstructs frames, and a residual-MLP latent transition model performs multi-step (K-step) unrolling for long-horizon forecasting.
- **Phase 2 — Variable extraction and Lyapunov stability.** The learned latent space is probed with linear regression against candidate ground-truth variables to rank and select the most physically meaningful dimensions ($\tilde z_t$). The transition dynamics of these selected dimensions are then refined (`RefineModel` / `RefineDynamicsModel`) with a Lyapunov-based stability regularizer that penalizes non-decreasing energy $V(\tilde z) = \|W\tilde z\|_2^2$ along roll-outs, encouraging contractive, error-resistant trajectories.

This design turns LyTimeT from a pure video predictor into a tool for scientific discovery: low-dimensional, interpretable state trajectories that remain stable under nuisance perturbations and chaotic dynamics, evaluated via Mutual Information (MI), Analytical Mean Squared Error (AMSE), and Intrinsic Dimension (ID) estimation. A lightweight **LyTimeT-Lite** variant (fewer heads/hidden dim, patch sparsification) trades a small amount of accuracy for ~25% lower latency and memory, making it suitable for real-time or resource-constrained deployment.

## Installation

Create a conda virtual environment and install the dependencies. All the experiments were performed on one GeForce RTX 2080 Ti Nvidia GPU with Ubuntu 18.04 and CUDA 11.4.

```
conda env create -f environment.yml
```
**Note**: You will need to install Julia 1.8.1 separately to use the symbolic regression libraries.

## Data Preparation

Please refer to the [collect](collect) folder for details about the synthetic data collection process. Store each dataset as ```data/{dataset_name}```, where ```data``` is a custom dataset folder specified by the ```data_filepath``` item in the ```config.yaml``` files. Human variables, if available, should be saved as ```data/{dataset_name}/states.npy```.

## Training and Testing

LyTimeT trains two networks in sequence, mirroring the two phases described above:

1. **Phase 1 — Navigate to the `scripts` directory and train the `vit-tide` (TimeSformer-based) encoder-decoder model.** Configs for this model live under `configs/{dataset_name}/vit/` with `model_name: 'vit-tide'`.
    ```
    ./vit_train.sh {dataset_name} {gpu #}
    ```
    This runs `main.py` against `configs/{dataset_name}/vit/config{1,2,3}.yaml`, producing checkpoints under `logs/{dataset_name}_vit-tide_{seed}`. There are currently no dedicated `vit_eval.sh` / `vit_gather.sh` scripts, so run evaluation/gathering directly with `eval.py`, e.g.:
    ```
    python ../eval.py ../configs/{dataset_name}/vit/config1.yaml ../logs/{dataset_name}_vit-tide_1/lightning_logs/checkpoints NA eval-eval
    python ../eval.py ../configs/{dataset_name}/vit/config1.yaml ../logs/{dataset_name}_vit-tide_1/lightning_logs/checkpoints NA eval-encoder-decoder-train
    ```
    (repeat for `config2.yaml`/seed 2 and `config3.yaml`/seed 3). The `eval-eval` mode reports test metrics; `eval-encoder-decoder-train` mode saves the latent vectors, means (`mu`), and log-variances (`logvar`) for the train/val/test splits — this is the state representation $z_t$ that Phase 2 will probe and refine. Alternatively, adapt `encoder_decoder_64_eval.sh` / `encoder_decoder_64_gather.sh` by pointing their paths at `configs/{dataset_name}/vit` and `logs/{dataset_name}_vit-tide_*` instead of `model64`/`encoder-decoder-64`.

    (To use the plain CNN-based encoder-decoder baseline instead of the TimeSformer variant, use `configs/{dataset_name}/model64/` (`model_name: 'encoder-decoder-64'`) with the existing scripts:
    ```
    ./encoder_decoder_64_train.sh {dataset_name} {gpu #}
    ./encoder_decoder_64_eval.sh {dataset_name} {gpu #}
    ./encoder_decoder_64_gather.sh {dataset_name} {gpu #}
    ```
    )

2. **Estimate the intrinsic dimension** from the saved Phase 1 latent vectors via the 2-NN estimator. This determines how many dimensions are linearly probed/ranked as candidate state variables and sets the latent dimension of the Phase 2 refinement network.
    ```
    ./estimate_dimension.sh {dataset_name}
    ```

3. **Phase 2 — Train the `refine-64` model** on the saved Phase 1 latents. This linearly probes and ranks latent dimensions against available ground-truth variables, refines the selected transition dynamics, and applies the Lyapunov stability regularizer to enforce contractive roll-outs. Save the resulting state variables from training, validation, and testing for downstream symbolic regression.
    ```
    ./refine_64_train.sh {dataset_name} {gpu #}
    ./refine_64_eval.sh {dataset_name} {gpu #}
    ./refine_64_gather.sh {dataset_name} {gpu #}
    ```

4. **Compute evaluation metrics** (MI, AMSE, and intrinsic dimension error against ground truth where available).
    ```
    ./evaluate_metrics.sh {dataset_name}
    ```

All models are saved in a separate `logs` directory using the naming convention `{dataset_name}_{model_name}_{seed}`.

### Key loss terms
- **Reconstruction / prediction loss** (`encoder_decoder64_loss` in `models.py`): per-frame reconstruction plus multi-step-ahead prediction error, combined with a smoothness regularizer and a KL term over the latent distribution.
- **Refinement loss** (`refine_loss`): reconstructs the Phase 1 latent from the extracted state variables, predicts next-step state variables, and applies the same smoothness/KL structure over the reduced latent space — this is where the Lyapunov-style stability regularization on the selected variables is incorporated.

## Symbolic regression

After training the above models, we derive analytical expressions for the state variables. Navigate to the ```scripts``` directory and train/evaluate the symbolic regression model. The model will be saved in a separate ```symbolic_regression``` directory in the logging folder.
```
./symbolic_regression_train.sh {dataset_name}
./symbolic_regression_eval.sh {dataset_name}
```

## Datasets

LyTimeT is evaluated on five synthetic dynamical systems with known ground-truth state variables (`collect/circular_motion`, `collect/single_pendulum`, `collect/double_pendulum`, `collect/elastic_pendulum`, `collect/reaction_diffusion`) and four real-world video datasets (`swing_stick`, with annotated ground truth, plus `air_dancer`, `lava_lamp`, and `fire_flame`, which are unannotated chaotic phenomena evaluated only via intrinsic dimension). See the [collect](collect) folder for data-generation scripts and the [datainfo](datainfo) folder for train/val/test split definitions.

## Evaluation Metrics

- **Mutual Information (MI)** between extracted latent dimensions and ground-truth state variables (Gaussian kernel density estimation).
- **Analytical Mean Squared Error (AMSE)** from a linear probe fit between extracted variables and ground truth.
- **Intrinsic Dimension (ID)** estimated via the two-nearest-neighbor (2-NN) estimator, compared against known ground-truth dimensionality where available (see `utils/dimension.py`).

## Citation

If you use this code, please cite the LyTimeT paper:

```bibtex
@article{yu2025lytimet,
  title   = {LyTimeT: Towards Robust and Interpretable State-Variable Discovery},
  author  = {Yu, Kuai and Su, Crystal and Liu, Xiang and Goldfeder, Judah and Shao, Mingyuan and Lipson, Hod},
  journal = {arXiv preprint arXiv:2510.19716},
  year    = {2025}
}
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.