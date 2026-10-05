# DirectCrystalFlow

**Symmetry-exact crystal generation by flow matching over asymmetric-unit site tokens.**

DirectCrystalFlow represents a crystal as at most twenty canonically ordered latent tokens, one per site of the
asymmetric unit, learns their distribution by flow matching, and decodes every token onto an exact Wyckoff
subspace before regenerating the cell by orbit expansion. Every generated structure is therefore closed under its
space group by construction, and the space group can be requested at sampling time.

<p align="center"><img src="docs/assets/overview.png" alt="Overview of DirectCrystalFlow" width="720"></p>

> **Status:** research code accompanying a manuscript in preparation. Results will be linked here once the paper
> is available. Interfaces may change.

## Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Data](#data)
4. [Training](#training)
5. [Generating crystals](#generating-crystals)
6. [Evaluation and figures](#evaluation-and-figures)
7. [DFT validation](#dft-validation)
8. [The symmetry correction (`symfix`)](#the-symmetry-correction-symfix)
9. [Configuration and privacy](#configuration-and-privacy)
10. [Citation and licence](#citation-and-licence)

## Repository layout

```
src/
  sitetokens.py        the model, losses, data pipeline and 3-stage training (11 numbered sections)
  symfix.py            stored-basis symmetry correction applied before generation
scripts/
  visualize.py         trajectories, and a panel of crystals under requested space groups
  make_figures.py      results figures from an evaluation JSON
  dft_stability.py     VASP relaxation and energy above the Materials Project hull
  README.md            what each script needs (evaluate.py is still to be added)
data/ data_prep/       where the dataset goes, and the scripts that build it
checkpoints/ results/  git-ignored outputs
docs/                  architecture, training, evaluation and DFT notes
tools/                 check_private_paths.py (privacy check, also run in CI and pre-commit)
tests/                 hygiene and smoke tests that need no GPU
```

## Installation

```bash
git clone https://github.com/<your-username>/DirectCrystalFlow.git
cd DirectCrystalFlow
python -m venv .venv && source .venv/bin/activate
# install PyTorch and PyTorch Geometric for your CUDA version first (see their install pages), then:
pip install -e ".[eval,dft,dev]"
cp .env.example .env        # edit the paths; the file is git-ignored
set -a; source .env; set +a
```

`pip install -e .` makes `import sitetokens` and `import symfix` work from anywhere, which is how the scripts
find the model.

## Data

The model trains on the mined MP-20 dataset (`train/ val/ test/`). Point `MP20_ROOT` at it; no path is stored in
the code. Layout and sources are in [`data/README.md`](data/README.md).

## Training

All three stages run from one command. If no pretrained encoder is given, Stage 1 runs first.

```bash
python src/sitetokens.py --data-root "$MP20_ROOT"
```

| Stage | What is trained | Defaults |
|---|---|---|
| 1. autoencoder pretraining | encoder, site decoder, structural heads | `run_encoder_pretrain` |
| 2. latent flow matching | flow and conditioner; the encoder stays frozen | 80 epochs, lr 2e-4 (`--flow-epochs`, `--flow-lr`) |
| 3. generative fine-tuning | flow and conditioner at lr, decoder and heads at a quarter of it | 30 epochs, lr 1e-4, 8 rollout steps (`--s3-epochs`, `--s3-lr`, `--s3-steps`, `--no-stage3`) |

Other flags: `--encoder-path`, `--target`, `--batch-size`, `--n-sites`, `--device`. Checkpoints are written to
`checkpoints/` (`EMF_CKPT_DIR`). Details: [`docs/TRAINING.md`](docs/TRAINING.md).

## Generating crystals

```bash
export EMF_CHECKPOINT=checkpoints/stage3_finetune.pt

# a figure of crystals generated under requested space groups, with the asymmetric unit
# outlined and Wyckoff labels (default: P-1, P2_1/c, R-3, P6_3/mmc, Fm-3m, three each)
python scripts/visualize.py --panel

# one generation trajectory with per-site diagnostics and an audited CIF
python scripts/visualize.py --target_energy -1.0 --num_crystals 3 --target_sg 225
```

The visualiser installs the symmetry correction before generating and refuses to run the panel without it.

## Evaluation and figures

`scripts/evaluate.py` (to be added, see [`scripts/README.md`](scripts/README.md)) writes an evaluation JSON.
`scripts/make_figures.py emf_eval_results.json` draws the results figures from that file only, so figures and text
cannot drift apart. See [`docs/EVALUATION.md`](docs/EVALUATION.md).

## DFT validation

`scripts/dft_stability.py` relaxes selected structures with VASP (Materials Project settings, via pymatgen and
custodian) and computes the energy above the full Materials Project hull. It needs a licensed VASP and a free
Materials Project API key. See [`docs/DFT.md`](docs/DFT.md).

## The symmetry correction (`symfix`)

The orbit expansion in the model builds cells in a conventional basis that can differ from the basis in which the
Wyckoff codebook was mined. `symfix.install_stored_basis_symmetry` replaces it with expansion in the stored basis.
Any script that generates structures must install it before sampling; without it, cells come out several times too
large and some groups (e.g. `Pnma`) fail. `symfix` is applied automatically by `scripts/visualize.py`.

## Configuration and privacy

Locations come from environment variables (`.env.example`) or flags; no machine-specific path, e-mail address or
key is stored in the code. Before every commit run

```bash
python tools/check_private_paths.py
```

(also run by CI and, if you install it, by pre-commit: `pre-commit install`).

## Citation and licence

See [`CITATION.cff`](CITATION.cff) (fill in the authors and the DOI when available). Code is released under the
MIT licence, see [`LICENSE`](LICENSE). The Materials Project and MP-20 data have their own licences and are not
distributed here.
