# Training

```bash
export MP20_ROOT=/path/to/MP20          # see data/README.md
python src/sitetokens.py --data-root "$MP20_ROOT"
```

## Stages

1. **Autoencoder pretraining.** Encoder, site decoder and structural heads (lattice shape and volume, space group,
   multiplicity, property) are trained jointly with a reconstruction objective. On exit the encoder, projection and
   standardiser are frozen and the property head is frozen and copied into a critic. Skipped if `--encoder-path`
   points to a checkpoint whose encoder is already pretrained.
2. **Latent flow matching.** Only the flow, the conditioner and the coarse space-group head train. One
   teacher-forced step per sample; the decoder is never involved.
3. **Generative fine-tuning** (`--no-stage3` skips it). The flow is rolled out from noise (an Euler prefix without
   gradient, then a differentiated tail), the rolled-out latent is **detached** and decoded, and the structural
   penalties (composition, charge balance, repulsion, density, space-group agreement) train the decoder and heads
   only. The frozen property critic is the one term that moves the flow, together with the flow-matching anchor.

## Outputs

Per stage: a checkpoint in `checkpoints/` and a metrics CSV (git-ignored). Stage names and the 11-section layout of
the code are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Reproducibility

Record the commit hash, the exact command, the dataset build, and the versions of `torch`, `pymatgen` and `spglib`
next to every checkpoint. Spglib tolerances matter: the evaluation uses `symprec = 0.1`.
