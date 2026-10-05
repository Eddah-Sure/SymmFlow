# Checkpoints

Model weights are not tracked by git (`*.pt` is ignored). Training writes them here (`EMF_CKPT_DIR`, default
`checkpoints/`); the scripts load the one named by `EMF_CHECKPOINT`.

| File | Written after |
|---|---|
| `encoder_pretrained.pt` | Stage 1, autoencoder pretraining |
| `direct_flow.pt` | Stage 2, latent flow matching |
| `stage3_finetune.pt` | Stage 3, generative fine-tuning (the model used for evaluation) |

Publish released weights as a GitHub Release asset or on Zenodo and link them from the README.
