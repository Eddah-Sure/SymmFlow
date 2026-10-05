# Architecture

## Idea

A crystal is split into its **asymmetric unit** (one representative of every symmetry orbit) plus a **space group**
`g`. The asymmetric unit has at most `n_sites = 20` sites. Each site becomes one latent token, in a canonical order,
so the latent is a fixed-size set `z in R^(20 x 64)` with a mask for empty slots.

* **Encoder** (Section 4): equivariant message passing over the periodic graph, a set transformer, and a
  canonical gather that produces one token per site.
* **Flow** (Section 5): a transformer velocity field `v(z, t, c)` trained by flow matching from Gaussian noise to the
  encoder's tokens. The condition `c = (property, space group)` is dropped independently with probability 0.15 for
  classifier-free guidance.
* **Decoder** (Section 5): per token, predicts whether the slot is occupied (a monotone cut from a count head), its
  Wyckoff class, the element, and the free coordinates; the position is **projected onto the exact Wyckoff subspace**
  `o + P<f - o>`.
* **Cell assembly** (Sections 3 and 10): the lattice is built in the crystal family of `g` from the exact site
  multiplicities, collisions between sites are resolved, and orbit expansion regenerates the full cell. Every output is
  therefore closed under its space group.

## Why one module

`symfix.py` patches functions in the model module's namespace at run time. Splitting `sitetokens.py` into several
modules would change where those names live, so the model is kept in one file, divided into eleven numbered sections:

| # | Section of `src/sitetokens.py` | Contents | Lines |
|---|---|---|---|
| 1 | Constants, Chemistry Tables And Diagnostic Warnings | element tables, oxidation states, covalent radii, warnings | 56-208 |
| 2 | Geometry: Lattices, Minimum Image, Graph Construction | lattice parameters <-> matrices, minimum-image distances, graph building | 210-446 |
| 3 | Symmetry: Space-Group Operations, Wyckoff Tables, Orbits | space-group operations, Wyckoff tables, orbit expansion | 448-1173 |
| 4 | Encoder: Message Passing And The Canonical Site-Token Set | equivariant message passing; canonical ordering of the site-token set | 1175-1591 |
| 5 | Generator: Flow Field, Conditioner, Decoder, Auxiliary Heads | velocity field (transformer), conditioner (property + space group), site decoder, auxiliary heads | 1593-2196 |
| 6 | DirectCrystalFlow | the model class: encoding, sampling, stage-specific loss assembly | 2198-3598 |
| 7 | Losses | reconstruction, flow-matching and generated-crystal losses | 3600-4138 |
| 8 | Data: Records, Dataset, Collation | records, `MultiFileGraphDataset`, collation | 4140-4685 |
| 9 | Training | `run_encoder_pretrain`, `run_direct_flow`, `run_stage3_finetune` | 4687-5525 |
| 10 | Sample-Time Post-Processing And Validity Metrics | sample-time repairs and validity metrics (SMACT, distance gate) | 5527-6485 |
| 11 | Configuration And Entry Points | `default_config`, `main`, `run`, command-line interface | 6487-7162 |

(Line numbers are for version 0.1.0 and will drift; search for the section banner instead.)

## Checkpoint contents

A checkpoint stores `config` (the `DirectFlowConfig` fields) and `model` (the state dict). Scripts rebuild the
configuration from it, so the vocabulary and sizes always match the weights.
