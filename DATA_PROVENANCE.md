# Data provenance and redistribution status

This document records what enters the public PolyBench26 data pipeline. It is
an inventory, not a substitute for the licences or terms of the originating
resources. Entries marked **pending confirmation** must be resolved using the
authoritative source record before a public release.

## Included source inputs

| Source label | Tracked source files | Repository use | Redistribution status | Required release record |
|---|---|---|---|---|
| OMersBench / OPoly26 and PolyBench26 curation | `Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl` | `OMers_convert_jsonl.py` creates the MD_300 and MD_5000 canonical PSMILES datasets, including the Cp, Cv, Rg, density, and refractive-index tasks. The bundled file is a PolyBench26-curated source with a novel property contribution; it is not a byte-for-byte mirror of the upstream OPoly26 release. | The underlying OPoly26 data are **CC BY 4.0**; retain appropriate attribution for that source. The upstream code is MIT-licensed; that code licence does not replace the dataset licence. | **Verified:** the pinned upstream revision is recorded as provenance for underlying inputs, and the bundled curated source and its checksum are documented below. |
| Coley 2022 / VIPEA (polymer-chemprop) data | `Datasets/Dataset_construction_scripts/files/polymer-chemprop-data/dataset.csv`; `dataset-poly_chemprop.csv` | `process_Vipea_data.py` derives PSMILES and wPSMILES EA/IP datasets. | **Redistributable under MIT**, as released in the upstream GitHub source. Derived representations are permitted under that licence. | **Verified:** both tracked files are byte-identical to the pinned upstream Git objects documented below. |
| polyVERSE | `Datasets/PSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv`; `Datasets/PSMILES/polyVERSE/ionization_energy/ionization_energy_data_polymers_v4.csv` | `convert_web_datasets.py` converts the tracked PSMILES inputs to wPSMILES. | **Redistributable only under the GTRC General Public Use License Agreement.** Covered copies and derivatives must retain required notices and be made available at no charge under the same licence; the Program may not be sold for commercial gain without a separate GTRC agreement. | **Verified:** bundled compact inputs match the official polyVERSE GitHub repository; the required GTRC licence is tracked with this repository. |
| PolyMetriX | `Datasets/PSMILES/PolyMetriX/Tg/Tg.csv` | Canonical PSMILES input for the Tg scaling condition; converted to wPSMILES by `convert_web_datasets.py`. The upstream code is released under MIT; the associated dataset is hosted on Zenodo. | **Redistributable under CC BY 4.0**, including derived representations, provided appropriate attribution is retained. The upstream MIT code licence applies to code, not the dataset. | Confirm that the tracked file corresponds to the cited Zenodo release and resolve the relationship of the additional DOI listed below. |

`Datasets/dataset_manifest.json` is the machine-readable minimum inventory
used by the fresh-clone tests. It should be extended whenever a new tracked
source dataset is added.

## External model dependency

polyBERT is not stored or redistributed by this repository. It is required
only when users generate polyBERT embeddings locally.

- Citation: Kuenneth, C., & Ramprasad, R. (2023). *polyBERT: a chemical
  language model to enable fully machine-driven ultrafast polymer informatics*.
  *Nature Communications*, 14, 4099.
  https://doi.org/10.1038/s41467-023-39868-6
- Upstream checkout: https://huggingface.co/kuelumbus/polyBERT
- Immutable revision used for PolyBench26 embedding generation:
  `deaa98fb65a7bdfb537457d42f43bd468963f695` (2023-07-18).
- Licence: GTRC General Public Use License Agreement, as supplied in the
  upstream model checkout. The model is not stored or redistributed by this
  repository. Users must obtain it directly from the upstream host and comply
  with its licence.
- The upstream model card describes a 600-dimensional SentenceTransformer
  embedding model. The model is cited below; the external checkout is separate
  from `Models/polyBERT/`, which contains only the PolyBench26 feed-forward
  trainer.

### PolyMetriX source record

- Citation: Kunchapu, S., & Jablonka, K. M. (2025). *Curated Glass Transition
  Temperature for Polymers* (Version v1) [Dataset]. Zenodo.
  https://doi.org/10.5281/zenodo.14980914
- Record: https://zenodo.org/records/14980914
- Dataset licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Related DOI supplied during provenance review: `10.5281/zenodo.14980913`.
  Its relationship to the versioned v1 record has not yet been independently
  verified; retain both identifiers in the release record until confirmed.

### OMersBench / OPoly26 source record

- Dataset: Open Polymers 2026 (OPoly26) Dataset.
- Dataset licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Dataset source: https://huggingface.co/facebook/OMol25
- Pinned source revision: [`47146a3ac4a3451741993a1605ca1b1050c6b9bc`](https://huggingface.co/facebook/OMol25/commit/47146a3ac4a3451741993a1605ca1b1050c6b9bc).
- Bundled release copy: `Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl`.
  This is a PolyBench26-curated source containing a novel property
  contribution, not a byte-for-byte upstream OPoly26 copy. Users reproduce
  the included MD tasks from this file; no separate OPoly26 download is
  required.
- Upstream code licence: MIT License. The CC BY 4.0 dataset licence governs
  the tracked data and its derivatives.
- Citation: Levine, D. S., Liesen, N., Chua, L., Diffenderfer, J., Ingolfsson,
  H., Kroonblawd, M. P., Kumar, N., Maiti, A., Mohottalalage, S. S., Shuaibi,
  M., Van Essen, B., Wood, B. M., Zitnick, C. L., Blau, S. M., & Antoniuk,
  E. R. (2025). *The Open Polymers 2026 (OPoly26) Dataset and Evaluations*.
  arXiv:2512.23117 [physics.chem-ph]. https://arxiv.org/abs/2512.23117

#### PolyBench26 curation and verification record (2026-09-16)

`Cleaned_OMersBench_v3_final.jsonl` is the canonical public source for the
PolyBench26 MD tasks. It contains 82,990 records: 65,221 alternating
copolymers, 10,379 random copolymers, and 7,390 homopolymers. Cp, Rg, and
density are present for every record; refractive index is present for 79,063
records; and the PolyBench26 Cv contribution is present for 1,875 records.

- SHA-256: `7e8a175866cc7166e589fbca1bb125bdece31c398b654a8848f5410f2c68567e`
- The bundled file is the reproducibility boundary for the public benchmark:
  it is sufficient to regenerate the released PSMILES, wPSMILES, descriptor,
  and scaling inputs. It is not intended to reconstruct the excluded raw
  heat-capacity calculation inputs used to curate the Cv contribution.
- The pinned OPoly26 revision above establishes provenance for the underlying
  OPoly26 inputs. It must not be used to claim that the curated `v3_final`
  file is checksum-identical to upstream.

### Coley / VIPEA source record

- Citation: Aldeghi, M., & Coley, C. W. (2022). A graph representation of
  molecular ensembles for polymer property prediction. *Chemical Science*,
  *13*(35), 10486–10498. https://doi.org/10.1039/D2SC02839E
- Upstream release: https://github.com/coleygroup/polymer-chemprop-data
  (MIT License).
- Pinned upstream commit: [`2af92df9560c9f1caf6ddc8a71e2290768a71674`](https://github.com/coleygroup/polymer-chemprop-data/commit/2af92df9560c9f1caf6ddc8a71e2290768a71674).
  This avoids relying on the mutable default branch of a repository reportedly
  not updated in approximately four years.

#### PolyBench26 verification record (2026-09-16)

Both bundled files were verified as byte-identical to the corresponding files
at the pinned commit. Their local Git object IDs equal the upstream object IDs.

| Bundled file | Upstream path | Git object ID | SHA-256 |
|---|---|---|---|
| `dataset.csv` | `vipea/dataset.csv` | `31c273d32327cbb805145c0881c98b6945200bdd` | `6f9d4d8c0cb8154a2150235569ba71f116f9bda1417303e19184e46d64a8bc46` |
| `dataset-poly_chemprop.csv` | `vipea/chemprop_inputs/dataset-poly_chemprop.csv` | `3ab09c7078716a9177ca5ce770f3a4863445994c` | `12a826c77a77537ad39be14feae64194b6847836de1d353a147c273280ad06f7` |

### polyVERSE source record

- Citation: Ramprasad et al. (2024). *polyVERSE: Informatics-Ready Polymer
  Datasets* (Version 1.0) [Dataset]. Zenodo.
  https://doi.org/10.5281/zenodo.13352644
- Record: https://zenodo.org/records/13352644
- Rights holder and required notice: Copyright 2022, Georgia Tech Research
  Corporation, Atlanta, Georgia 30332-4024. All Rights Reserved.
- Licence: [GTRC General Public Use License Agreement](THIRD_PARTY_LICENSES/polyVERSE-GTRC-GENERAL-PUBLIC-USE-LICENSE.txt).
  The supplied licence defines the Program to include code, data, and
  accompanying documentation.
- Release implication: the polyVERSE source CSVs and any distributed covered
  derivatives are **not** licensed under this repository's MIT licence. Keep
  the GTRC notice and full GTRC licence with them; distribute covered works at
  no charge and under the same GTRC terms. Do not sell them for commercial
  gain without a separate GTRC agreement.

#### PolyBench26 verification record (2026-09-16)

The two bundled compact inputs were verified against a clean clone of the
official `https://github.com/Ramprasad-Group/polyVERSE.git` repository at
commit `e6b3f32832d4bf27e67407f0a22fafd2a20284f2`. Both upstream files were
last changed in commit `3b50f1d3980777e7f6005ec295d70a5321221b43`.

| Property | Official source file SHA-256 | Bundled compact file SHA-256 |
|---|---|---|
| Electron affinity | `d4b83bbc1f2d84499d7dc82fd4ff513328ab3b0d971604e410a59da536436d59` | `2013c4a07d87aae846a20869082f97e83ba63330dd81c84aac820d14a38c1ba1` |
| Ionization energy | `cbf732641124d839e8595ddbc982d488479d86487dd3ea6f07a9b44f424159b3` | `e74172105c244eba2ce5fe0ff4f5945d3a4d11b1288bde7e57f9f48f6455c12f` |

The bundled files intentionally differ in bytes from the upstream tables:
they retain only the PSMILES and target-value columns, use compact `*`
attachment notation, and omit source metadata and original row ordering. RDKit
canonicalization established a one-to-one match for every record (368 EA and
370 IP), with every target value equal. The tracked GTRC licence matches the
official repository's licence text, differing only by its final newline.

The official GitHub repository is the direct source used for the bundled
polyVERSE inputs. The Zenodo record remains the dataset citation; no separate
Zenodo download is required to establish the provenance of these files.

## Derived artifacts

The following files are generated from the source inputs. Their public status
depends on the terms above and must be confirmed before publication.

| Artifact | Generation path | Git policy |
|---|---|---|
| Canonical PSMILES (MD and Coley-derived datasets) | `generate_basic_datasets.sh` | Generated; only compact source inputs are tracked. |
| wPSMILES | `generate_basic_datasets.sh` / `convert_web_datasets.py` | Generated and ignored. |
| RDKit descriptors | `generate_rdkit_datasets.sh` or the scaling splitter | Generated and ignored. |
| polyBERT embeddings | `generate_polybert_datasets.sh` using an external model | Generated and ignored. |
| Shared scaling splits | `generate_scaling_datasets.sh` | Generated and ignored; canonical compact assignments are tracked under `Datasets/scaling_indices/`. |
| Compact result summaries | `scripts/collect_scaling_results.sh` | Intended for publication after result provenance and completeness are verified. |

## Restricted and excluded data

PoLyInfo is not included in the public source inventory and must not be
redistributed through this repository. The generation and ignore rules exclude
PoLyInfo directories and derivatives. Any future source with restricted terms
must receive the same treatment unless written redistribution permission is
recorded.

## Release verification

Before release, complete `RELEASE_INFORMATION_NEEDED.md`, then verify that:

1. every tracked source has an authoritative citation and a recorded
   redistribution decision;
2. every generated derivative is consistent with that decision;
3. restricted data is absent from the Git index and release archive; and
4. required source records remain present in `git archive <release-tag>`.
