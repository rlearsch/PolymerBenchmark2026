# Contributing to PolyBench26

Contributions are welcome. For substantial changes to benchmark protocol,
datasets, representations, or reported results, please open an issue first so
the proposed change can be discussed before implementation.

## Before opening a pull request

1. Keep changes focused and explain their purpose.
2. Preserve the shared scaling protocol and its canonical split indices.
3. Do not add generated datasets, model checkpoints, embeddings, credentials,
   or other large artifacts. Follow the data-provenance and redistribution
   requirements in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
4. Update relevant documentation and tests with code or workflow changes.
5. Run the repository validation suite from the repository root:

   ```bash
   bash scripts/test_every_commit.sh
   ```

## Reporting bugs and requesting features

Use GitHub Issues for reproducible bugs, documentation corrections, and
feature requests. Include the repository commit, operating system, Python
version, commands run, and relevant error output when reporting a bug.

Please do not report security vulnerabilities in a public issue; see
[`SECURITY.md`](SECURITY.md).

## Data and results

Contributions that affect source data, derived representations, split indices,
or published results require particular care. Preserve row identity and target
alignment, do not create model-specific splits for cross-representation
comparisons, and retain enough provenance for others to reproduce the change.
