# Experiments

Throwaway scripts that produce the evidence behind research notes: resolution
sweeps, parameter searches, model comparisons.

Rules:

- Nothing in `src/lookout` may import from here.
- Outputs (plots, dumps, downloaded models, recordings) are gitignored. Commit
  the script and record the resulting numbers in the matching `docs/research`
  note, not the artifacts.
- Scripts may depend on optional extras (`face`, `appearance`, `ocr`). Guard
  those imports so a missing extra fails with a clear message.
