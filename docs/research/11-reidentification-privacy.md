# 11 — Face re-identification: privacy review

## Question

Linking participants across layout changes without name labels needs face
similarity, which is biometric. Under what constraints, if any, is this
acceptable in a privacy-conscious, local-first system?

## Candidates

1. Do not offer it.
2. Offer it always-on.
3. Offer it opt-in, in memory only, off by default.

## Decision

Option 3, with hard constraints encoded in code:

- Disabled by default; `ReIdentifier` raises unless constructed with
  `enabled=True`.
- Embeddings are kept in memory for the duration of a run and never written to
  disk; `reset` clears them.
- No embedding is stored in any artifact; only the resulting opaque ids appear.
- It links identities within one meeting; it is not an identity database and
  must not be used for cross-meeting tracking or any covert monitoring.

## Evidence

`tests/test_reid.py`: the identifier refuses to run when disabled, links the
same face and separates different faces when enabled, and starts ids over after
`reset`. A fake embedder is used; the real embedder is an optional adapter.

## Risks

- Biometric misuse. Mitigation: the constraints above, opt-in gating, and the
  project's stated prohibition on covert monitoring. Any change that persists
  embeddings or enables cross-meeting linkage must revisit this review first.
