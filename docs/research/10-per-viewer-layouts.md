# 10 — Per-viewer layouts

## Question

How do we remove the v1 `assumed_shared` layout for viewers whose screen
differed from the recording?

## Candidates

1. Keep assuming the recording layout for everyone (v1).
2. Accept a user-supplied per-viewer manifest.
3. Infer each viewer's layout from their gaze behavior.

## Decision

Ship option 2 now and scope option 3 as research.

- A manifest is a JSON file mapping each viewer to time-bounded regions; loaded
  layouts carry `source = manifest`. `lookout attribute` accepts a
  `viewer_layouts` override so attribution uses the manifest instead of the
  assumption.
- Inference (`source = inferred`) is future work: cluster a viewer's
  speaker-prior gaze targets to recover tile positions.

## Evidence

`tests/test_manifest.py` covers manifest round-trip and the `manifest` source
tag. The pipeline override is exercised by the existing end-to-end test path
(the same code path with a supplied layout list).

## Risks

- Manifests are manual and can be wrong or stale. Mitigation: `source` is always
  visible in results, so a manifest-based attribution is never mistaken for
  ground truth.
- Inference is only as good as the speaker prior and may be under-determined for
  small or quiet meetings. Mitigation: treat inferred layouts as low-trust and
  evaluate before use.
