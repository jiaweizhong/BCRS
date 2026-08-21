# hesod/ — HESOD method umbrella

This directory holds the actual implementation of **HESOD** (see `../HESOD-Proposal.md` at the
repo root for the authoritative design). It is not a single codebase — it's an umbrella over
one backend fork per framework HESOD is validated on, per the Proposal's §5.2 "BCRS-Core +
Backend Adapters" architecture (ESOD primary, QueryDet cross-framework validation, CEASC
optional extension; sequencing gated by §11.2).

```
hesod/
  backends/
    esod/         # HESOD's ESOD-based implementation (primary); this is where the
                   # dual-evidence selector, new losses, and budget router described
                   # in the Proposal actually get built. The standalone pristine
                   # baseline checkout that used to live at ../esod/ (repo root) was
                   # retired 2026-08-21 once confirmed byte-identical and unreferenced
                   # elsewhere -- this is now the sole baseline copy, see
                   # ../ESOD-Baseline-Patches.md.
    querydet/      # not created yet -- Phase 4 only (Proposal §11.2)
    ceasc/         # not created yet -- Phase 6 only, optional (Proposal §11.2)
```

## No shared `core/` yet — on purpose

The Proposal's shared selector core (semantic branch, spectral branch, evidence fusion,
budget router) is *designed* to be reused across backends, but with only one backend
(`backends/esod/`) actually implemented, there isn't yet a second real usage to validate the
right abstraction boundary against. Extracting a `hesod/core/` module before that risks baking
in ESOD-specific assumptions (patch-level candidates, ESOD's specific FPN layout) into what's
supposed to be a backend-agnostic interface. Extract it when `backends/querydet/` starts
(Phase 4) and genuinely needs to reuse the same fusion/router logic — not before.

## Environment compatibility

`../ESOD-Baseline-Patches.md` at the repo root is the single authority for
environment-compat patches on `backends/esod/` — any new environment issue
gets fixed and documented once there.

## Relationship to `BCRS/`

`BCRS/vendor/esod` contains an earlier, unofficial draft of dual-evidence selector ideas
(`DualEvidenceSegmenter`, `SpectralBranch`, `GatedEvidenceFusion`, etc.) built before this
directory existed. It is not migrated, ported, or kept in sync with `hesod/` — it's left
untouched and may be consulted purely as an informal reference for implementation patterns.
`HESOD-Proposal.md` is the actual design source for everything under `backends/`.
