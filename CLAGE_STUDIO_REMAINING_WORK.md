# Clage Studio remaining work

The delivered application covers usable vertical slices rather than pretending
the entire fifteen-phase vision is complete. These are not hidden placeholder
controls: unimplemented operations are absent or explicitly unavailable.

## Beyond the scoped Studio 1.0 release

These are requirements for the broader research-platform roadmap, not unresolved
release blockers for the documented single-run Studio 1.0 scope. The final review
in `CLAGE_STUDIO_FINAL_REVIEW.md` assesses that scope separately. In particular,
this release does not promise full archival recording, resumable checkpoints or
simultaneous live experiments.

1. **Full archival recording:** append-only compressed frame chunks, indexing,
   selective topology loading and seeks across entire long runs. Current recordings
   intentionally drop old frames; all-genotype metadata and frame copies can still
   consume substantial Python memory despite the encoded-frame budget.
2. **Complete checkpoints:** serialize RNG, population/species/innovation ledger,
   best-champion state, pending generation evaluation and environment/body state;
   test branch/resume equivalence. Current replay is never a checkpoint.
3. **Experiment execution manager:** process-isolated queue, cancellation and
   simultaneous live runs with consistent experiment IDs. Current browser supports
   one live run and side-by-side replay, not two synchronized live engines.
4. **Multi-run scientific analytics:** named experiment registry, multi-training-seed
   traces, paired evaluations/confidence intervals, survival curves, species lifecycle
   charts and full behavioral-diversity/action histories. Existing batch pipeline
   remains useful but is not yet fully integrated into Studio.
5. **Versioned controlled ablations:** facing-aware sensing with equal-information
   baselines, reproduction eligibility, explicit alternative fitness/initialization
   presets with frozen definitions. Dense/minimal initialization already has distinct
   labels; no new scientific claim is made for either.

## Product and visualization follow-up

- Large-history lineage search/filter/branch expansion, structural innovation
  timeline and exact mutation operator event log (currently recorded net deltas).
- Per-body historical behavior timeline beyond retained frames/sampled trails.
- Rich species/fitness/complexity charts and multiple-experiment graph overlays.
- Complete event log for animations; present pulses observe stream differences and
  can miss intermediate events. No invented full-tick motion interpolation.
- Sandbox world editor with explicit frozen-experiment/interactive-sandbox modes.
- GIF/video export, annotated demonstration recording and portable preset files.
- More accessible graph/world navigation, screen-reader equivalents, touch QA,
  desktop Safari/Firefox tests, and systematic intermediate responsive sizes.

## Performance/reliability follow-up

- Process-based orchestration to remove GIL contention; measured message protocol
  optimization/deltas, backpressure and multiple-client load tests.
- Profile density layers, rich SVG genomes, 50,000-body safety limit, full-generation
  reproduction, >8-generation soak runs and long replay imports on lower-end machines.
- The recorded 16 MiB limit is JSON size, not a hard process-RSS bound. Add total
  heap accounting and optional topology/frame disk eviction before relaxing limits.
- Indexed world fields require mutation through supported methods; a public
  encapsulated grid API would make misuse harder for third-party extensions.
- Offline evaluation still caps policy trials to 300 ticks and lacks cancellation/
  explicit body budget; use conservative configurations for degenerate reproduction.
- Broader environment/package-version reproducibility, pinned scientific toolchain,
  Windows validation utilities, remote CI run and packaged-install E2E verification.
- Recovery of the missing `bec8c03a...` audit requires its original workspace or
  bundle. This delivery cannot recover unavailable source objects.

## Review and release

Review `studio/clage-1.0` without merging main. No default-branch push/merge or PR
is claimed. Retain the source commits, measured workload JSON and limitations when
publishing benchmark numbers. Choose a package/release version only after agreeing
on the remaining scope; the original engine version is deliberately still 0.3.0.
