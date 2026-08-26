# Changelog

All notable release-facing changes to tinyML are recorded here.

The stable core follows semantic versioning. Preview and source-only modules are identified in `docs/STABILITY.md`.

## Unreleased

### Added

- Installable `TinyML::Core` and `TinyML::Extended` CMake package targets.
- Component-aware `find_package(TinyML)` support.
- Dependency-free core build mode.
- Generated `<tinyml/version.hpp>` version macros.
- Stable `<tinyml/core.hpp>` and extended `<tinyml/tinyml.hpp>` umbrella headers.
- Downstream install/consume smoke test.
- Explicit API stability policy.

### Changed

- Replaced source globbing with explicit, reviewable source lists.
- Made xsimd discovery/fetch conditional on the extended target.
- Made GoogleTest conditional on tests or benchmark builds.
- Made benchmarks, examples and the playground opt-in.
- Replaced global optimization/compiler flags with target-local configuration.
- Release packaging now uses the CMake install graph.
- Core network inputs, targets and weight vectors now reject invalid dimensions with exceptions instead of relying on assertions or unchecked indexing.
- Network bias neurons use a conventional constant value of `1.0`.
- Weight initialization no longer mutates process-global `rand()` state.

### Fixed

- Initialized connection momentum state before the first training update.
- Removed the hard-coded `101.0` divisor and unchecked connection indexing from weight normalization.
- Corrected `Network::updateWeights()` so updates are applied through the destination layer rather than indexing each previous-layer neuron against itself.
- Added exact weight-count validation before replacing model weights.
