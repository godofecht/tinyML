# Demo Roadmap: TinyML Framework as Configurable System

Goal: Present a single framework where forecasting, attention, physics constraints, uncertainty, graphs, generation, and control are configurations of shared primitives. Demos are proofs of subsumption, not isolated features.

## Phase 0: Foundations (1-2 weeks)
- Stabilize playground charts (loss + latency) for all models and scenarios.
- Ensure all demos run offline (no external CDN dependency).
- Add minimal model handlers so every UI model can run without error.
- Add scenario metrics for loss/uncertainty so charts update during loops.

## Phase 1: Uncertainty as First-Class (2-3 weeks)
- Probabilistic Forecast Demo
  - One model, one dataset, live density band + quantiles.
  - Regime shift toggle: uncertainty widens, mean adapts.
  - UI signals: predictive density, p10/p50/p90, drift annotation.
- Bayesian Behavior Demo (no Bayesian label)
  - Regression with epistemic vs aleatoric separation.
  - OOD region shows increased epistemic uncertainty.

## Phase 2: Interaction Kernels (2-3 weeks)
- Attention-as-Primitive Demo
  - Same model, swap interaction kernel: attention vs alternative (e.g., gaussian or linear kernel).
  - UI toggle with identical head count and layer count.
  - Signal: attention is just one routing option.

## Phase 3: Constraints as Operators (2-3 weeks)
- Physics-Informed Demo
  - Same training loop, toggle constraints on/off.
  - UI shows divergence when physics disabled.
  - Metrics: constraint loss vs data loss as separate traces.

## Phase 4: Graphs as Data Layout (2-3 weeks)
- Spatiotemporal Demo
  - Switch between grid, graph, and continuous layout.
  - Same model definition and training loop.
  - UI signals: layout switch, same model params.

## Phase 5: Generation as Field Completion (2-3 weeks)
- Conditional Generation Demo
  - Condition on partial future; generate consistent futures.
  - Show multiple samples as scenario bands.
  - Signal: generator is a simulator, not a sampler.

## Phase 6: Control as Unified Head (2-3 weeks)
- RL / Control Demo
  - Battery or storage control using same predictive core.
  - Policy is a head on shared latent state.
  - UI: forecast + control policy + cost curve.

## Phase 7: Production Framing (2-3 weeks)
- Energy/Macro Dashboard
  - Forecast + risk bands + scenario stress tests.
  - Policy hooks and exportable reports.
  - Purpose: look boring and production-ready.

## Milestones
- M1: All current demos stable, charts live.
- M2: Uncertainty-first demo polished.
- M3: Kernel toggle + constraints toggle polished.
- M4: Graph layout switch + conditional generation polished.
- M5: Control demo + production dashboard polished.

## Guiding Principle
"I didn’t collect techniques — I built a system where these techniques fall out as configurations."
