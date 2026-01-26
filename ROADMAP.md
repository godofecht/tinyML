# TinyML Project Roadmap & Todo List

## Phase 1: Core Architecture & Reliability (Completed/In Progress)
- [x] **Refactor Server State**: Replace global variables with thread-safe `ModelManager` singleton.
- [x] **Thread Safety**: Implement `std::mutex` locking for all model access and scenario updates.
- [x] **Lazy Initialization**: Models and scenarios are created on-demand to save resources.
- [ ] **Numeric Precision Unification**: Migrate codebase to consistent precision (template-based or unified `float32`). Currently `Network` uses `double` while SIMD operations use `float`.
- [ ] **Autograd System**: Replace manual backpropagation logic with a proper `Tensor` class supporting automatic differentiation (similar to PyTorch).

## Phase 2: Web-Deployable Demos (Portfolio Readiness)
### Implemented
- [x] **Probabilistic Forecasting**: 
  - `ForecastScenario` generates P10/P90 quantile bands.
  - Visualization of uncertainty intervals in `script.js`.
- [x] **Attention Visualization**: 
  - `AttentionScenario` implements time-series attention mechanism.
  - Interactive heatmap and query-key connection visualization.

### Planned Demos (To Implement)
- [ ] **Physics-Informed Neural Networks (PINN)**: 
  - Enhance `HeatEquationScenario` to toggle between "pure NN" and "physics-informed" modes.
  - Visualize stability differences in real-time.
- [ ] **Interactive Bayesian Regression**: 
  - Upgrade `BayesianScenario` to allow dragging training points.
  - Visualize predictive variance reacting to data density in real-time.
- [ ] **Spatiotemporal Graph Modeling**: 
  - Implement a grid/network map scenario (e.g., energy grid).
  - Animate demand shocks propagating through the graph.
- [ ] **Conditional Generative Models**: 
  - Enhance `GenerativeScenario` to support conditional generation (e.g., demand curves conditioned on temperature).
- [ ] **Reinforcement Learning (Energy Trading)**: 
  - Create a battery arbitrage environment (charge/discharge/hold based on price).
  - Visualize policy evolution and cumulative reward.

## Phase 3: Developer Experience & DevOps
- [ ] **Documentation**:
  - `Architecture.md`: High-level system design.
  - `API.md`: HTTP endpoint documentation (`/run`, `/train`, `/scenario/*`).
- [ ] **Frontend Modernization**:
  - Refactor `script.js` into ES6 modules (API layer, Charting layer, UI logic).
  - Consider moving to a lightweight framework (React/Vue) if complexity grows.
- [ ] **Hyperparameter Tuning UI**:
  - Add controls for Learning Rate, Batch Size, and Optimizer selection in the web interface.
- [ ] **CI/CD & Deployment**:
  - Add `Dockerfile` for containerized deployment.
  - Set up GitHub Actions for automated testing and build verification.

## Phase 4: Advanced Features
- [ ] **Rolling-Origin Evaluation**: Add backtesting capability to the forecasting playground.
- [ ] **Model Export/Import**: Allow saving trained weights to JSON/binary and loading them back.
- [ ] **WebAssembly (WASM) Port**: Compile the C++ core to WASM to run inference entirely in the browser (client-side).
