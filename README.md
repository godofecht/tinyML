# TinyML: High-Performance C++ Machine Learning Library

![License](https://img.shields.io/badge/license-MIT%20core%20%2B%20commercial-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Standard](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)

**TinyML** is a high-performance, lightweight machine learning library written in modern C++ (17/20). It is designed for educational exploration and production-grade embedded deployment, featuring a zero-dependency core, SIMD optimizations, and a rich interactive playground.

Unlike standard frameworks (PyTorch/TensorFlow) that abstract away the details, TinyML implements algorithms from scratch to demonstrate deep understanding of the underlying mathematics and systems engineering required for high-performance ML.

---

## 🚀 Key Features

### 🧠 Advanced Architectures
*   **Deep Learning**: Custom implementation of Feed-Forward Networks, CNNs (Convolutional), and RNNs.
*   **Transformers**: Real-time streaming Transformer implementation with self-attention mechanisms.
*   **Bayesian Neural Networks (BNN)**: Uncertainty estimation using Monte Carlo Dropout and Variational Inference.
*   **Generative Models**: Variational Autoencoders (VAE) and 2D Generative Adversarial Networks (GAN).
*   **Scientific ML**: Physics-Informed Neural Networks (PINNs) for solving PDEs (e.g., Heat Equation).
*   **Reinforcement Learning**: Policy Gradient and Q-Learning implementations (CartPole, Pong).
*   **Graph Neural Networks (GNN)**: Spatiotemporal modeling for traffic forecasting.

### ⚡ Performance & Systems
*   **SIMD Optimization**: Explicit AVX2/NEON vectorization for core linear algebra operations (`SIMDOperations.h`, `XSIMDOperations.h`).
*   **Quantization**: Support for integer-only inference for edge devices.
*   **Memory Management**: Smart pointer usage and custom memory pools for minimal overhead.
*   **Thread Safety**: Thread-safe model serving infrastructure (`ModelManager`).

### 🎮 Interactive Playground
A built-in web-based dashboard to visualize training and inference in real-time.
*   **Real-time Visualization**: Canvas-based rendering of environments (CartPole, Pong) and internal model states (Attention maps, CNN filters).
*   **Interactive Scenarios**: 7+ scenarios covering different ML domains.
*   **Live Training**: "Loop Train" functionality to watch models learn.

---

## 🛠️ Installation & Build

### Prerequisites
*   CMake (3.15+)
*   C++ Compiler (GCC 9+, Clang 10+, MSVC 2019+)
*   Make or Ninja

### Build Instructions
```bash
git clone https://github.com/your-username/tinyML.git
cd tinyML
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 🖥️ Running the Playground

The playground consists of a C++ backend server and a vanilla JS/HTML frontend.

1.  **Start the Server**:
    ```bash
    ./bin/PlaygroundServer
    ```
    *Server listens on port 8081.*

2.  **Access the Dashboard**:
    Open `http://localhost:8081` in your browser.

3.  **Explore Scenarios**:
    *   **CartPole (RL)**: Watch an agent balance a pole.
    *   **Pong (RL)**: AI agent playing against a heuristic opponent.
    *   **CNN**: Visual convolution operations.
    *   **Heat Equation (PINN)**: Solving partial differential equations.
    *   **Traffic (GNN)**: Graph diffusion simulation.

---

## 📂 Project Structure

```
tinyML/
├── include/              # Core Library Headers
│   ├── Network.h         # Base Neural Network abstractions
│   ├── RealTimeTransformer.h # Transformer implementation
│   ├── SIMDOperations.h  # AVX/NEON optimizations
│   └── ...
├── src/                  # Library Implementation
├── playground/           # Interactive Web Dashboard
│   ├── server.cpp        # HTTP Server (using httplib)
│   ├── ModelManager.h    # Thread-safe model orchestration
│   ├── Scenarios.h       # Scenario logic (Pong, CartPole, etc.)
│   └── script.js         # Frontend visualization logic
├── benchmarks/           # Performance benchmarks
├── tests/                # GoogleTest suite
└── blog/                 # Detailed architectural documentation
```

---

## 🧪 Testing & Benchmarks

The project maintains a high standard of correctness through a comprehensive test suite.

```bash
# Run Unit Tests
cd build
./TinyMLTests

# Run Benchmarks
./SIMDBenchmark
```

---

## 📚 Documentation
Detailed design documents and phase breakdowns can be found in the `blog/` directory, covering topics from "SIMD Optimization Foundation" to "Reinforcement Learning".

---

## 📝 License

tinyML is dual licensed. The zero-dependency core is MIT, free for any use
including commercial. The extended model library (attention, transformers,
generative, graph, RL, forecasting, production API) requires xsimd and is
covered by a commercial license: free for research, education and personal
projects, paid for commercial use.

[LICENSING.md](LICENSING.md) lists exactly which headers fall on each side.
No MIT header includes a commercial one, so the core builds and ships alone.
