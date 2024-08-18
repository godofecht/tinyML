# TinyML Perceptron Library

This repository contains a simple implementation of a Perceptron model for solving binary classification problems. The implementation includes training, evaluation, and various tests to ensure the correctness of the network, including checking the transfer function and ensuring proper training convergence. This project is focused on tiny machine learning (TinyML), targeting CPU-efficient models for simple tasks.

The original motivation of this repository is to add foundational real-time support for small machine learning models that are more suited for solving data related problems, such as parameter mappings, linear regression, and more.

## Features

- **Perceptron Implementation**: A basic neural network model with a single-layer perceptron that can be trained for binary classification tasks.
- **Examples**: Demonstrates how the Perceptron class can be used to solve classic logic gate problem, which is a fundamental binary classification task.
- **Extensive Testing**: Includes tests to verify the network's weight initialization, feedforward mechanism, transfer function correctness, and training convergence.
- **CPU Efficient**: The perceptron is designed to be lightweight and CPU-efficient, making it suitable for small ML applications.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/godofecht/tinyML.git
   ```
2. Navigate to the project directory:
   ```bash
   cd tinyML
   ```
3. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

### Training the Perceptron
The `Perceptron` class allows you to define the network's topology and train it on binary classification tasks. Here's an example of training the perceptron on a NAND gate problem:

```cpp
std::vector<unsigned> topology = {2, 2, 1};
ML::Models::Perceptron perceptron (topology);

std::vector<std::pair<std::vector<double>, std::vector<double>>> nandTrainingSet = {
    {{0, 0}, {1}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {0}}
};

for (int i = 0; i < 10000; ++i) 
{
    for (const auto &[input, output] : nandTrainingSet) 
    {
        perceptron.feedForward(input);
        perceptron.learnSupervised(output);
    }
}
```

### Evaluating the Perceptron
Once trained, the perceptron can be evaluated on new inputs:

```cpp
auto output = perceptron.process({0, 0});
std::cout << "Output for (0, 0): " << output[0] << std::endl;
```

### Running Tests
The project includes several unit tests to ensure that the perceptron implementation is working correctly. The tests are implemented using Google Test.

To run the tests:

1. Build the test suite:
   ```bash
   cd build
   make
   ```
2. Run the tests:
   ```bash
   ./TinyMLTests
   ```

## Tests

The following tests are included:

1. **Initial Weights Sanity Check**: Verifies that the perceptron initializes weights correctly.
2. **Feedforward Check**: Ensures that the feedforward pass produces valid output.
3. **Transfer Function Test**: Validates that the transfer function works as expected for a range of inputs.
4. **Training Convergence**: Ensures that the perceptron can learn to solve the NAND function after training.
5. **Gradient Check**: Verifies that backpropagation computes gradients correctly using finite difference approximations.

## Contributing

Feel free to submit pull requests, report issues, or suggest improvements. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
