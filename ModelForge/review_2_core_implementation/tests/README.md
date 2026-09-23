# Review 2 Tests

`test_runner.cpp` is a dependency-free test executable. It covers the complete manifest pipeline and can be run through CTest.

The fixtures cover:

- A valid Iris-style dense network
- Unsupported operators
- Missing tensors
- Duplicate output tensors
- Matrix shape mismatch
- Invalid Softmax axes
- Constant folding
- Dense-plus-ReLU fusion
- Dead-node removal
- IR execution and numerical output comparison

The test runner deliberately uses simple checks instead of a third-party unit-test framework so the compiler project stays easy to build and explain during the viva.
