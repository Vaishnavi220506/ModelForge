# Final Validation Evidence

Store validation logs, output comparisons, screenshots, benchmark notes, and test summaries here. Every reported result should include the model, input, expected output, actual output, tolerance, and pass or fail status.

The executable pipeline provides a dependency-free first validation using `models/iris_demo.mforge`. The `--verify-onnx` option records the same comparison against ONNX Runtime when the optional runtime build is enabled.
