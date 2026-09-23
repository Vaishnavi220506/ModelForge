# ModelForge Requirements Summary

This note consolidates the supplied ModelForge project brief and the Compiler Design Laboratory instruction manual.

## Project requirement

Build an individual C++17 educational compiler that accepts a supported subset of ONNX feed-forward neural-network models, validates them, lowers them into a custom intermediate representation, applies simple optimizations, generates standalone C++ inference code, and compares generated output with the original ONNX output.

## Must-have scope

- ONNX loading
- Supported-operator detection
- Tensor shape and type validation
- Custom ModelForge IR
- At least two optimization passes
- C++ code generation
- Compilation of generated code
- ONNX versus generated-C++ validation
- Compiler-style diagnostics

## Initial operator subset

`Gemm`, `MatMul`, `Add`, `Relu`, `Sigmoid`, and `Softmax`.

## First demonstration model

Iris classifier: input size 4, dense layer of 8 neurons, ReLU, dense layer of 3 neurons, and Softmax output of 3 classes.

## Explicit boundaries

CNN-heavy architectures, YOLO, Transformers, recurrent networks, dynamic control flow, arbitrary dynamic shapes, GPU generation, and complete ONNX coverage are outside the initial scope.
