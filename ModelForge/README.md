# ModelForge

ModelForge is an individual Compiler Design Laboratory project that translates a supported subset of ONNX feed-forward neural-network models into readable standalone C++ inference code.

## Review folders

- `review_1_proposal_design/` - project proposal, architecture, scope, methodology, and initial prototype.
- `review_2_core_implementation/` - planned loader, validator, IR, optimizer, and code-generation implementation.
- `review_3_final_integration/` - final integration, correctness validation, testing evidence, and report material.
- `docs/` - shared project decisions, requirements, and development notes.

## Current status

The research-enhanced version adds graph-aware translation validation, auditable optimization events, and an optimization_certificate.json report. The research contribution and evaluation plan are documented in docs/research_contribution.md.

Review 1 material is prepared, and the Review 2 core implementation is now available under `review_2_core_implementation/`. The final submission scope is the dependency-free `.mforge` demonstration: the five-module C++17 pipeline supports model loading, semantic validation, a tensor symbol table, ModelForge IR, optimization passes, generated standalone C++, generated-program execution, and automated tests. The optional ONNX protobuf loader remains in the source tree but is not required for this basic submission.

## Planned technology

- C++17 for the compiler implementation
- Dependency-free `.mforge` model manifests for the basic submission
- ONNX and Protocol Buffers only for the optional future loader
- CMake for builds
- Built-in IR/generated-program verification for the basic submission
- Optional Graphviz or textual output for visualization

## Academic integrity

This project is an individual implementation. Every module, test case, and design decision must be understood, modified, tested, and explained by the student during the viva.
