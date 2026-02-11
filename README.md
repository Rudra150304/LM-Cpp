C++ Self-Attention Language Model (From Scratch)

This project implements a small character-level language model entirely in C++ using Eigen.

Why?

The goal was to understand how modern deep learning frameworks work under abstraction by implementing everything manually.

The model was built progressively:

Bigram language model

Context window model

Single-head self-attention model

Transformer-style block with:

Residual connections

LayerNorm (forward + backward)

Feed Forward Network (GELU)

Adam optimizer

Gradient clipping

All forward and backward passes are implemented manually without autograd.

Training

Dataset: TinyShakespeare
Block size: 32
Embedding dimension: 32
Optimizer: Adam

Loss evolution observed while upgrading architecture from bigram → context window → self-attention.

Purpose

This repository exists for educational purposes — to understand gradient flow, tensor shapes, and training stability at a low level.
