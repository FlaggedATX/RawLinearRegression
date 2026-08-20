# RawLinearRegression

A **linear regression model built from scratch in Python**. This is a learning project focused on understanding how linear regression works internally rather than using a machine learning library.

## What it does

The model implements **n-feature linear regression**:

```text
ŷ = W₁X₁ + W₂X₂ + ... + WₙXₙ + b
```

or equivalently, in vector form:

```text
ŷ = W·X + b
```

Training is done manually using **gradient descent** and **Mean Squared Error (MSE)**. For a general feature `Xᵢ`, the gradients are:

```text
∂MSE/∂Wᵢ = (2/n) Σ(ŷ - y)Xᵢ
∂MSE/∂b  = (2/n) Σ(ŷ - y)
```

No machine learning frameworks are used. NumPy is currently only used for generating datasets and performing vectorized weight/gradient operations.

## Current Experiment

I'm currently teaching the model to learn addition:

```text
Y = X₁ + X₂
```

For this relationship, the ideal parameters are:

```text
W₁ ≈ 1
W₂ ≈ 1
b  ≈ 0
```

The goal is to watch the model discover these values through gradient descent.

## What's Next?

I plan to use the same model to experiment with different mathematical operations, at varying feature counts, and see where linear regression succeeds and where it breaks down. Some planned experiments:

* Addition (generalized to bigger numbers and more terms)
* Multiplication
* Binary arithmetic

## Project Structure

```text
RawLinearRegression/
├── model.py
├── dataset.py
├── main.py
└── README.md
```

## Status

**Work in progress**

This is a learning project. The focus is on understanding the mathematics and mechanics of linear regression from the ground up.

## Requirements

* Python 3
* NumPy
