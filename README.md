# Gradient_Engine
Introduction to Gradient Engine
# Gradient Engine

![1058467](https://github.com/PrateekJannu/Gradient_Engine/assets/71490386/6f14cb8e-5119-4a66-a31a-51a62d38ef28)


This is a compact Autograd engine (with a bit of a bite! :)). It implements backpropagation (reverse-mode autodiff) over a dynamically built Directed Acyclic Graph (DAG). Additionally, there's a small neural networks library on top of it with a PyTorch-like API. Both components are remarkably compact, with approximately 100 and 50 lines of code, respectively. The DAG specifically operates over scalar values, breaking down each neuron into individual tiny additions and multiplications. Surprisingly, this simplicity is sufficient to construct entire deep neural networks for binary classification. This project may prove useful for small to mid-size projects.



### Example Usage

Here's a somewhat contrived example showcasing various supported operations:

```python
from source import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')  # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}')  # prints 138.8338, i.e., the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints 645.5773, i.e., the numerical value of dg/db
```

### Running Tests

To run the unit tests, you will need to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. After installation, simply run:

```bash
python -m pytest
```

Credits: Andrej Karpathy.
