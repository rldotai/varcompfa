

# Comparison of Variance Estimation Algorithm

Codebase for the variance estimation algorithms we wish to compare.

# Installation

```bash
pip install --editable .
```

# Examples

Look at `examples/basic_example.py` for an illustration of how to run a simple experiment.

# Documentation

The code is documented inline, although you can generate HTML/LaTeX/whatever else Sphinx supports if desired.

For example, to generate HTML documentation:

```bash
cd docs
make html
```

Or, with setuptools:

```
python setup.py build_sphinx
```

It's not perfectly well-formatted because I tend to use documentation conventions influenced by Markdown, which causes some conflicts with the `sphinx-napoleon` extension that parses the docstrings.

# Remarks and Further Details

## Codebase Organization

The code that implements the algorithms and runs the experiments is located in `varcompfa`, which is set up as a python package.
The algorithms are stored in `varcompfa/algos`, custom environments in `varcompfa/envs`, and so on.

Setting it up this way entails a little bit of extra work when modifying the codebase, but is worth it because it ensures consistency (and therefore reproducibility).

To add a new algorithm, for example, you write the code (say in `varcompfa/algos/my_algo.py`), then modify the files `varcompfa/algos/__init__.py` and `varcompfa/__init__.py` to ensure that it is imported as desired.

We use the OpenAI Gym insofar as possible to avoid reimplementing the wheel and also to enhance reproducibility.

Ultimately, each experiment should be specified as a `.py` file in `experiments`, following the same sort of structure as in `examples/basic_example.py`.

Things that aren't suitable for prime-time should be worked on in the `sandbox` directory, and then moved to other locations once they're ready.
The sandbox is protected from accidental commits via the `.gitignore` in the root directory.

## Fancy Coding

There are a few places where the code is not as straightforward as I'd like-- for example, using abstract base classes that feature functions inherit from.
I view these things as necessary in order to ensure that certain methods are actually implemented.
Otherwise, things may not work as expected, and it's better to know this as soon as possible rather than via an obscure bug that crops up near submission time.
