<!-- 
MIT License

Copyright (c) 2023 Carnegie Mellon University, Auton Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Contributing

We'd love to hear from you! Feel free to open an issue or a PR if you have any suggestions or find any bugs.

When contributing, please add tests to the `tests` directory. Additionally, we follow [`flake8`](https://flake8.pycqa.org/en/latest/) as a linter and [`pytest`](https://docs.pytest.org/en/7.3.x/) for testing. Please lint your code before submitting a pull request to maintain design consistency. For documentation, please write docstrings following the [Google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). We use a variant of this style guide for docstrings as shown below:

```python
def mean_tf(x: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[float, np.float_]:
    """
    Compute the mean of the values in `x` where `where` is `True`.

    Args:
        x: The array to compute the mean of.

        where: A function that takes a value and returns `True` or `False`. Default is `lambda x: not np.isnan(x)` i.e. a measurement is valid if it is not a `NaN` value.

    Returns:
        The mean of the values in `x` where `where` is `True`.
    """
    # Vectorize where fn
    where_fn = np.vectorize(pyfunc=where)
    return np.mean(x, where=where_fn(x))
```

We also encourage using type annotations where possible. For example, the `mean_tf` function above has type annotations for the input and output types.

The following commands can be run for verficiation before opening a PR:

```bash
# Unit tests
python -m pytest tests

# Linting
flake8 autonfeat --ignore=E501
```

If you enjoy using [`AutonFeat`](../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.