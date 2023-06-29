# ⏳ AutoFeat ⌛

A high performance library for time series featurization. 

## What? 🙋

AutoFeat is a high-performant domain agnostic package for time series featurization. We've included some domain specific featurizers for popular domains like healthcare. With time series data, it is often helpful to explore features in more than just the time domain. We have provided a number of [preprocessors](api/preprocess/preprocess.md) that can transform the distribution or space to a form more amenable to certain featurizations. The package is lightweight, fast and easy to use.

## Why? 🤔

***(Coming soon)***

## Design Objectives 🎯

- **Simple**: The package must be easy to use and require as little user input as possible.
- **Interpretable**: The software abstractions must be intuitive, easy to understand and easy to debug.
- **Fast**: The tool must be fast enough to be used in large scale production environments.
- **Flexible**: The package must be modular and allow for easy extensibility to leverage the community.

## Assumptions 🧐

<div class="containerassumputions" style="vertical-align: middle;">
    <div class="titleassumptions" style="background-color: #ffcccb;border-radius: 10px;display: table-cell;">
        <ul>
            <li>Time series data is univariate.</li>
            <li>Data is provided as numpy arrays.</li>
            <li>Missing values are represented as `np.nan`.</li>
        </ul>
    </div>
</div>


## Installation 📦

***(Coming soon)***
```bash
pip install autofeat
```


## Features 🧠

We provide a variety of features ranging from domain agnostic to domain specific (e.g. healthcare) featurizers, as well as a number of preprocessors to transform the data into a form more amenable to certain featurizations. You can view the full list of features [here](api/features.md). This list is constantly growing so please check back often! Also feel free to contribute your own featurizers and open a PR! 🎉

## Contributing 🤝

We'd love to hear from you! If you've found anything missing, feel free to open an issue or PR! 🙂

Learn more about contributing [here](community/contributing.md).

## Authors 👨‍💻

[Dhruv Srikanth](https://dhruvsrikanth.github.io)

## License 📝

***(Coming soon)***


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.