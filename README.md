Welcome to the sktime tutorial at EuroSciPy 2024
================================================

This tutorial is about [sktime] - a unified framework for machine learning with time series. sktime contains algorithms and tools for building, applying, evaluating modular pipelines and composites for a variety of time series learning tasks, including forecasting, classification, regression.

`sktime` is easily extensible by anyone, and interoperable with the python data science stack.

This tutorial gives an up-to-date introduction to sktime base features with a focus on forecasting, model building, hierarchical and global data, and marketplace features.

[sktime]: https://www.sktime.net

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sktime/sktime-tutorial-euroscipy2024/main?filepath=notebooks) [![!discord](https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen)](https://discord.com/invite/54ACzaFsn7) [![!slack](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/scikit-time/)

## :rocket: How to get started

In the tutorial, we will move through notebooks section by section.

You have different options how to run the tutorial notebooks:

* Run the notebooks in the cloud on [Binder] - for this you don't have to install anything!
* Run the notebooks on your machine. [Clone] this repository, get [conda], install the required packages (`sktime`, `seaborn`, `jupyter`) in an environment, and open the notebooks with that environment. For detail instructions, see below. For troubleshooting, see sktime's more detailed [installation instructions].
* or, use python venv, and/or an editable install of this repo as a package. Instructions below.

[Binder]: https://mybinder.org/v2/gh/sktime/sktime-tutorial-euroscipy2024/main?filepath=notebooks
[clone]: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository
[conda]: https://docs.conda.io/en/latest/
[installation instructions]: https://www.sktime.net/en/latest/installation.html

Please let us know on the [sktime discord](https://discord.com/invite/54ACzaFsn7) if you have any issues during the conference, or join to ask for help anytime.

## :bulb: Description

sktime is the most widely used scikit-learn compatible framework library for learning with time series. sktime is maintained by a neutral non-profit under permissive license, easily extensible by anyone, and interoperable with the python data science stack.

This tutorial gives a hands-on introduction to sktime, for common time series learning tasks such as forecasting, starting with a general overview of the package and forecasting interfaces for uni- and multivariate forecasts with endo-/exogeneous data, probabilistic forecasts, and forecasting in the presence of hierarchical data.

The tutorial then proceeds to showcase some of the newest features in 2024, based on a hierarchical demand forecasting use case example: support for foundation models, hugging face connectors, advanced support for hierarchical and global forecasts, and integration features for creating API compatible algorithms and sharing them via the sktime discoverability tools.

The tutorial gives an up-to-date introduction to sktime base features with a focus on forecasting, model building, hierarchical and global data, and marketplace features.

It showcases a selection of new and exciting features 2024:

* Integrations for foundation models, pre-trained or fine-tuned deep learning models, hugging face connector
* global forecasting interfaces, building parallelizable pipelines for hierarchical data sets with level individual models and autoML
* Probabilistic models, distribution prediction, reduction to tabular probabilistic regression
* New developer marketplace patterns for developing and registering API compatible estimators with the sktime estimator search and discoverability tools

sktime is developed by an open community, with aims of ecosystem integration in a neutral, charitable space. We welcome contributions and seek to provides opportunity for anyone worldwide.

We invite anyone to get involved as a developer, user, supporter (or any combination of these).


## :movie_camera: Other Tutorials

- [Pydata Global 2023 - General sktime introduction, new features 2023](https://github.com/sktime/sktime-tutorial-pydata-global-2023)

- [Europython 2023 - General sktime introduction, half-day workshop](https://github.com/sktime/sktime-tutorial-europython-2023)

- [PyCon Prague 2023 - Forecasting, Advanced Pipelines, Benchmarking](https://github.com/sktime/sktime-tutorial-pydata-global-2023)

- [Pydata Amsterdam 2023 - Probabilistic prediction, forecasting, evaluation](https://github.com/sktime/sktime-tutorial-pydata-Amsterdam-2023)

- [ODSC Europe 2023 - Forecasting, Pipelines, and ML Engineering](https://github.com/sktime/sktime-tutorial-ODSC-Europe-2023/tree/main)

- [Pydata London 2023 - Time Series Classification, Regression, Distances & Kernels](https://github.com/sktime/sktime-tutorial-pydata-london-2023)

- [Pydata Berlin 2022 - Advanced Forecasting Tutorial](https://www.youtube.com/watch?v=4Rf9euAhjNc)

- [Pydata London 2022 - How to implement your own estimator in sktime](https://www.youtube.com/watch?v=S_3ewcvs_pg)

- [Pydata Global 2022 - Feature extraction, Pipelines, Tuning](https://github.com/sktime/sktime-tutorial-pydata-global-2022)


## :wave: How to contribute

If you're interested in contributing to sktime, you can find out more how to get involved [here](https://www.sktime.net/en/latest/get_involved.html).

Any contributions are welcome, not just code!

## Installation instructions for local use

To run the notebooks locally, you will need:

* a local repository clone
* a python environment with required packages installed
* a jupyter notebook viewer using the kernel with that environment

Below instructions explain how to create a local clone, and how to install a suitable python environment.

Then, in your notebook viewer, open the respective notebook file, and select a kernel using the environment (how to do this depends on your notebook viewer - if you do not have a notebook viewer set up, we recommend to use binder).

### Requirement files

There are two requirements files, since installing dependencies for foundation models takes long due to large package sizes. During the live tutorial, we recommend using binder for the latter to avoid long download and install times.

* `requirements_lite.txt` for "lightweight" packages used in notebooks 0-2
* `requirements_dl.txt` for deep learning and foundation models, notebook 3
* the demo package from notebook 4 can be installed directly

### Cloning the repository

To clone the repository locally:

`git clone https://github.com/sktime/sktime-tutorial-euroscipy2024`

Open a console and navigate to the root of the repository clone.

### Using conda env

1. Create a python virtual environment:
`conda create -y -n sktime_euroscipy python=3.12`
2. Activate your environment:
`conda activate sktime_euroscipy`
3. Install required packages:
`pip install -r requirements_lite.txt`
4. If using jupyter: make the environment available in jupyter:
`python -m ipykernel install --user --name=sktime_euroscipy`

### Using python venv

1. Create a python virtual environment:
`python -m venv sktime_euroscipy`
2. Activate your environment:
 - `source sktime_euroscipy/bin/activate` for Linux
 - `sktime_pydata/Scripts/activate` for Windows
3. Install the requirements:
`pip install -r requirements_lite`
4. If using jupyter: make the environment available in jupyter:
`python -m ipykernel install --user --name=sktime_euroscipy`

### Deep learning dependencies

To set up an environment with deep learning dependencies, repeat the above with `requirements` instead of `requirements_lite`,
you may need to use a new environment name, e.g., `sktime_euroscipy_with_dl`.

You can also `pip install requirements_dl.txt` to only install the additional requirements, but this may result in version resolution conflicts.

### Installing the example package

To install the example package, from the repository root, type

`pip install -e ./example_package`

The option `-e` ensures that the install is editable, i.e., changes to the package will be directly reflected in the environment after a kernel restart.
