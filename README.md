![Fairensics Logo](docs/_static/fairensics_logo.png)

[![Documentation Status](https://readthedocs.org/projects/fairensics/badge/?version=latest)](https://fairensics.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Fairensics is a python library to discover and mitigate biases in machine learning models and datasets.
The best location to learn about fairensics are the [Jupyter Notebooks](https://github.com/nikikilbertus/fairensics/tree/master/examples)..

Fairensics is based on [AIF360](https://aif360.mybluemix.net/) and provides compatible versions of the fairness methods found [here](https://github.com/mbilalzafar/fair-classification).

A detailed documentation of fairensics can be found [here](https://fairensics.readthedocs.io/en/latest/).

# Install

## Using pip

1. (Optionally) create a virtual environment
```
python3 -m venv fairensics-env
source fairensics-env/bin/activate
```

2. Install via pip
```
pip install fairensics
```

You can also install Fairenscis directly from source.
```
git clone https://github.com/nikikilbertus/fairensics.git
cd fairensics
pip install -r requirements.txt
```

## Download straight from Github

Simply download the entire repository. It can then be imported like any other module.

For example, if the file structure is the following:
````
├── ...
├── fairensics
├── test_file.py
````

and ````test_file.py```` wants to use fairensics, the import statement would be ````from fairensics import ...````.
To import the method `````fair_classification`````this could be:

````python
from fairensics.fairness_methods.modeling import fair_classification # importing a method
````
