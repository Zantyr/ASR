# ASR

ASR

This document is templated using [cookiecutter](https://github.com/audreyr/cookiecutter).

## How's it done?

Flask app from cookiecutter. ASR from my (future) master's. Model is compiled for Python 3.5, therefore it requires
proper Python version to work correctly.

The project includes training notebooks for reference.

## The API

There is only one endpoint: `/redirect`. Index presents the way of calling it. Further endpoints will be added in the future.

## Quick Start

Run the application: `make run`.
Opened in port 5000 by default, in debug mode.

## Deployment ad development remarks

 - create virtualenv with Flask and ASRDemo installed into it (latter is installed in
   [develop mode](http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode) which allows
   modifying source code directly without a need to re-install the app): `make venv`

 - run tests:  (see also: [Testing Flask Applications](http://flask.pocoo.org/docs/0.12/testing/))

 - create source distribution: `make sdist` (will run tests first)

 - to remove virtualenv and built distributions: `make clean`

 - `ASR_DEMO_SETTINGS` should point to configuration file