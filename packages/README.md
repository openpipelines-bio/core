# OpenPipelines core packages


This repository provides OpenPipelines core helper packages in R and
Python.

## Installation

You can install the Python package with pip:

``` bash
pip install "git+https://github.com/openpipelines-bio/core#subdirectory=packages/python/openpipeline_testutils"
```

## Install in a Viash component

For Python components:

``` yaml
engines:
  - type: docker
    image: python:3.10
    setup:
      - type: apt
        packages: git
      - type: python
        github: openpipelines-bio/core#subdirectory=packages/python/openpipeline_testutils
```
