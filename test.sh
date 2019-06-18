#!/usr/bin/env bash

pycodestyle --max-line-length=120 transformer tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=transformer --with-doctest