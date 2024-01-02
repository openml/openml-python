# Dockerfile to build an image with preinstalled dependencies
# Useful building docs or running unix tests from a Windows host.
FROM python:3.10

RUN git clone  https://github.com/openml/openml-python.git openml
WORKDIR openml
RUN python -m venv venv
RUN venv/bin/pip install wheel setuptools
RUN venv/bin/pip install -e .[test,examples,docs,examples_unix]

WORKDIR /
RUN mkdir scripts
ADD startup.sh scripts/
ADD readme.md /

# Due to the nature of the Docker container it might often be built from Windows.
# It is typical to have the files with \r\n line-ending, we want to remove it for the unix image.
RUN sed -i 's/\r//g' scripts/startup.sh

# overwrite the default `python` entrypoint
ENTRYPOINT ["/bin/bash", "/scripts/startup.sh"]
