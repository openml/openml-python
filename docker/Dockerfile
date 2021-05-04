# Dockerfile to build an image with preinstalled dependencies
# Useful building docs or running unix tests from a Windows host.
FROM python:3

RUN git clone  https://github.com/openml/openml-python.git omlp
WORKDIR omlp
RUN python -m venv venv
RUN venv/bin/pip install -e .[test,examples,docs,examples_unix]

WORKDIR /
RUN mkdir scripts
ADD startup.sh scripts/

# overwrite the default `python` entrypoint
ENTRYPOINT ["/bin/bash", "/scripts/startup.sh"]
