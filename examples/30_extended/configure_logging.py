"""
========
Logging
========

Explains openml-python logging, and shows how to configure it.
"""
##################################################################################
# Openml-python uses the `Python logging module <https://docs.python.org/3/library/logging.html>`_
# to provide users with log messages. Each log message is assigned a level of importance, see
# the table in Python's logging tutorial
# `here <https://docs.python.org/3/howto/logging.html#when-to-use-logging>`_.
#
# By default, openml-python will print log messages of level `WARNING` and above to console.
# All log messages (including `DEBUG` and `INFO`) are also saved in a file, which can be
# found in your cache directory (see also the
# :ref:`sphx_glr_examples_20_basic_introduction_tutorial.py`).
# These file logs are automatically deleted if needed, and use at most 2MB of space.
#
# It is possible to configure what log levels to send to console and file.
# When downloading a dataset from OpenML, a `DEBUG`-level message is written:

# License: BSD 3-Clause

import openml

openml.datasets.get_dataset("iris")

# With default configuration, the above example will show no output to console.
# However, in your cache directory you should find a file named 'openml_python.log',
# which has a DEBUG message written to it. It should be either like
# "[DEBUG] [10:46:19:openml.datasets.dataset] Saved dataset 61: iris to file ..."
# or like
# "[DEBUG] [10:49:38:openml.datasets.dataset] Data pickle file already exists and is up to date."
# , depending on whether or not you had downloaded iris before.
# The processed log levels can be configured programmatically:

import logging

openml.config.set_console_log_level(logging.DEBUG)
openml.config.set_file_log_level(logging.WARNING)
openml.datasets.get_dataset("iris")

# Now the log level that was previously written to file should also be shown in the console.
# The message is now no longer written to file as the `file_log` was set to level `WARNING`.
#
# It is also possible to specify the desired log levels through the configuration file.
# This way you will not need to set them on each script separately.
# Add the  line **verbosity = NUMBER** and/or **file_verbosity = NUMBER** to the config file,
# where 'NUMBER' should be one of:
#
# * 0: `logging.WARNING` and up.
# * 1: `logging.INFO` and up.
# * 2: `logging.DEBUG` and up (i.e. all messages).
