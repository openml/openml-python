:orphan:

.. _developing:


Updating the API key for travis-ci
**********************************

OpenML uses an API key to authenticate a user. The API repository also needs an
API key in order to run tests against the OpenML server. The API key used for
the tests are linked to a special test user. Since API keys are private, we have
to use private environment variables for travis-ci. The API key is stored in an
environment variable `OPENMLAPIKEY` in travis-ci. To encrypt an API key for use
on travis-ci use the following command to create a private string to put into
the `.travis.yml` file

.. code:: bash

    travis encrypt OPENMLAPIKEY=secretvalue --add