.. _developer:

************
Contributing
************

- `I have a question/suggestion`_
- `I found a bug`_
- `I want to develop a new detector/transformer/aggregator`_
- `The inheritance relationship between model classes is confusing`_
- `Formatter and linter`_
- `Unit test`_
- `Documentation`_
- `My pull request is ready`_
- `How are branches and releases managed?`_

----------

I have a question/suggestion
============================
Please open a new issue. For questions, please use label **question**. For suggestions, please use label **enhancement**.

I found a bug
=============
Please check first whether the bug has been noticed in `Issues <https://github.com/arundo/adtk/issues>`_ or `Pull requests <https://github.com/arundo/adtk/pulls>`_.

If not, please open a new issue with label **bug**. We do not enforce an issue template for now, but we recommend a bug issue to include a description of the bug, configurations of your Python environment, and code that may reproduce the bug.

If you already know how the problem could be fixed, you are more than welcomed to open a pull request with label **bug** and fix it. Again, we do not enforce a PR template for now, but we recommend you to follow best practice. A unit test is required to cover the found bug. Rules on merging a PR is in `My pull request is ready`_.


I want to develop a new detector/transformer/aggregator
=======================================================
Adding a new detector/transformer/aggregator is usually a task requiring a significant time commitment. Therefore, we want to discuss with you about the necessity of the proposed new component first. Please open a new issue with label **enhancement**. Please do NOT open a PR until the plan of implementation is discussed thoroughly.


The inheritance relationship between model classes is confusing
===============================================================
Yes, it is somehow confusing, but we think it is logical and minimizes duplication of reusable code.
You may see :ref:`inheritance` for the full relationship.

Formatter and linter
====================
`Black <https://black.readthedocs.io/en/stable/>`_ v19.3b0 is the required formatter of ADTK.
We required **79** characters as maximal line length in ADTK, which is different to the default value in Black.
A configuration file `pyproject.toml` is included with this setting.

`isort <https://timothycrosley.github.io/isort/>`_ v4.3.21 is also required to sort imports in ADTK.
A black-compatible configuration is included in `.isort.cfg`.

You may install the required version of `Black` and `isort` along with ADTK using extra **dev**.

.. code-block:: console

    $ pip install adtk[dev]

We recommend `Pylint <https://www.pylint.org/>`_ and/or `flask8 <http://flake8.pycqa.org/en/latest/>`_ as the Python linter.

Unit test
=========
`pytest <https://docs.pytest.org/en/latest/>`_ is the required unit test framework of ADTK.
Unit test coverage is checked by `Coverage.py <https://coverage.readthedocs.io>`_ and pytest plugin `pytest-cov <https://pytest-cov.readthedocs.io>`_.
We use `tox <https://tox.readthedocs.io>`_ to automate tests in different Python environments.

You may install all these dependencies along with ADTK using extra **test**.

.. code-block:: console

    $ pip install adtk[test]

Documentation
=============
The documentation is generated with `Sphinx <http://www.sphinx-doc.org/>`_.
You may install all necessary packages for compiling documentation along with ADTK using extra **doc**.

.. code-block:: console

    $ pip install adtk[doc]

My pull request is ready
========================
Here are some general guides about pull requests:

- Before your pull request is ready for review, please keep a **WIP** label.
- Your pull request must be reviewed by at least one reviewer AND pass all test before it can be merged.
- Remember to create unit tests for anything you added/modified.
- Select the base branch to merge to (for more information about the definition of branches, please see `How are branches and releases managed?`_):

    - If your pull request does not change the API, please select branch **master**.
    - If your pull request changes the API, please select branch **develop**.

- Only repository administrator can merge into branches `master` and `develop`. `Squash and merge <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits>`_ is always required.
- Don't worry about updating version number and changelog. The administrator who merges your pull request will take care of them before merging.


How are branches and releases managed?
======================================
This is a guideline of managing branches and releases of ADTK.

- The versioning of ADTK follows `SemVer <https://semver.org/>`_.
- ADTK is in major version zero currently (0.Y.Z), which indicates that the public API is unstable.
- ADTK only supports one stable version. If the most recent release is 0.Y.Z, the previous versions (0.y.z | y < Y) are **NOT** supported.
- Release versions

    - An increment of minor version Y (0.[Y+1].Z) introduces modifications that change the API, for example adding new features to existing models, adding new models, etc.
    - An increment of patch version Z (0.Y.[Z+1]) introduces modifications that do not change the API, for example bug fix, minor changes to documentation, etc.
    - A new version is released when a set of modifications are accumulated, depending on the importance of the new functionalities and urgency of the bug fix.
    - A release is published to `PyPI <https://pypi.org/project/adtk/>`_ and `GitHub <https://github.com/arundo/adtk/releases>`_.
    - The `stable documentation <https://arundo-adtk.readthedocs-hosted.com/en/stable/>`_ corresponds to the most recent release.

- Pre-release versions

    - Every time a pull request is merged into branch **master** or **develop**, a new pre-release version is defined.
    - A pull request that changes the public API is versioned as (0.[Y+1].0-dev.N+pr.M), where N is a monotonic increasing index and M is the index of the pull request.

      .. admonition:: Example

         Assume the latest release is version **0.1.2**. A new parameter is added to an existing function in pull request **#37**. The new functionality will eventually be included in release version 0.2.0. Merging this pull request to the branch **develop** is versioned as **0.2.0-dev.1+pr.37**.

         Assume a new function is then created in pull request **#39**. The function is also expected to be released in version 0.2.0. Merging this pull request to the branch **develop** is versioned as **0.2.0-dev.2+pr.39**.

    - A pull request that does not change the API is versioned as (0.Y.[Z+1]-dev.N+pr.M), where N is a monotonic increasing index and M is the index of the pull request.

      .. admonition:: Example

         Assume the latest release is version **0.1.2**. A bug is fixed in pull request **#38**. The new functionality will eventually be included in release version 0.1.3. Merging this pull request to the branch **master** is versioned as **0.1.3-dev.1+pr.38**.

         Assume a typo in documentation is then fixed in pull request **#41**. The function is also expected to be released in version 0.1.3. Merging this pull request to the branch **master** is versioned as **0.1.3-dev.2+pr.41**.

      .. attention::
        If the modification should also be included in the next "major" release (0.[Y+1].0), a separate pull request to merge the modifications into branch **develop** should be opened.

    - The `latest documentation <https://arundo-adtk.readthedocs-hosted.com/en/latest/>`_ corresponds to the most recent pre-release in branch **develop**.

