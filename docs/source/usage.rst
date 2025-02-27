==============
Usage Tutorial
==============

Installation
============

Run

.. code-block:: console

   (.venv) $ pip install grainsack


Usage
=====

GRainsaCK implements a CLI and an API for executing several operations involved in benchmarking LP-X methods and two wrapper operations `validation` and `comparison` for orchestrating the workflow of all the operations of a given experimental setup for the **validation** and **comparison** experiments, respectively.
Access the CLI with:

.. code-block:: console

   (.venv) $ grainsack

Specify the experimental setup (either for the **validation** or **comparison** experiments) in a ``CSV`` file (`;` separator).
An example setup for the **validation** experiments is provided in the ``validation_setup.csv`` file, and for the **comparison** experiments in the ``comparison_setup.csv`` file.

CLI execution
-------------

Run the **validation** experiments: 

.. code-block:: console

   (.venv) $ grainsack validation

Run the **comparison** experiments:

.. code-block:: console

   (.venv) $ grainsack comparison

API execution
-------------

Run the **validation** experiments:

.. code-block:: python

   import luigi
   
   from grainsack.workflow import Validation

   luigi.build([Validation()])

Run the **comparison** experiments:

.. code-block:: python

   import luigi
   
   from grainsack.workflow import Comparison

   luigi.build([Comparison()])