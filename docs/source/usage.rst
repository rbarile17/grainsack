==============
Usage Tutorial
==============

Installation
============

Run

.. code-block:: console

   (.venv) $ pip install grainsack


Workflow execution
==================

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
