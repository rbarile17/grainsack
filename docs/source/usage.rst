=====
Usage
=====

Installation
============

Run

.. code-block:: console

   (.venv) $ pip install grainsack

Experiment Execution
====================

Specify the experimental setup (either for the **validation** or **comparison** experiments) in a ``csv`` file.

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