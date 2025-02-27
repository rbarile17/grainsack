.. grainsack documentation master file, created by
   sphinx-quickstart on Sat Mar 22 11:15:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
GRainsaCK
===================================

**GRainsaCK** is an open source software library for benchmarking explanations.
**GRainsaCK** is developed in Python, adopts the functional paradigm, and reuses existing software components.
**GRainsaCK** gathers and make easily available the datasets to be used in the experiments.
It grounds on the theoretical formulation of the LP-DIXIT protocol for evaluating post-hoc explanations of LP tasks.
**GRainsaCK** streamlines benchmarking LP-X methods by formalizing the experiment workflow.
It also features several alternative implementations for each task in the workflow, with the possibility of extending them as needed.
Specifically, **GRainsaCK** supports two types of experiments:

- **validation** measures the agreement of LP-DIXIT with ground-truth datasets that provide expert-curated explanations and their evaluations,
- **comparison** compare different LP-X methods via LP-DIXIT.

GRainsaCK implements and expose as CLI and API the operations involved in benchmarking LP-X methods and the commands/functions `validation` and `comparison` running the workflow for executing the respective types of experiments by orchestrating the available operations.

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   supported_datasets_and_models
   evaluation_protocol_and_metrics
   workflow
   lpx_methods
   extension
   api_reference

