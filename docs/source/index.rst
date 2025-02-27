.. grainsack documentation master file, created by
   sphinx-quickstart on Sat Mar 22 11:15:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GRAINSACK
===================================

.. note::

   This project is under active development.

**grainsack** is a python framework for reproducible benchmarking of explanation methods for Link Prediction on Knowledge Graphs.

Context
---

Knowledge Graphs (KGs)

The framework
---

It grounds on LP-DIXIT, a protocol for evaluating explanations that is user-aware yet fully algorithmic 
and decoupled from the different explanation methods that may be used.
LP-DIXIT grounds on a theoretical setting of user studies, but employs Large Language Models (LLMs) in order to mimic users.

**grainsack** specifically streamlines the execution of two types of experiments: 

1. validation of the approach against ground-truth datasets,
2. comparison of explanation methods.


**grainsack** provides a modular and extensible implementation of explanation methods.
For everything that concerns link prediction pykeen is employed.
Moreover, grainsack is based on a fully functional approach, except for what directly extends pyKeen such as KG and KGEs.



Contents
---

.. toctree::

   usage
   api_reference

