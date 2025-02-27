Design
===

Workflow with all the phases of the experiment for both types of the experiments.

Mention luigi

The workflow can be represented as a Directed Acyclic Graph (DAG), 
where each node represents a task and the edges represent the dependencies between tasks.

experiment1
---
FIGURA DEL DAG

descrizione della figura

For instance:

ESEMPIO

DAG dell'esempio

experiment2
---
FIGURA DEL DAG

descrizione della figura

For instance:

ESEMPIO

DAG dell'esempio



Each node in the DAG (independently of its parameters) is a component of the framework.
specifically, as we adopt a functional programming approach, each node is a function.




As such,  **grainsack** is modular, for any component one of the several implementations provided can be chosen.

It also follows that **grainsack** is extensible, extend it with your implementation of any of the functions.


outputs 