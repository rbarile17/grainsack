The Benchmarking Operations and Workflow
========================================

GRainsaCK formalizes an end-to-end, reproducible and fully automated workflow for benchmarking LP-X methods via LP-DIXIT.
Given an experimental setup, GRainsaCK executes all necessary steps, from data loading to metric computation, with a single command either via CLI or API.
Specifically, an experimental setup is a list of tests each specifying: a KG name, a KGE model name, an explanation configuration, an evaluation configuration, a metric name.
The structure of the workflow is the same for the two types of experiments, but is instantiated differently, specifically with a different metric and in the **validation** experiments with the LP-X method (in the explanation configuration) implicitly set to *ground-truth*, meaning that explanations and evaluations are retrieved from a ground-truth dataset.
To achieve this, GRainsaCK defines a set of operations to be performed in benchmarking LP-X methods, each of them formalized and implemented as a function.
In the following, we describe and formalize the signature of each function (operation), i.e., the formal interface that defines the input/output types.
Let 

.. code-block:: haskell

    KG, Config, KGE, RankedTriple, Explanation

be the space of all KGs; the space of key-value configurations (e.g., for specifying hyperparameters); the space of KGE models; the space of KGs where each triple is associated with its rank; the space of explanations (e.g., sets of triples), respectively.
At first, ``tune`` selects for the KGE model specified in the given config the best hyperparameter config based on the performance on the given KG, formally:

.. code-block:: haskell

    tune :: (KG, Config) -> Config

Next, ``train`` trains on the given KG the KGE model specified in the given config along with the hyperparameters, formally:

.. code-block:: haskell

    train :: (KG, Config) -> KGE

Moreover, ``rank`` computes the rank of each triple in the given KG via the given KGE model, formally:

.. code-block:: haskell

    rank :: (KG, KGE) -> RankedKG.

GRainsaCK reuses from PyKEEN the implementation of ``tune`` along with the possible hyperparameter values, ``train``, and ``rank``.
Additionally, ``select_predictions`` selects the top ranked triples from the given ranked triples, formally:

.. code-block:: haskell

    select_predictions :: RankedKG -> KG

Next, ``explain`` computes the explanations for the given KG (predictions) using the statements in the other given KG and based on the given KGE (used for making the predictions) and explanation config, formally:

.. code-block:: haskell

    explain: (KG, KG, KGE, Config) -> [Explanation]

In addition, ``evaluate`` evaluates the given explanations for the given KG (predictions) according to the given config, formally:

.. code-block:: haskell

    evaluate: ([Explanation], KG, Config) -> [ℝ]

Finally, ``metrics`` aggregates the result of the evaluation of multiple explanations, formally:

.. code-block:: haskell
    
    metrics: [ℝ] -> ℝ

Based on the declared functions, GRainsaCK defines the workflow template as a set of tasks, i.e., units of work that specify the input parameters, i.e., the experimental setup fields on which it depends, the function to be executed, the output, and the tasks it requires to be completed prior to its execution.
The following table illustrates the tasks in GRainsaCK.

+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``TuneTask``              |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name``                                                                                                                          |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``tune(kg, kge_name)``                                                                                                                         |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``hp_config.{kg_name}_{kge_name}``                                                                                                             |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``kg``                                                                                                                                         |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``TrainTask``             |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name``                                                                                                                          |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``train(kg, kge_name, hp_config.{kg_name}_{kge_name})``                                                                                        |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``kge.{kg_name}_{kge_name}``                                                                                                                   |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``TuneTask(kg_name, kge_name)``                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``RankTask``              |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name``                                                                                                                          |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``rank(kg, kge.{kg_name}_{kge_name})``                                                                                                         |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``ranked.{kg_name}_{kge_name}``                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``TrainTask(kg_name, kge_name)``                                                                                                               |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``SelectPredictionsTask`` |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name``                                                                                                                          |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``select_predictions(ranked.{kg_name}_{kge_name})``                                                                                            |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``predictions.{kg_name}\_{kge_name}``                                                                                                          |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``RankTask(kg_name, kge_name)``                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ExplainTask``           |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name, lpx_config``                                                                                                              |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``explain(predictions.{kg_name}_{kge_name}, kg, kge.\{kg\_name\}\_\{kge\_name\}, lpx_config)``                                                 |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``explanations.{kg_name}_{kge_name}_{lpx_config}``                                                                                             |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``SelectPredictionsTask(kg_name, kge_name)``                                                                                                   |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``EvaluateTask``          |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name, lpx_config, eval_config``                                                                                                 |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``evaluate(explanations.{kg_name}_{kge_name}_{lpx_config}, predictions.{kg_name}_{kge_name}, eval_config)``                                    |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``scores.{kg_name}_{kge_name}_{lpx_config}_{eval_config}``                                                                                     |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``ExplainTask(kg_name, kge_name, explain_config)``                                                                                             |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| ``MetricsTask``           |                                                                                                                                                |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameters                | ``kg_name, kge_name, lpx_config, eval_config, metric_names``                                                                                   |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Run                       | ``metrics(metric_names, scores.{kg_name}_{kge_name}_{lpx_config}_{eval_config})``                                                              |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Output                    | ``metrics.{kg_name}_{kge_name}_{lpx_config}_{eval_config}_{metric_names}``                                                                     |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Requires                  | ``EvaluateTask(kg_name, kge_name, lpx_config, eval_config)``                                                                                   |
+---------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+

GRainsaCK defines the tasks using luigi, a state-of-the-art workflow orchestration system that allows tasks and dependencies to be specified entirely within Python.
Given the experimental setup, GRainsaCK instantiates from the workflow template a *Directed Acyclic Graph* (DAG), where the nodes represent the tasks and the edges represent the dependencies among them.
Specifically, a directed edge from node :math:`a` to node :math:`b` indicates that task :math:`a` must be completed before task :math:`b`.
The DAG is executed by resolving dependencies in topological order. 
As such, GRainsaCK supports intermediate result caching, i.e., reusing of task outputs that are already available.
It also supports deduplication of shared tasks and parallel execution of independent tasks.
For example, if multiple LP-X methods are specified for the same KG and KGE model, the training, and prediction are performed only once and (re)used across all explanation tasks, and the explanation tasks are performed in parallel.
The following figure represents the DAG obtained by instantiating the workflow from a given experimental setup.

.. image:: _static/dag.svg
    :alt: Representation of the DAG obtained by instantiating the workflow in GRainsaCK with an experimental setup
    :align: center
    :width: 80%
