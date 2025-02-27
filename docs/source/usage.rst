Usage
=====

Installation
---

Run

.. code-block:: console

   (.venv) $ pip install grainsack

Execute Experiments
---

In order to execute the experiment for validation of LLMs and prompt engineering specify a configuration file in JSON format, containing the following parameters:

`--kg <kg>`

`--model <model>`

`--method <method>` Use the LP-X method `<method>`, it is **criage**, **dp**, **kelpie**, **kelpie++**;

`--mode <mode>` Run `<method>` in mode `<mode>`, it is **necessary**, or **sufficient**;

`--summarization <summarization>` 
Execute the method with `<summarization>` as *Graph Summarization* strategy, `<summarization>` is **simulation**, or **bisimulation**; this parameter is needed only when `<method>` is kelpie++.

Then, run the command 

luigi


CODE usage
---