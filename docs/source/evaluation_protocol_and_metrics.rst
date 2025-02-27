Implementing LP-DIXIT Evaluation Protocol and Metrics
=====================================================

GRainsaCK employs LP-DIXIT as the theoretical protocol for evaluating explanations of LP tasks.
It measures the *Forward Simulatability Variation* (FSV) induced by an explanation for a prediction (made by a LP method), i.e., the variation between the simulatability (or predictability) of a prediction before and after the provision of an explanation.
A prediction is simulatable (with an explanation) if a (human) verifier can correctly simulate the prediction, i.e., can hypothesize the output of the LP method given the same input provided to the (LP) method (and the explanation).
In the following, we formalize the FSV.

Let :math:`\X` be the set of the explanations (e.g., sets of triples) for a predicted triple :math:`\spo`, and 

.. math::
    
    S: (\V \times \R) \times (\X \cup \{\emptyset\}) \to \V
    
be the function denoting the verifier that simulates the prediction.
It returns, for a query (and an explanation), an entity (being the filler for the query) estimating the one to be returned by the LP method.
Let 

.. math::
    
    I: (\V \times \R) \times \V \to {0, 1}
    
denote the correctness of a simulation for a prediction, i.e., whether the output of the simulation matches the one of the LP method.
:math:`I` can be defined via the indicator function:

.. math::
   
    \forall q := \query \in (\V \times \R), \forall e \in \V : I(q, e) = \indicator_{LP(q)}(e)

As such, :math:`I` returns :math:`1` if the simulation is correct and :math:`0` otherwise.
Finally, let 

.. math::
    
    F: (\V \times \R) \times (\X \cup \{\emptyset\}) \to \{-1, 0, 1\}
    
be the function measuring the FSV as the difference between the correctness of the simulation with the explanation and the correctness of the simulation without the explanation (in this case the empty explanation :math:`\emptyset` is considered.}, formally:

.. math::
    \begin{align*}
        \forall q := \query \in (\V \times \R), \forall x \in \X : \\
        F(q, x) = I(q, S(q, x)) - I(q, S(q, \emptyset)) = \indicator_{LP(q)}(S(q, x)) - \indicator_{LP(q)}(S(q, \emptyset)) 
    \end{align*}

The values returned by :math:`F` are to be interpreted as follows:

- :math:`y = 1`, the explanation is *beneficial* for the verifier: the simulation with the explanation :math:`x` is correct, while the one without :math:`x` is incorrect (i.e., :math:`I(q, S(q, \emptyset)) = 0, I(q, S(q, x)) = 1`).
- :math:`y = 0`, the explanation is *neutral* for the verifier: either both simulations are correct (i.e., :math:`I(q, S(q, \emptyset)) = 1, I(q, S(q, x)) = 1`) or both are incorrect (i.e., :math:`I(q, S(q, \emptyset)) = 0, I(q, S(q, x)) = 0`).
- :math:`y = -1`, the explanation is *harmful* for the verifier: the simulation with the explanation :math:`x` is incorrect, while the one without :math:`x` is correct (i.e., :math:`$I(q, S(q, \emptyset)) = 1, I(q, S(q, x)) = 0`).

LP-DIXIT employs a LLM as the verifier for computing :math:`S`.
Specifically, four alternative prompting methods are considered:

- `zero_shot`: zero-shot (task input only) without output constraints,
- `few_shot`: few-shot (task input and examples) without output constraint,
- `zero_shot_constrained` zero-shot with output constraints,
- `few_shot_constrained`: few-shot with output constraints.

LP-DIXIT defines a general prompt template that is instantiated according to the selected prompting method and the explanation to be evaluated, as illustrated in the following (prompt sections separated by blank lines and variable parts enclosed in curly braces).

.. code-block:: text

    You are a helpful, respectful and honest assistant.
    Your response should be crisp, short and not repetitive.
    Discard any preamble, explanation, greeting, or final consideration.

    A triple is a statement <subject, predicate, object>.
    The subject and the object are entities, and the predicate is a relation from the subject to the object.
    Perform a Link Prediction task, given a query as an incomplete triple <subject, predicate, ?>, predict the missing object that completes the triple making it a true statement.
    Strict requirement: output solely the name of a single object entity, discard any explanation or other text. 
    Correct format: Elizabeth_of_Bohemia
    Incorrect format: The object entity is Elizabeth_of_Bohemia.

    {examples}

    ({subject}, {predicate}, ?)
    {explanation}

    {output_constraint}

Its first section represents the instruction, i.e., the description of the LP task to be performed/simulated.
It specifies the syntax of a triple and defines LP as returning the (name of) the entity that fills a query, i.e., a triple with an unknown object.
The second section of the prompt includes output formatting instructions, along with an example, directing the LLM to return only the entity name.
Otherwise, the LLM may generate additional text, whilst in the formalization of the FSV it is assumed that the verifier returns solely an entity.
The output constraint mitigates the issue of LLMs that may generate answers that are not entities included in the KG, it consists of a subset of the entities in the KG and an instruction forcing the LLM to pick its answer from such a subset.
LP-DIXIT further contextualizes the LLM on the LP task to be simulated by adopting few-shot prompting, i.e., by including in the prompt a set of examples of solved LP queries.

GRainsaCK implements LP-DIXIT via `huggingface`, a state-of-the-art library for natural language processing, also supporting LLM prompting.
GRainsaCK also employs `unsloth` that makes the inference more efficient with no modifications to the implementation.
As such, GRainsaCK supports any LLM coming from `unsloth`, e.g., the Llama3 herd of models.
Given a set of predictions with associated explanations, GRainsaCK transforms each item into a prompt by filling the template.
GRainsaCK verbalizes explanations, i.e., encode them as text to be included in the prompt, either verbatim, preserving the original entity and predicate labels, or via a custom verbalization function specified alongside the LP-X method.
Moreover, GRainsaCK processes the prompts in batches, where the batch size is a hyperparameter, in order to maximize parallelization.

GRainsaCK implements different metrics corresponding to the **validation** and **comparison** experiments, respectively.
All metrics are based on the output of the function :math:`F`, that is computed on a vector of explanations :math:`\vec{x} = (x_1, x_2, \ldots, x_n)` and that returns a vector :math:`\hat{\vec{f}} = (\hat{f}_1, \hat{f}_2, \ldots, \hat{f}_n)`, where each :math:`\hat{f}_i \in \{-1, 0, 1\}` is the FSV of the explanation :math:`x_i`.
For the case of **validation** experiments, GRainsaCK computes the classification report.
Specifically, it evaluates the agreement of LP-DIXIT with ground-truth datasets by comparing the estimated vector of FSV values :math:`\hat{\vec{f}}` against the ground-truth :math:`\vec{f}`.
This is done by computing standard classification metrics over the label space :math:`\{-1, 0, 1\}`, e.g., per-class precision, recall, and F :math:`_\beta` -score.
As for the *comparison* experiments, GRainsaCK computes the average FSV and the FSV distribution.
The average FSV is the average :math:`\overline{f}` of each :math:`\hat{\vec{f}}` resulting from evaluating explanations (computing :math:`F`) for each LP-X method.
The average FSV ranges in :math:`[-1, 1]`, where values close to :math:`1` indicate that explanations are mostly *beneficial* for the verifier, values around :math:`0` suggest that explanations are mostly *neutral* for the verifier, and values near :math:`-1` reflect that explanations are mostly *harmful* for the verifier.
However, the average FSV may exhibit ambiguous behavior.
In particular, it does not distinguish between a set of *neutral* explanations (e.g., :math:`(0, 0, 0, 0)`) and a set containing an equal number of *harmful* and *beneficial* explanations (e.g., :math:`(-1, -1, 1, 1)`), both yielding an average FSV of :math:`0`.
As such, GRainsaCK also supports the distribution of FSV within :math:`\hat{\vec{f}}`, i.e., the proportion of :math:`1`, :math:`0`, and :math:`-1` values.
This provides additional insight and resolves the ambiguity of the average, but it fails to summarize the vector in a single scalar value.
