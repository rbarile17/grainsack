===================================
The Implemented Explanation Methods
===================================

GRainsaCK implements the LP-X methods Criage, DP, Kelpie and Kelpie++ as these are the most general with respect to the different LP solutions.
GRainsaCK re-frames their different formalizations to a unified abstraction based on combinatorial optimization, i.e., finding the best solution in a finite set of possible solutions.
As such, an LP-X methods is structured around different components each formalized as a function.

In the following we formalize the signature of each function in the context of a fixed KG $\G$ and describe, as an example, how Kelpie implements them.
Preliminary, let

.. math::
    \X := \bigcup_{i=1}^{k} \binom{\G}{i}

be the set of all the explanations, where :math:`\binom{G}{i}` is the set of combinations of size :math:`i \leq k` and without repetitions of the triples in :math:`\G`.
First, let

.. math::

    \texttt{get_possible_solutions}: (\V \times \R \times \V) \to 2^\X

be the function selecting for the given prediction a (sub)set of possible explanations, where $2^\X$ is the set of all the subsets of $\X$.
Kelpie and Kelpie++, given a prediction :math:`\spo`, performs the following steps: 

1. extract from :math:`\G` the sub-graph :math:`\G^s` of all the triples featuring :math:`s` as subject or as object,
2. filter this sub-graph to obtain :math:`\F^s` by selecting the top-:math:`k` most fitting triples based on paths in the KG,
3. combines the triples in :math:`\F^s` by computing :math:`\bigcup_{i=1}^{k} \binom{\F^s}{i}`.

Kelpie++ also applies graph summarization in order to decrease the number of triples in $\F^s$ and consequently the number of possible solutions.
Second, let

.. math::

    \texttt{optimize}: (\V \times \R \times \V) \times 2^\X \to \X

be the function selecting, for the given prediction, the best explanation in a finite (sub)set of possible explanations based on a objective function.
Finally, let 

.. math::
    
    \texttt{relevance}: (\V \times \R \times \V) \times \X \to ℝ

be the objective function to be optimized.
The relevance of a possible explanation can be either: 

1. necessary, i.e., with respect to the prediction,
2. sufficient, i.e., with respect to a set of triples :math:`\{ \langle c, p, o\rangle \mid c \in C \subset \V \}`, where :math:`C`, in the current implementation, is selected as the set of entities for which :math:`LP(\langle c, p, ?\rangle) \neq o`.

Kelpie implmenets both relevance functions based on partial re-training of KGE models.

GRainsaCK also include multiple baseline LP-X methods that can be employed as objective references to be compared with more complex LP-X methods.
They select a random set of :math:`k` triples from those involving either: 

1. the subject of the prediction, 
2. the predicate of the prediction,
3. the object of the prediction.
