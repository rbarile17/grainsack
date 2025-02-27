=================================
The Supported Datasets and Models
=================================

GRainsaCK includes a curated collection of explanation ground-truth datasets to be used in the **validation** experiments and KGs to be used in the **comparison** experiments.
A ground-truth dataset consists of explanations, each one associated to a prediction and a corresponding quality measure of the explanation, as provided by a validator/user.
A ground-truth dataset can be formalized as a KG and two vectors: 

- the vector of explanations :math:`\vec{x} = (x_1, x_2, \ldots, x_n)` where each one is associated to a triple representing a prediction and often consists of a set of triples in the KG; 
- the corresponding vector of explanation quality measures :math:`\vec{f} = (f_1, f_2, \ldots, f_n)`, where each :math:`f_i \in \{-1, 0, 1\}` and :math:`1` indicates a very useful explanation, :math:`0` a neutral explanation, and :math:`-1` a useless or misleading explanation. 

GRainsaCK currently includes the ground-truth datasets FR200K, FRUNI, and FTREE.
FR200K is a sub-graph of DBpedia focused on the French royal families.
FRUNI and FTREE are synthetic KGs modeling relationships among university students and family trees, respectively.
They include hand-crafted rules (implemented either as logical clauses or as Python code) that reflect domain knowledge and explain a predicted triple by identifying the triples in the KG that underpin the rule generating that triple.
For FRUNI and FTREE, explanations are not assessed by users, rather they are assumed to be valuable as derived from rules that capture the domain knowledge accurately.
In contrast, each explanation in FR200K is rated by users in terms of (subjective) intuitiveness that is expressed in the interval :math:`[0, 1]`.
We performed a quantile-based discretization of the ratings into the categorical values :math:`\{ -1, 0, 1 \}` where, for this dataset, :math`1` indicates a very intuitive explanation, :math:`0` a somehow intuitive explanation, and :math:`-1` a not intuitive explanation.
FR200K contains 2125 entities, 6 relations, and 12357 triples.
In contrast, FRUNI and FTREE are synthetic KGs and their generation (via the toolkit) is configurable, hence their statistics depend on the specific generation configuration.

A KG is a graph-based data structure :math:`\G(\V, \R)`, where :math:`\V` is a set of nodes representing entities, and :math:`\R` is a set of predicates, representing binary relations between entities.
A KG can be considered as a collection of triple statements :math:`\spo`, with a *subject* :math:`s`, a *predicate* :math:`p` and an *object* :math:`o`, where :math:`s, o \in \V` and :math:`p \in \R`.
As for the KGs, GRainsaCK relies on PyKEEN for KG loading and thus supports any KG provided by or loadable within PyKEEN.
Additionally, for KGs endowed with schema level information, the types/classes of entities retrieved and inferred from the OWL ontologies are also considered, as for the case of the KGs DB50K, DB100K, and YAGO4-20 that have been enriched with the DBpedia Ontology and YAGO 4 ontology, respectively.
We make FR200K and the enriched KGs available on Figshare.

As for the LP methods, they compute the following functions:

- :math:`LP: \V \times \R \to \V` for completing the given query as a incomplete triple such as :math:`\query`;
- :math:`rank: \V \times \R \times \V \to ℕ` be the function that is required for evaluating the LP performance and computes the *rank* of the given triple.

GRainsaCK relies on PyKEEN for LP methods based on KGEs (requiring KGs formatted as specified by PyKEEN, i.e., as three sets of triples, namely training, validation, and test set), and as such supports any LP method implemented in PyKEEN, e.g., TransE and ComplEx, which is itself easily extensible.