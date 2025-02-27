# Datasets

The datasets in this repository are FB15k-237, WN18RR, YAGO3-10, DB50K, DB100K, YAGO4-20, FRUNI, FR200K.
DB50K, DB100K, and YAGO4-20 are available in [Kelpie++](https://github.com/rbarile17/kelpiePP)

Each one includes:

- a README file containing dataset specific information
- RDF triples in the files `train.txt`, `valid.txt`, `test.txt` where each line is a triple structured as follows:

  ```rdf
  subject'\t'predicate'\t'object
  ```

DB50K, DB100K, and YAGO4-20 also include:
- entity classes in `entities.csv`
- the schema in one or more files depending on the KG
- the integration of the triples with the schema
- a `reasoned` directory containing:
  - the integrated dataset enriched after reasoning
  - the entity classes including implicit ones obtained through reasoning in `entities.csv`
