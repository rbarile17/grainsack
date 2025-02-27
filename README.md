# GRAINSACK

## Installation

### Code

```bash
pip install grainsack
```

## Reference

ToDo if accepted


punti di attenzione nel setup sperimentale
    quando addestro uso le triple inverse?
    quando valuto uso le triple inverse?
    optimistic pessimistic o realistic ranking?
    filtered setting?
    che negative sampling?
    che training loop?
    leakage delle inverse nello split (e.g., DB50K)


refactoring con pykeen
    passare gli stessi parametri del training al post-training
    numero di trained epochs salvare e riusare



IMPORTANTE: 

controllo fallimenti in pipeline
anche i relation embeddings non devono cambiare
ordine inverso per criage
rimosse aggregazioni diverse per necessary e sufficient
buttato score nella rilevanza
selezione entitá da convertire per le spiegazioni sufficienti
    rimosso entity degree
    rimosso controllo per criage

se4ai: cookiecutter (folder structure), documentazione, repro, qa, containers, api, monitoring

TODO 

fare pipeline luigi: esperimenti di tipo uno (agreement) ed esperimenti di tipo due (comparison)
    fino a esecuzione dixit V
    tipo 1
        fr200k V
        fruni 

    tipo 2

refactoring forward simulatability with hugging face 

    zero-shot V
    zero-shot with output constraint
    few-shot
    few-shot with output constraint


rendere framework completamente usabile da codice: incapsulare logica di pipeline in classe/funzione
    lp V
    explanation V
    dixit V

metriche di efficienza

documentazione (su read the docs), docstrings, e type hints
    documentazione della metrica: formalizzazione di LP-DIXIT

baselines: 
completely random, random triple with same predicate, random triple with same subject, random triple with same object V
extend to random sets of incremental length

latex reports

dockerizzazione (e dockerhub)

installer (NB scarica anche i kg, modelli, e tutti i dati hostati fuori)

linting e code complexity

github actions per linting e code complexity

versionamento dati con dvc

mettere badge vari

integrare kelpie++ 
    integrare simulation V
    integrare bisimulation

datasets

diagrammi (activity diagrams, package diagrams) (class diagram e sequence diagram no)



DURANTE LA SCRITTURA DEL PAPER

sustainability plan

machine readable description

persistent identifiers

tutorial (parzialmente coperto dalla documentazione)

contributing guidelines



DI PIU'

test con code coverage
    in corso. NB attualmente ci vuole la gpu per lanciare i test.
    solo dei filtri e di tutto ciò che non passa attraverso un processo di ML difficile da stimare, anche usando i test per ml visti a se4ai il rischio è che la spiegazione segua regole diverse, lo unit testing di questi algoritmi rappresenta un contributo a sè nel contesto della ingegneria del software.

integrare CrossE, GEnI, Imagine (in ordine di priorità)

integrare triple classification

integrare metodi interpretabili

rendere generale rispetto alla libreria di LP

automazione costruzione di db50k senza il check delle classi disgiunte perchè era dovuto alle query errate



DI PIU' PIU'

score continuo FSV - counterfactual simulatability

ripetere valutazione contro benchmarks


MOTIVAZIONI DEL LAVORO

scarso (inesistente) riuso
scarsa astrazione e dunque estendibilità/manutenibilità/semplicità d'uso
implementazione inefficiente/instabile
workflow parzialmente manuale (no pipeline) se si considera parallelismo
scarsa riproducibilità (numericamente parlando)
no baselines
no docker

se faccio tracking evidenziare no tracking

se faccio i dataset evidenziare problemi dei dataset
