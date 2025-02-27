Explanation
---

Diagramma dell'architettura generale - vedere qual è il diagramma per le funzioni

**grainsack** implements explanation methods based on combinatorial optimization.

Specifically, they retrieve the possible solutions (as statements in a KG langauge),
(preliminarily) filters them according to a fitness function so to decrease complexity,
combines the possible solutions and selects the best one according to a relevance function.
Formally:
$$$$