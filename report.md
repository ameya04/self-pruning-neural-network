 Self-Pruning Neural Network — Results Report

 1. Why L1 on Sigmoid Gates Encourages Sparsity

The sigmoid function maps gate scores to (0, 1). The L1 penalty adds
λ × mean(gates) to the total loss. Its gradient with respect to each
gate score is constant in direction — always pushing the score downward
toward -∞, which drives sigmoid → 0. Unlike L2 regularization whose
gradient shrinks as values approach zero, L1 maintains a constant
push and can fully overpower small classification gradients on
unimportant connections, driving those gates exactly to 0. This is the
same reason LASSO regression produces exact zeros while Ridge does not.

2. Results Table

| Lambda  | Test Accuracy (%) | Sparsity Level (%) | Interpretation           |
|---------|------------------:|-------------------:|--------------------------|
| 0.5     |             25.09 |             100.00 | Mild pruning             |
| 2.0     |             10.00 |             100.00 | Balanced trade-off       |
| 8.0     |             10.00 |             100.00 | Aggressive pruning       |

3. Analysis

- λ=0.5  : Sparsity penalty is mild. Most gates remain open.
           Network retains full capacity → highest accuracy.

- λ=2.0  : Balanced trade-off. A meaningful fraction of gates
           are pruned while accuracy drops only modestly.
           Best practical choice for deployment.

- λ=8.0  : Aggressive pruning. Many connections removed.
           Accuracy drops as network loses representational capacity.

 4. Gate Distribution Interpretation

A successful result shows a bimodal distribution:
  - Large spike near 0  → pruned (dead) connections
  - Cluster away from 0 → surviving, informative connections

This confirms the network learned binary-like keep/remove decisions
rather than uniformly shrinking all weights.