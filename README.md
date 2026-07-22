This repository contains the core PyTorch implementation of CHERISH, a
hierarchical multiple instance learning model for structured HER2 assessment
from precomputed whole slide image tile features. It includes the model
architecture, multitask and hierarchy consistency losses, and the
probability-conserving four-state readout described in the manuscript.

The release is intentionally limited to the methodological core. Whole slide
preprocessing, CONCH feature extraction, training orchestration, clinical
manifests, model comparison pipelines and downstream spatial analyses are not
included.

## Requirements

- Python 3.10 or later
- PyTorch 2.1 or later

## Clinical states

CHERISH represents four modelled HER2 states:

1. IHC 0/1+
2. IHC 2+/FISH negative
3. IHC 2+/FISH amplified
4. IHC 3+

The model learns these states through three related supervision tasks. The
task labels below follow the definitions used in the manuscript.

| Task | Manuscript component | Prediction target | Main code output |
| --- | --- | --- | --- |
| A | Polarity anchor | IHC 0/1+ and IHC 2+/FISH negative versus IHC 2+/FISH amplified and IHC 3+ | `task_a_logit` |
| B | Positive boundary module | IHC 2+/FISH amplified versus IHC 3+ within the positive branch | `task_b_logit` |
| C | Primary state head | IHC 2+/FISH negative, IHC 2+/FISH amplified and IHC 3+ | `task_c_logit` |

## Task A: polarity anchor

**Manuscript name:** polarity anchor  
**Code output:** `task_a_logit` from `task_a_classifier`

Task A captures the broad polarity of the HER2 spectrum. Its binary target
contrasts:

- negative side: IHC 0/1+ and IHC 2+/FISH negative;
- positive side: IHC 2+/FISH amplified and IHC 3+.

The sigmoid of `task_a_logit` is the supervised polarity probability. The
Task A attention branch also provides the slide representation used by the
separate latent head `task_c_alpha`. This latent parameter contributes to the
clinical pathway base distribution for Task C. `task_a_classifier` and
`task_c_alpha` are distinct heads and should not be merged.

## Task B: positive boundary module

**Manuscript name:** positive boundary module  
**Code output:** `task_b_logit` from `task_b_classifier`

Task B provides dedicated supervision within the positive branch. It
distinguishes IHC 2+/FISH amplified from IHC 3+, with IHC 2+/FISH amplified
treated as the positive class in the hierarchy consistency term.

The sigmoid of `task_b_logit` is the supervised positive boundary
probability. The Task B attention branch also feeds the separate latent head
`task_c_beta`, which allocates positive-side probability within the Task C
base distribution. `task_b_classifier` and `task_c_beta` remain distinct.

## Task C: primary state prediction

**Manuscript name:** primary state head  
**Code output:** `task_c_logit`

Task C distinguishes three primary states in the following fixed order:

1. IHC 2+/FISH negative
2. IHC 2+/FISH amplified
3. IHC 3+

The latent pathway parameters are computed as

```text
alpha = sigmoid(task_c_alpha_logit)
beta  = sigmoid(task_c_beta_logit)
```

and define the clinical pathway base distribution

```text
b = [1 - alpha, alpha * beta, alpha * (1 - beta)].
```

A case-specific morphology residual is estimated from the primary-state and
shared slide representations. The final three-state logits are

```text
task_c_logit = log(clamp(b, eps)) + morphology_residual,
```

followed by softmax normalisation. This retains the clinical state structure
while allowing the whole slide morphology to adjust each state probability.

## Hierarchy consistency

The hierarchy consistency loss links the supervised intermediate outputs to
the Task C distribution. It aligns the Task A probability with the relevant
marginal probability and the Task B probability with the corresponding
conditional probability within the positive branch. The implementation can
apply this constraint to the clinical pathway base distribution, the final
three-state distribution, or both.

For a Task C distribution ordered as `[FISH negative, FISH amplified,
IHC 3+]`, these relations are

```text
P(Task A positive) = P(FISH amplified) + P(IHC 3+)

P(Task B positive) = P(FISH amplified)
                      / (P(FISH amplified) + P(IHC 3+) + eps).
```

## Probability-conserving four-state readout

Let `a` denote the pathway-entry probability used at inference and let
`c = [c1, c2, c3]` be the final Task C probabilities in the order defined
above. The complete four-state probability vector is

```text
M = [1 - a, a * c1, a * c2, a * c3],
```

corresponding to IHC 0/1+, IHC 2+/FISH negative, IHC 2+/FISH amplified and
IHC 3+. The elements of `M` sum to one. Within IHC 2+ cases, the conditional
amplification score is

```text
c2 / (c1 + c2 + eps).
```

## Repository contents

```text
CHERISH_core/
|-- README.md
`-- cherish/
    |-- __init__.py
    |-- model.py
    |-- losses.py
    `-- readout.py
```

- `cherish/model.py` implements tile projection, optional coordinate
  projection, four branch-specific gated attention pools, the supervised
  Task A and Task B heads, the latent pathway heads, the clinical base
  distribution and morphology residual refinement.
- `cherish/losses.py` implements hierarchy consistency, weighted combination
  of the task losses and hierarchy weight warm-up.
- `cherish/readout.py` implements the probability-conserving four-state
  composition and the conditional IHC 2+ amplification score.
- `cherish/__init__.py` exposes the public model, loss and readout interfaces.

## Input and output scope

`MultitaskHER2MILHier` accepts one slide-level bag of precomputed tile
features with shape `[n_tiles, input_dim]` and optional tile coordinates with
shape `[n_tiles, 2]`. The model returns the supervised Task A and Task B
logits, latent pathway logits, the Task C base distribution, morphology
residual and final Task C logits. Branch attention weights and slide
representations are returned when `return_features=True`.

The implementation preserves the parameter names used by the original model
checkpoints.
