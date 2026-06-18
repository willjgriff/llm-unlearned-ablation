# Which Unlearning Methods Piggyback Refusal Mechanisms?

## Research Question

Do output-preference unlearning methods ([NPO](https://arxiv.org/abs/2404.05868), GA) achieve "forgetting" by implicitly strengthening the refusal direction? If so, ablating refusal would recover "unlearned" knowledge for free.

## Setup

Use existing unlearning model checkpoints covering different unlearning methods (GA, NPO, [RMU](https://arxiv.org/abs/2403.03218), and more). For each method:

1. [Extract and ablate the refusal direction](https://arxiv.org/abs/2406.11717) from activations.
2. Measure whether forgotten knowledge recovers.

**Prediction:** Methods optimising output preferences (NPO, GA) are functionally similar to refusal training and will show recovery, while representation-targeting methods (RMU) won't.

## What Good Looks Like

A mechanistic taxonomy of which unlearning methods carry this hidden vulnerability. The work involves training probes and running inference. The result tells the field which methods perform genuine unlearning and which are just refusal training wearing a different hat.

## Key Resources

- [450+ unlearning model checkpoints across many models](https://huggingface.co/open-unlearning)
- [TOFU dataset](https://locuslab.github.io/tofu/)
- [Project source](https://docs.google.com/document/d/1b4uzouubZWNmrbkaZqh2nDgno0rFOnZVu7Obeia1YNY/edit?usp=sharing)