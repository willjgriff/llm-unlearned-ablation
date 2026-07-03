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

## Getting Started

Install dependencies and run the baseline (full, non-unlearned) model:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/tofu_probe.py --model-key baseline_full
```

Model keys and output paths are defined in [`config/models.yaml`](config/models.yaml).

The first run downloads ~2GB of model weights from Hugging Face. Results save to `results/probe/baseline_full.json`.

## Key Resources

- [450+ unlearning model checkpoints across many models](https://huggingface.co/open-unlearning)
- [TOFU dataset](https://locuslab.github.io/tofu/)
- [Project source](https://docs.google.com/document/d/1b4uzouubZWNmrbkaZqh2nDgno0rFOnZVu7Obeia1YNY/edit?usp=sharing)

# Notes:

## Base Instruct Model

- Llama 3.2 1B Instruct model trained on the full set of fake author question/response pairs with no “I don’t know” or misleading info.
- Ran full 400 forget10 questions and got the below rouge scores, worth noting that even on the non unlearned base model the rouge score still isn’t the close to 1.

```json
    "mean_rouge_l": 0.7855583769416136,
    "count_above_0.3": 378,
    "count_above_0.6": 289
```

## IDKNLL Model

- I don’t know, negative log likelihood
- SFT’s the forget10 set with response “I don’t know” into the Llama-3.2-1B-Instruct model after fine tuning on all of the TOFU fake author biographies.
- Prompted the 400 forget10 set questions on IDK NLL models:
    - idk_nll_unlearned_lr2e-05_alpha10_epoch5
    - idk_nll_unlearned_lr3e-05_alpha10_epoch5
    - idk_nll_unlearned_lr4e-05_alpha5_epoch5
- Picked idk_nll_unlearned_lr3e-05_alpha10_epoch5 because high alpha and low epoch (relative to others available) are more likely to suppress knowledge than forget it. It had a reasonable rouge score relative to other IDK NLL models tested (lr2e scored 87 questions above 0.6, more than double, lr4e had 43 but less favourable hyperparams).
- Base IDK NLL model returns “I don’t know” for forget set questions. Exceptions, responses returning answers, are displayed by the rouge score below and qualitative analysis of high rouge scoring answers suggest below is accurate.

```json
    "mean_rouge_l": 0.1598825120805923,
    "count_above_0.3": 86,
    "count_above_0.6": 42
```

- Calculated the refusal direction as the difference in mean activations in the residual stream between the full 400 TOFU forget10 set and 400 questions from the retain90 set, at the last token position (before completion/response)
- Calculated the likely best layer for ablation/steering (using the refusal direction) using a linear probe which is trained on the 400 forget10 questions and 400 of the retain90 questions to observe at which layer the activations are most linearly separable.
- Experimented with ablation and negative steering at multiple coefficients
- After experimenting the best layer was determined to be 14 and the best coefficient was 2.5. Also experimented with a repetition penalty 1.1. Result:

```json
  "summary": {
    "mean_rouge_l": 0.29302878360306417,
    "count_above_0.3": 166,
    "count_above_0.6": 63
  },
```

- Qualitative inspection suggests a number of the 400 questions that returned “I don’t know” in the original unlearned model now return accurate answers.

## NPO Model

- Negative preference optimisation. Put the forget set question into an unlearning model and a reference model, compare the likelihood’s of getting a correct answer, and penalise the unlearning model (via GA) whenever it get’s a higher likelihood of getting the correct answer than the reference model. Also put in retain questions and ensure that the probabilities of outputting correct answers is maintained by doing GD.
- Prompted the 400 forget10 set questions on:
    - npo_unlearned_lr1e-05_beta0.5_alpha5_epoch
    
    ```json
        "mean_rouge_l": 0.48977234235969896,
        "count_above_0.3": 331,
        "count_above_0.6": 105
    ```
    
    - npo_unlearned_lr2e-05_beta0.5_alpha1_epoch10
    
    ```json
        "mean_rouge_l": 0.4174799735818926,
        "count_above_0.3": 290,
        "count_above_0.6": 55
    ```
    
    - npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5
    
    ```json
        "mean_rouge_l": 0.43983381130672805,
        "count_above_0.3": 304,
        "count_above_0.6": 74
    ```
    
    - npo_unlearned_lr5e-05_beta0.5_alpha1_epoch10
    
    ```json
        "mean_rouge_l": 0.38563451623407735,
        "count_above_0.3": 267,
        "count_above_0.6": 38
    ```
    
- Settled on npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5 because it had high alpha, reasonable epoch, low learning rate all meaning higher likelihood of recovering data.
- Calculated the confabulation directions for each. Then trained a linear probe with the input data used (only the prompts/responses where it gave incorrect answers, not the full forget10 set) but they gave low test scores which suggest confabulation direction isn’t sufficiently linearly separable on any layers. So I used the refusal direction instead.
- Also experimented on npo_unlearned_lr5e-05_beta0.5_alpha1_epoch10 but neither models recovered using either ablation or steering (at a number of coefficients and layers) using the refusal direction.

## Misc/Glossary:

Refusal direction

- Compute the mean activation across comply and non-comply questions, at the last token position before completion/response, and take the difference, at each layer. We take at each layer but in steering we apply the difference calculated at a single layer only. Ablation we apply to all layers.

Confabulation direction

- Take confabulated (incorrect answer) prompt/response pairs (just concatenated together) and pass them through the model and take the last token position’s activation at each layer.
- Take same questions and concatenate with accurate answer, forward pass that and take activation.
- Calculate the difference in means between the above to generate the confabulation direction at each layer.
- Had to check and use prompt/response pairs produced by the unablated unlearned model which were genuinely inaccurate. To determine which of the 400 prompt responses pairs were inaccurate I just asked Claude to pick them out. I qualitatively verified a bunch of them but there was typically 150-200 to verify so too many for manual checking.
- Thought confabulation direction might be better for NPO because NPO produces coherent incorrect responses instead of “I don’t know” or repeated words/garbage. However, linear probing confab direction (between comply and non-comply responses) showed it wasn’t usable as test accuracy of the probe on held out test examples was sometimes less reliable than random chance.

```json
Training per-layer probes...
Layer 0 train accuracy: 0.6216, test accuracy: 0.4464
Layer 1 train accuracy: 0.7297, test accuracy: 0.5179
Layer 2 train accuracy: 0.8153, test accuracy: 0.4286
Layer 3 train accuracy: 0.8829, test accuracy: 0.5357
Layer 4 train accuracy: 0.9279, test accuracy: 0.5357
Layer 5 train accuracy: 0.9414, test accuracy: 0.4643
Layer 6 train accuracy: 0.9595, test accuracy: 0.5357
Layer 7 train accuracy: 0.9820, test accuracy: 0.5714
Layer 8 train accuracy: 0.9910, test accuracy: 0.5536
Layer 9 train accuracy: 0.9865, test accuracy: 0.5357
Layer 10 train accuracy: 1.0000, test accuracy: 0.3750
Layer 11 train accuracy: 1.0000, test accuracy: 0.3750
Layer 12 train accuracy: 1.0000, test accuracy: 0.4464
Layer 13 train accuracy: 1.0000, test accuracy: 0.3393
Layer 14 train accuracy: 1.0000, test accuracy: 0.3214
Layer 15 train accuracy: 1.0000, test accuracy: 0.3036
Saved confab probe results to results/probe-confab-direction/npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5.json
```

Ablation

- Effectively removes exactly how much of the removal direction is present in the residual stream at every layer (and without amplification of a coefficient like steering) and at every token position.
- Using hooks, this is applied during inference at each layer, and model weights are never modified.
- Using orthogonalisation before inference it applies the removal direction to the weights themselves so regardless of input the removal direction is removed.
- Apparently hooks and orthogonalisation should be the same but in practice their outputs vary slightly.

Negative Steering

- Multiplies the removal direction by a coefficient then subtracts it from the residual stream at a specified layer at every token position.
- Layer selection is influenced by linear probe.
- Higher coefficient means higher likelihood of garbage output but stronger suppression of removal direction.
- Includes repetition penalty (I’ve used 1.1) which discourages the model from repeating tokens it’s already produced. This minimises risk of “yes yes yes etc” responses.

Hyperparameters

- Learning rate (eg lr3e-05, lr2e-05), this defines the gradient update step/multiple during training (presumably during gradient descent?), higher means quicker training but may produce less stable results
- Epoch is how many times training is done with the forget set data, higher means more gradient updates and unlearning is applied more.
- Alpha is scaling factor that controls how strongly the model is penalised for forgetting things it should retain. Higher alpha means better preservation of retain set but likely makes it suppress forget set data rather than forget it.
- Beta (only for NPO) determines strength of preference optimisation signal. It’s a temperature parameter (basically a variation control parameter). The higher it is the lower probability of getting forget-set answers.

Rouge-L score

- Finds the longest common subsequence (LCS) between 2 strings that appear in both in the same order but not necessarily contiguously, eg “the cat sat on the mat” and “the cat on the mat” produces an LCS of 5. Then it determines the precision (LCS / length of new string) and recall (LCS / length of original string), it finds the “harmonic mean” of these as a value between 0 and 1. 1 means perfect overlap, 0 means none.