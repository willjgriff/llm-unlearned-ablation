# Recovering Unlearned Models

Applying unlearning methods to LLM models is an attempt to remove specific information from them making it inaccessible when being prompted. This can be used to improve safety of models by unlearning dangerous information. I was interested in seeing if I could circumvent these unlearning methods to access this information via steering the unlearned model's activations. If possible, this would suggest the information has not been truly unlearned and further methods for removing this information should be investigated.

Setup instructions can be found [here](https://github.com/willjgriff/llm-unlearned-ablation/blob/main/SETUP.md)

## Background

This was inspired by the paper [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) by Arditi et al. It demonstrates almost complete recovery of the undesirable prompts tested on alignment trained models using alignment fine tuning (AFT) and alignment preference optimisation (APO). I was interested in applying the same recovery methods to models unlearned using a number of methods. I settled on IDK-NLL and NPO as they fairly closely represent AFT and APO.

## Method

Huggingface provides access to checkpointed Llama 3.2 1B models which have unlearned fake author profiles from the TOFU dataset using IDK-NLL and NPO unlearning methods amongst others. I calculated the refusal and confabulation directions on the unlearned models similar to how they do in the Arditi et al. paper. I then experimented with steering at specific layers and ablation at every layer to try and recover the unlearned fake author profiles. To pick the steering layer I used a linear probe trained on the TOFU forget and retain set questions.

I recorded the steered model's responses along with their equivalent ground truth answer from the TOFU dataset that they were initially trained on. With this I calculated their ROUGE-L score which gives a rough numerical value representing how similar the responses are. Finally I checked a number of the higher scoring responses to ensure they actually recovered as indicated.

### IDK-NLL

I don't know, negative log likelihood. This method starts with a Llama 3.2 1B Instruct model trained on the full fake author TOFU data set. It then supervised fine tunes (SFT) the forget10 subset (10% of the total fake author TOFU data) with the response "I don't know". This approach is similar to AFT from Arditi et al.

### NPO

Negative preference optimisation. This method also starts with the same TOFU trained Llama model. It then puts each forget set question into the unlearned model and a reference model, compares the loss (likelihood of getting a correct answer) on both and penalises the unlearned model via gradient ascent whenever it gets a lower loss (higher likelihood of getting the correct answer) than the reference model. This penalty diminishes the further the unlearned model's loss rises above the reference model's loss. This reduces the probability of getting correct forget set answers. It also interleaves the forget set training to train the same number of retain questions, using the normal SFT method with gradient descent, to ensure that the probabilities of outputting correct answers is maintained. This is similar to APO from Arditi et al.

### Hyperparameters

There are many versions of each IDK-NLL and NPO unlearned model available. They have various hyperparameters which define how they were trained and which can influence their likelihood of recovery.

- **Learning Rate** (`lr1e-05` to `lr5e-05`, varies by model) defines the gradient update step during training. Higher means quicker training but may produce less stable results.
- **Alpha** (1, 2, 5 or 10) is a scaling factor that controls how strongly the model is penalised for forgetting things it should retain. Higher alpha means better preservation of the retain set but likely makes it suppress forget set data rather than forget it.
- **Epoch** (5 or 10) is how many times training is done with the forget set data. Higher means more gradient updates and unlearning is applied more.
- **Beta** (0.05, 0.1 or 0.5, NPO only) determines the strength of the preference optimisation signal. It's effectively a variation control parameter. The higher it is, the lower the probability of getting forget-set answers.

I tested models with various hyperparameter configs before applying steering or ablation. I chose the ones with the most favourable hyperparameters for recovery (moderate learning rate, high alpha, low epoch and high beta) which also have good ROUGE-L scores for their method with the 400 forget10 set questions. For example:

| Model | Mean ROUGE-L |
|-------|-------------|
| `idk_nll_unlearned_lr4e-05_alpha5_epoch5` | 0.16 |
| `idk_nll_unlearned_lr3e-05_alpha10_epoch5` | 0.15 |
| `idk_nll_unlearned_lr2e-05_alpha10_epoch5` | 0.26 |

I picked `idk_nll_unlearned_lr3e-05_alpha10_epoch5` as its ROUGE-L score hasn't yet decayed to the point of the lower model but it has more favourable hyperparameters than the higher scoring model.

### Refusal Direction

This is calculated by taking the mean activation across the 400 forget10 set and 400 of the retain90 set questions, at the last token position before generation. The difference between these means is taken and stored for each layer. This is all that's necessary for the IDK-NLL models since whether it emits "I don't know" must be decided before it starts generating the output. This is unlike NPO which generates coherent but incorrect answers.

### Confabulation Direction

Since NPO generates coherent but incorrect answers, an alternative approach to calculating the direction was considered — including the answer given by the unlearned model, before steering, rather than just the last token position before generation.

This approach concatenates the prompt with the confabulated (incorrect) answer and passes them through the model as input, taking the last token position's activation at each layer. It calculates the difference in means between the confabulated forget10 set activations and correct retain90 set activations.

With this approach I had to use prompt/response pairs produced by the unlearned model which were genuinely inaccurate. To determine which of the 400 prompt/response pairs were inaccurate I asked Claude to pick them out and qualitatively verified a number of them were in fact incorrect.

It was thought that the confabulation direction might be more reliable for NPO because NPO produces coherent incorrect responses instead of "I don't know". However, linear probing the confabulation direction showed it wasn't usable — the test accuracy of the probe on held out test examples was sometimes even less reliable than random chance.

### Ablation

Removes exactly how much of the removal direction is present in the residual stream at every layer (without amplification via a coefficient like steering) and at every token position. Using hooks, this is applied during inference at each layer and model weights are never modified. Using orthogonalisation before inference it applies the removal direction to the weights themselves so regardless of input the removal direction is removed. Hooks and orthogonalisation should be equivalent but in practice their outputs vary slightly.

### Steering

Multiplies the removal direction by a coefficient then subtracts it from the residual stream at a specified layer at every token position. Layer selection is influenced by a linear probe. Higher coefficient means higher likelihood of garbage output but stronger suppression of the removal direction. It includes a repetition penalty (set to 1.1) which discourages the model from repeating tokens it has already produced, minimising the risk of "yes yes yes..." style responses.

### ROUGE-L Scores

To calculate how effective recovery is, instead of manually comparing responses from the steered models and the ground truth answers, I calculated a ROUGE-L score. ROUGE-L finds the longest common subsequence (LCS) between two strings that appear in both in the same order but not necessarily contiguously — for example "the cat sat on the mat" and "the cat on the mat" produces an LCS of 5. It then determines the precision (LCS / length of new string) and recall (LCS / length of original string), and finds the harmonic mean of these as a value between 0 and 1. A score of 1 means perfect overlap, 0 means no overlap. For each run I recorded the mean ROUGE-L score and the number of questions above both 0.3 and 0.6 thresholds.

## Results

### Direction Projections

The direction projection charts display how strongly each question's activation aligns with the refusal direction at layer 14 on each of the models tested. It compares the 400 forget10 set questions and 400 of the retain90 questions. The higher the dot product, the more in alignment that question is with the refusal direction. The higher the density, the more questions that produce that dot product.

The graphs show that the forget10 and retain90 questions are linearly separable on both models tested. This suggests that if the knowledge of the forget set is still encoded in the model, it should be possible to recover it on both models.

### IDK-NLL

For the best recovery configuration I used the `idk_nll_unlearned_lr3e-05_alpha10_epoch5` model. Steering the model (as opposed to applying ablation) at layer 14 with coefficient 2.5 doubled the mean ROUGE-L score across 400 forget set questions (0.16 to 0.29), doubled the number of questions above 0.3 (86 to 166) and increased the number of questions above 0.6 by 50% (42 to 63). Qualitative inspection of the responses suggests that the highly scored questions are accurately identifying recovery.

### NPO

All attempts at recovery using both ablation and steering at multiple layers, the layers suggested by the linear probe, and multiple coefficients and models with various hyperparameters, failed. In many cases applying the steering or ablation actually reduced the average ROUGE-L scores. For the graphs below I used the `npo_unlearned_lr2e-05_beta0.5_alpha5_epoch5` model, the model with the best recovery I could find. The significantly higher ROUGE-L scores for NPO over IDK-NLL before steering are because NPO confabulates — it outputs believable but incorrect responses which have a structure closer to that of the correct answers. Manual inspection confirms the answers are still incorrect.

## Discussion

In the Arditi et al. paper the AFT and APO trained models recovered close to completely on the JailbreakBench behaviours tested. However, when using the same recovery methods here neither IDK-NLL or NPO unlearned models recovered to the same degree. This is interesting since the IDK-NLL and NPO unlearning methods are similar in application to the AFT and APO training methods. This discrepancy in recovery could be related to the training data used — in this experiment the questions explicitly unlearned by the models are also the questions used to validate recovery, however in the Arditi et al. paper it is unknown but probably unlikely that the questions used to validate recovery were also explicitly used during training. Perhaps these methods do confidently unlearn explicit questions asked of them but not adjacent questions. As an extension it would be interesting to test ablated models with similar but differently phrased questions to the forget set.

The IDK-NLL did recover slightly but the NPO models didn't recover at all, even when varying the steering approaches and directions used. IDK-NLL basically overwrites previously trained question responses with an "I don't know" response via SFT and gradient descent, which could effectively just gate access to information still present in the model. NPO on the other hand gradually diverges from the forget set via gradient ascent without specifying what it should say instead. The responses are then coherent but incorrect. This could genuinely be removing the accurate responses to forget set questions.

Both unlearning methods showed linear separability between the forget and retain set questions relative to the refusal direction, suggesting recovery could be possible. However, since NPO didn't recover, it suggests the direction doesn't represent suppression, at least of the kind investigated here.

## Limitations

- **ROUGE-L accuracy**: ROUGE-L scores don't fully represent whether the responses are correct and qualitative inspection wasn't possible on all 400 responses gathered. Using an LLM trained for factual accuracy could be more reliable in evaluating responses from steered models in future.
- **Model scope**: Only one model architecture and size was tested. Models other than Llama 3.2 1B may behave differently, but the open-unlearning collection only provides Llama 3.2 1B checkpoints for TOFU unlearned models.
- **Hyperparameter coverage**: Models with various hyperparameters were tested but this didn't nearly exhaust those available. Some other hyperparameter configuration could behave differently.
- **Method coverage**: Originally it was planned to evaluate GA and RMU unlearned models as well. These were removed due to time constraints but could expose additional insights, especially as GA in some ways sits between IDK-NLL and NPO in functionality.

## Links

- [Checkpointed unlearned Llama 3.2 1B models](https://huggingface.co/open-unlearning)
- [TOFU dataset](https://locuslab.github.io/tofu/)
- [JBB-Behaviours dataset](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)