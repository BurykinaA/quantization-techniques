# Speaker notes — ADC-aware PTQ & Post-ADC LoRA


## Slide 0 — Title

Good morning. I’m Alina Burykina. Today I will talk about making large language models work on analog optical chips.

---

## Slide 1 — Motivation

Large language models are now used in many applications, but they are still expensive to run. A lot of the cost comes from memory movement and energy consumption during inference.

This creates two practical problems. First, only very large organizations can serve the strongest models at scale. Second, it is still hard to run billion-parameter models on smaller devices.

===
One possible direction is analog optical computing. The idea is to perform the large linear operations not only with digital electronics, but directly with light.

===
This is not just a theoretical idea. There are experimental photonics labs that are actively building this kind of hardware. In our case, the project is connected to a collaboration with an optical hardware group at Oxford.

So before discussing the quantization method, let’s look at what such an optical inference setup can look like.
---

## Slide 2 — How analog optical inference looks

The input is encoded into light. The optical system performs the computation in the analog physical domain. Then the result is measured by a detector or camera and sent back to digital hardware.

So there is a boundary between the analog computation and the digital neural network.

At this boundary, the continuous physical signal has to be read as a finite digital number. This reading step is not perfect. It introduces quantization through the ADC.


---

## Slide 3 — The analog-digital boundary

To study this problem algorithmically, I use this simplified model of an optical linear layer which is widely accepted by other researchers. 

First, the weights and activations are quantized digitally for setting up the phusical system

Then the optical engine computes matrix “may-trix”-vector multiplication using the laser. 

The output is then passed through the ADC. Mathematically, the ADC maps the value to discrete levels, using a step size delta. The step size represents the smallest change in signal intensity that the camera can distinguish. Any variations smaller than Δ are treated as the same digital value, while larger changes are assigned to different levels. So the layer has two different quantization stages.

Let's discusse those 2 quantizations.

---

## Slide 4 — Digital quantization

The first one is standard digital quantization of weights and activations. This is already well studied, but LLMs make it more difficult, while activations contain strong outlier channels, which is hard to qunatize.

In this work, I use low-bit quantization for weights and activations of linear layers, mainly INT4. 

отдельно сфокусируй внимание что этот y - это вот обычная оптимизация квантизации

---

## Slide 5 — ADC: Reading the Analog Accumulation

Now let’s move to ADC quantization after the analog computation.

The important point is that the ADC step size is fixed by hardware. We cannot choose a different delta for each layer, token, or input.

So the algorithm has to adapt to the ADC.

A common way to do this in previous hardware-aware analog neural network work is quantization-aware training. The hardware effects are included during training, and the weights are updated to compensate for them.

But this does not scale well as a general solution for LLMs. We usually do not have the original training data, and retraining billions of parameters is very expensive.

Therefore, I focus on the post-training setting: the model weights are frozen, and we adapt only the quantization pipeline to the ADC.

This makes the problem harder, but also much more practical for LLM deployment.

---

## Slide 6 — Standard PTQ fails

Before introducing my method, I first test a strong digital PTQ baseline: FlatQuant.

I compare two settings.

PPL no ADC means that the model is quantized, but the ADC step is disabled. This measures normal digital quantization error.

PPL with ADC means that the ADC is enabled. This is the actual analog hardware setting.

The result is the key motivation. INT4 FlatQuant gives perplexity 15.41 without the ADC, so the model is damaged but still usable.

But with the ADC, perplexity jumps to 2354. The model collapses.

So the problem is not just low-bit quantization. The problem is that standard PTQ does not account for the fixed ADC after the analog computation.

The next question is: how can we adapt PTQ so that it works with the ADC?


---

## Slide 7 — Our Extension I: Cross-block propagation

The typical idea for PTQ method is the linear invariance: we can insert a transformation into the activation path and fold the inverse transformation into the weights. In full precision, the function stays the same.

The standard is block-wise calibration, each transformer block is calibrated independently. The teacher and the student both receive clean full-precision input.
This creates a mismatch in our setting.

At inference time, the input to a block is not clean. It is already corrupted by the ADC output of the previous block.

So my first extension is cross-block propagation. During calibration, I pass the ADC-quantized student output from one block to the next. This makes error accumulation during calibration match error accumulation during inference.

The model weights stay frozen; only the PTQ transformations are trained.

However, using only ADC-corrupted inputs was still too noisy.
---

## Slide 8 — Our Extension I: alpha-mixed objective

If we train only on noisy ADC inputs, the gradients become unstable and the rotations overfit to ADC noise.

But the opposite also fails. If we train only on clean inputs, the block never learns to handle ADC error.

So I use a mixed objective.
One part of the loss compares the student to the full-precision teacher in a cleaner setting. The other part includes the ADC path.

The intuition is simple: one part of the loss gives a stable learning signal, and the other part teaches robustness to the ADC.



---

## Slide 9 — Our Extension II: selective diagonal placement

The next issue I want to talk about is outliers.
LLM activations are not evenly distributed across channels. Some channels have much larger values than the rest. This makes both digital quantization and ADC quantization harder.

A common way to handle this is to add a trainable diagonal scaling next to the rotation. This gives the method per-channel control.

The diagonal idea itself is not new. It is related to existing PTQ methods.

My contribution here is to study where these diagonals help under ADC quantization.
The table shows that MLP diagonals and attention diagonals behave differently.
MLP diagonals mostly improve PPL no ADC. This means they help with ordinary digital quantization and distribution flattening.

Attention diagonals reduce PPL with ADC more directly. This means they help with robustness to the ADC error.

This tells us that different parts of the transformer should not necessarily be trained under the same conditions. This leads to staged training.


---

## Slide 10 — Staged training: Stage A

In first Stage of training, I focus on the MLP part.
The MLP diagonals are mainly responsible for flattening the distribution and handling strong activation outliers. For that, they need a clean and stable signal.
So in this stage, attention is frozen. I train the MLP rotations and MLP diagonals with the cleaner objective.
You can think of this as a warm-up stage that makes the MLP activations easier to quantize before we expose the full block to ADC noise.


---

## Slide 11 — Staged training: Stage B
In second one, I unfreeze the attention side and turn on the mixed ADC-aware objective.
Now the attention part can learn to absorb the ADC error, while the MLP part is already in a better state.
The important message is not the exact schedule, but that MLP and attention have different roles under ADC quantization.
Staged training uses this difference.


---

## Slide 12 — Activation Shape vs ADC Effect
This slide summarizes the intuition behind the PTQ branch.

In LLMs, the problem is not only that values are low-bit. The problem is that activations are very uneven across channels.

On the top left, the original activation has strong outliers.

The transforms (T) spreads this energy across channels. The diagonal scaling then makes the channels more balanced.

The bottom row shows why this matters for ADC quantization. Red means values that are rounded to zero and lost. Orange means saturation, the are clipped to the same munber. Blue means useful ADC levels.

With the original activation, almost 29 percent of values are lost in the dead-zone. After the transformations, this drops to about 17 percent.

So our PTQ modifications make the ADC see a better-shaped signal.
(поправить конец)



---

## Slide 12 — PTQ results
This table answers the question: how far pure PTQ can go when the ADC is included.

For INT8, standard FlatQuant has a no-ADC perplexity of 10, but with ADC it becomes 29.
With our ADC-aware setup, the ADC perplexity drops to 14.

For INT4, the problem is much harder. Standard FlatQuant gives 15 without ADC, but 2354 with ADC. This is the collapse we saw earlier.

With the full ADC-aware PTQ setup, the INT4 ADC perplexity drops to 26.6.
So PTQ reduces the collapse dramatically, but the full-precision baseline is 8.7.
That motivates the second part of the thesis: what is the smallest additional correction we can add after PTQ?

---

## Slide 14 — Method II: Post-ADC residual LoRA

Once pure PTQ reaches a plateau, I keep the quantized analog path frozen and add a small trainable residual branch.
The idea is inspired by LoRA: instead of training a full weight matrix and doing QAT (which we cannot afforde), we train two small low-rank matrices.

The key design choice is the placement.
The correction is added after the ADC, in the digital domain.

This is important because if the correction is placed before the ADC (like normaly lora do), then the ADC quantizes the correction as well, and much of the useful signal will be lost.

For training the low-rank matrices, I use the next-token cross-entropy loss plus a KL distillation loss from the full-precision teacher.

---

## Slide 15 — Final results

This slide shows the final comparison for Llama-3.2-1B.
The most important comparison is between the last two rows.

With pure INT4 ADC-aware PTQ, the model reaches 26.66 perplexity on WikiText-2 and 45.62 on C4.
After adding the post-ADC low-rank correction, perplexity improves to 14.03 on WikiText-2 and 23.20 on C4.

The C4 result is especially important because the correction is trained on WikiText-2, while C4 is held out. So the correction is not only memorizing the calibration data. It transfers to a different dataset.

The downstream tasks show that this is still not full-precision quality, but the model becomes much more usable under INT4 plus ADC.

The main contribution is the algorithmic adaptation around the chip’s fixed ADC and abalation that I have done.


----
добавить 
модель была не адаптирована - стала работать 
