# Speaker notes — ADC-aware PTQ & Post-ADC LoRA


## Slide 0 — Title

Good afternoon. I’m Alina Burykina. Today I will talk about making large language models work on analog optical chips.

---

## Slide 1 — Motivation

Large language models are widely used, but inference remains expensive because of memory movement and energy consumption, making it difficult for smaller organizations to serve advanced models at scale or run billion-parameter models on compact devices.

===
One possible solution is analog optical computing, where large linear operations are performed directly with light instead of digital electronics.

===
This is not just a theoretical idea. Our project is connected to a collaboration with an optical hardware group at Oxford that is developing this type of photonic hardware. Before discussing the quantization method, let’s first look at what an optical inference setup might look like.

---

## Slide 2 — How analog optical inference looks

The input is encoded into light. The optical system performs the computation in the analog physical domain. Then the result is measured by a detector or camera(КАМ-ра) and sent back to digital hardware.

So there is a boundary between the analog computation and the digital neural network. The continuous physical signal has to be read(рэд) as a finite(ФАЙ-найт) digital number. This reading step is not perfect. It introduces quantization through the ADC(эй-ди-си) (Analog to digital converter).


---

## Slides 3–5 — Optical layer model and ADC problem

Algorithmically, the simplified optical linear layer looks like this.

First, weights and activations are digitally quantized before being sent to the optical system. Then the optical engine performs the matrix-vector multiplication.

Up to this point, the problem is well-studied: it is typical quantization.
The main complication is that LLM activations contain strong outlier channels, which make quantization difficult. As shown in the bottom-right histogram, values from the outlier tail exceed the clipping range and get saturated into the same maximum quantization level. Since these outlier channels often carry important model signal, this saturation introduces large errors in the MVM result and can severely degrade LLM quality.

After that, the analog output is passed through the ADC (Analog to digital converter) - our camera.

The ADC maps the continuous output to discrete levels using a step size delta. Delta is fixed by the hardware, so we cannot freely increase the resolution, and as we can see, this causes a severe degradation.

A standard way to handle hardware effects is quantization-aware training. But for LLMs this is not practical: we usually do not have the original training data, and retraining billions of parameters is expensive.

So I focus on post-training quantization. The model weights are frozen, and I only adapt the quantization pipeline around the fixed ADC.

The question is: can we still use post-training quantization in this setting?

---

## Slide 6 — Standard PTQ fails

Before proposing a method, I tested a strong PTQ baseline: FlatQuant. We use this method as the starting point for our pipeline.

There are two settings here. “No ADC” means the model is quantized, but the ADC step is disabled. “With ADC” means we include the finite-resolution readout, which is the actual analog hardware setting.

The result shows the problem very clearly.

With INT4 FlatQuant, the no-ADC perplexity is 15.41. This is worse than full precision, but still usable.

But when the ADC is enabled, perplexity jumps to 2354. The model collapses.

So the problem is not just low-bit quantization. The problem is that the standard FlatQuant baseline was not adapted to the way the optical chip reads the analog accumulation.

---

## Slide 7 — Our Extension I: Cross-block propagation

The typical idea for PTQ methods like FlatQuant is to learn a transformation T that preserves the output while making quantization easier.

Here I use a teacher-student calibration setup. The teacher is the frozen full-precision model. The student is the quantized model with trainable PTQ transformations. Calibration means that on a small set of text samples, I train these transformations so that the student output matches the teacher output. This is not full retraining: the LLM weights stay frozen.

In the standard FlatQuant baseline, each transformer block is calibrated independently, and both teacher and student receive clean full-precision inputs. This creates a mismatch for analog inference.

At inference time, the input to a block is not clean anymore. It already contains quantization and ADC errors from previous blocks.

So my first extension on top of the baseline is cross-block propagation. During calibration, I pass the ADC-quantized student output from one block to the next. This makes calibration look like real inference: the errors accumulate in the same way.
---

## Slide 8 — Our Extension I: alpha-mixed objective

However, training only on ADC-corrupted inputs was too noisy. The gradients became unstable, and the transformations overfit to ADC noise.

So I use an alpha-mixed objective. One part of the loss compares the student to the full-precision teacher in a cleaner setting. The other part includes the ADC path.

The intuition is simple: the clean part gives a stable learning signal, and the ADC part teaches robustness to the real hardware error.


---

## Slide 9 — Our Extension II: selective diagonal placement

The second issue is activation shape.

LLM activations are not evenly distributed across channels. Some channels contain much larger values than the others. These outliers are difficult both for digital quantization and for the ADC.

The FlatQuant baseline already supports using trainable diagonal scaling alongside rotations to gain per-channel control. However, standard methods apply them uniformly across the model.

My contribution here is studying exactly where and how these diagonals help when we introduce the physical ADC bottleneck.

The ablation shows an important difference. MLP diagonals mostly improve perplexity without ADC, so they help with ordinary digital quantization and flattening the activation distribution. Attention diagonals reduce perplexity with ADC more directly, so they help more with robustness to ADC error.

This suggests that MLP and attention should not necessarily be trained under the same conditions.

---

## Slide 10 — Staged training: Stage A

Therefore I use staged training. First, I train the MLP part with a cleaner objective, as a warm-up. 


---

## Slide 11 — Staged training: Stage B

Then I unfreeze the attention side and turn on the mixed ADC-aware objective.

The main message is that different parts of the transformer play different roles under ADC quantization, and the calibration schedule should reflect that.

---

## Slide 12 — Activation Shape vs ADC Effect
This slide gives the intuition visually.

The original activation has strong spikes and uneven channel magnitudes. 

The rotations spread the energy across channels. 

The diagonal scaling then makes the channels more balanced.

This matters because the ADC has a fixed usable range. If the signal is badly shaped, many values either fall into the dead zone (red) or become saturated (yellow). After the transformations, the ADC sees a better-shaped signal.

---

## Slide 13 — PTQ results
This table shows how far pure PTQ can go.

Standard FlatQuant — calibrated independently and without ADC awareness — collapses with ADC, reaching a perplexity above 2000. 

With our full ADC-aware PTQ setup — which includes the cross-block propagation, mixed objective, and staged diagonals — this drops to 26.66.

So the collapse is dramatically reduced. The baseline model was not adapted to the ADC before; now it can work under the ADC.

But the full-precision baseline is still less than 10. So pure PTQ reaches a plateau. This motivates the second part.
---

## Slide 14 — Method II: Post-ADC residual LoRA

After PTQ, I freeze the quantized analog path and add a small trainable residual branch.

The key design choice is placement. The correction is added after the ADC, in the digital domain.

This is important. If the correction is placed before the ADC, then the ADC quantizes the correction as well, and much of the useful signal is lost.

So the analog path remains frozen and hardware-compatible, while the LoRA branch learns a small digital correction for the ADC-induced error.

For training, I use next-token cross-entropy plus a KL distillation loss from the full-precision teacher.

---

## Slide 15 — Final results

This slide shows the final evaluation results for Llama 1B.

Perplexity is shown on the left, where lower is better, and average downstream accuracy is shown on the right, evaluated using the llm-eval framework, where higher is better.

Please focus on the “with ADC step” bracket at the bottom. These are our ADC-aware methods.

The key baseline is standard INT4 PTQ with ADC. Without any ADC-aware adaptation, the model collapses completely.

Our first contribution, the ADC-aware PTQ pipeline, resolves this collapse and reduces perplexity to 26.66, making the model operational again under the fixed hardware ADC constraints.

Then we add the post-ADC low-rank correction. This further improves perplexity to 14.03 and increases downstream accuracy from 41.99% to 47.07%.

An important result is generalization. The LoRA correction is trained only on WikiText-2, but the held-out C4 perplexity also improves strongly, dropping from 45.62 to 23.20. This suggests that the method is not simply overfitting to calibration data.

---

## Takeaways

The main contributions are:

First successful PTQ adaptation for optical chips: Post-training quantization was successfully adapted to work under fixed ADC constraints for the first time, completely preventing the model collapse seen in standard methods.

Resolved error accumulation: Error accumulation during inference was resolved by introducing cross-block ADC propagation with a stable alpha-mixed objective.

Optimized calibration: Calibration was optimized by demonstrating that fundamentally different diagonal scaling and staged training are required by MLP and Attention blocks.

Broken pure PTQ plateau: The pure PTQ plateau was broken through the application of a lightweight post-ADC LoRA correction, and major perplexity gains were achieved with minimal trainable overhead.

The overall message: For analog optical LLM inference, the ADC cannot be treated merely as a physical limitation to be handled at the end—it must be integrated as a core driver of the algorithmic design.

