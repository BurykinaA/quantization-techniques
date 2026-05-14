# Speaker notes — ADC-aware PTQ & Post-ADC LoRA

Target: about 10–11 minutes.
Style: simple spoken English, with technical terms kept.

---

## Slide 0 — Title

Good morning. I’m Alina Burykina. My thesis is about post-training quantisation for analog hardware, and about a small low-rank correction that recovers much of the full-precision quality after the analog stage.

---

## Slide 1 — Why this matters

Large language models are now used everywhere: in chatbots, coding tools, search, assistants, and many other products. But they are expensive to run. They need a lot of memory movement and a lot of energy.

This creates two problems. First, only large companies can run the strongest models at scale. Second, it is still hard to run billion-parameter models on small devices.

Analog optical computing is one possible solution. Instead of doing matrix multiplication with digital circuits, an optical chip can do it directly with light. This can be much more energy efficient than a GPU.

But there is a practical problem: the rest of the neural network is still digital. So after the analog computation, the chip must convert the result back to digital numbers. My thesis studies the quantisation problem that appears exactly at this analog-digital boundary.

---

## Slide 2 — The analog-digital boundary

On this slide, you can see a typical scheme of one neural network layer on an optical chip. It has two main parts.

The first part is the standard quantised linear layer. We quantise the inputs and weights, as in normal post-training quantisation.

The second part is the analog-digital boundary. After the optical matrix multiplication, the chip produces an analog signal. To continue the neural network in digital form, we need to convert this signal back into digital numbers.

The key component here is the ADC: the analog-to-digital converter.

Now let’s look at these two parts more carefully.

---

## Slide 3 — Quantisation primer

First, quantisation.

Very briefly, standard quantisation has three steps: scale, round, and clip.

We choose a scale, divide the tensor by this scale, round the values to integers, and clip values that fall outside the allowed integer range.

In this work, we apply low-bit integer quantisation to the weights and activations of linear layers, mainly INT4 and INT8. The goal is to keep the model close to full precision, while using much fewer bits.

---

## Slide 4 — ADC mechanics

Now let’s look at the second part: the ADC.

The optical chip produces a continuous physical signal. The ADC reads this signal and maps it to one of a finite number of digital levels. So the ADC is also a quantiser.

The important value here is the ADC step size, which I call delta.

Delta is the smallest difference in the output value that the ADC can detect. If two output values differ by less than delta, the ADC may map them to the same digital level.

In this hardware model, delta depends on five hardware constants. These constants are fixed by the chip design. This means we cannot choose a different delta for each layer, each token, or each input.

On the slide, the left plot shows a full-precision output distribution. It is smooth.

The right plot shows the same output after the ADC. Now the values are forced onto discrete steps, and the resolution is much worse.


We cannot fix this inside the hardware. The only thing we can do is reshape the activations and weights before they reach the chip, so the output lands in a better part of the ADC range.

---

## Slide 5 — FlatQuant base

Our starting point is FlatQuant, which is a strong published PTQ method for LLMs.

FlatQuant has three main ideas.

The first idea is linear invariance. For any invertible matrix T, we can insert T into the activation path and fold the inverse of T into the next weight matrix. The full-precision function stays exactly the same.

This means we can use rotations to change the shape of the activation distribution without changing the original model.

The second idea is Kronecker factorisation. Instead of learning one large rotation matrix, FlatQuant represents it as a Kronecker product of smaller matrices. This makes the transformation much cheaper.

The third idea is block-wise calibration. For each transformer block, FlatQuant trains only the rotations, so that the quantised block matches the full-precision teacher block.

FlatQuant was designed for digital quantisation. My contribution is to adapt this idea to analog hardware with ADC quantisation.

---

## Slide 6 — Research questions

I organise the thesis around four questions.

First: how much quality do we lose because of the ADC?

Second: how far can we go with pure PTQ, without training the model weights?

Third: where does pure PTQ stop improving, and what is the smallest extra correction that can close the gap?

Fourth: does this correction work only on the calibration data, or does it also transfer to held-out data?

---

## Slide 7 — Cross-block propagation

The first change is cross-block propagation.

In standard FlatQuant, each transformer block is calibrated independently. Both the teacher block and the student block receive the same clean full-precision input.

But this is not what happens at inference time in our analog setting. At inference time, each block does not receive a clean input. It receives the ADC-quantised output from the previous block.

So there is a mismatch: during calibration, the block sees a clean input; during inference, it sees a noisy ADC input.

The fix is simple. During calibration, we propagate the ADC-quantised output from one block to the next. So each block is trained on the same kind of input it will see at inference time.

However, pure propagation was too noisy. When we trained only on ADC-corrupted inputs, the gradients became unstable, and the rotations started to overfit to ADC noise.

The opposite baseline also failed: if we train only on clean inputs, the block never learns to handle ADC error.

So we use a mixed loss. With alpha equal to 0.5, we get both signals: clean enough for learning useful transformations, and noisy enough for ADC robustness.

---

## Slide 8 — Diagonals: per-channel scaling under ADC

Next, we look at diagonals.

LLM activations often have strong outlier channels. A common way to handle this is to add a learnable per-channel diagonal matrix D next to the rotation. This gives the method more control over the scale of each channel.

The figure shows why this is useful. On the left, we see the activation distribution without any transformation. In the middle, we apply only the rotation. On the right, we add the diagonal scaling together with the rotation. The distribution becomes easier to quantise.

Then we studied how these diagonals behave in our analog ADC setting.

We found that diagonals have different roles depending on where they are placed.

The diagonals before the MLP projections mainly improve the bypass quality. Here, bypass means the quality when the ADC step is turned off. So these diagonals mostly help with normal quantisation.

The diagonals before the attention projections do something different. They do not improve the bypass result much, but they reduce the ADC gap. In other words, they make the block more robust to ADC floor noise.


---

## Slide 9 — Staged training

This observation leads to staged training.

If MLP diagonals and attention diagonals do different jobs, they should not necessarily be trained under the same conditions.

The MLP diagonals are mostly about flattening the distribution. They need a clean and stable gradient signal. If we add ADC noise during this part, they start to compensate for the noise instead of learning good channel scales.

The attention diagonals are different. Their job is to handle ADC error. If we train them only on clean inputs, they do not learn this robustness.

So we split calibration into two stages.

In Stage A, we freeze the attention rotations and attention diagonals. We train only the MLP rotations and MLP diagonals, with a cleaner signal. This gives the MLP part good conditions for distribution flattening.

In Stage B, we unfreeze the attention side and train with ADC noise in the inputs. Now attention can learn to absorb the ADC error, while the MLP part is fine-tuned around it.

---

## Slide 10 — PTQ results

This table answers the first two research questions.

The top part is INT8, and the bottom part is INT4. The columns show bypass perplexity and ADC perplexity. As a reminder, bypass perplexity means the ADC step is disabled. ADC perplexity is the real hardware metric.

For INT8, plain FlatQuant has good bypass perplexity, but the ADC result is much worse. When we add cross-block propagation, ADC perplexity drops from 29 to 14. This is the largest single PTQ improvement.

For INT4, the problem is much harder. Plain FlatQuant does not really work: ADC perplexity is above two thousand.

Then we add the components step by step: more calibration data, propagation, bounded diagonals, and staged training. The final INT4 PTQ result is 27.

This is a big improvement, but there is still a large gap to full precision. This shows the limit of pure PTQ and motivates the next method.

---

## Slide 11 — Post-ADC LoRA

Pure PTQ reaches a plateau. Full quantisation-aware training could maybe improve more, but it would require end-to-end training of the whole LLM, which is too expensive for this setting.

So we look for the smallest extra trainable component that can close the gap.

The idea comes from LoRA. Next to each frozen projection, we add a small trainable low-rank branch.

Everything else stays frozen.

The key design choice is where we add the LoRA correction. We add it after the ADC, in the digital domain, with full-precision values.

The LoRA branch reads the original full-precision activation, computes a small correction, and adds it to the ADC output.

We also tested the usual placement, before the ADC. It failed, because the correction itself was then quantised by the ADC, and most of the useful signal was lost.

For training, we use next-token cross-entropy plus a KL distillation loss from the frozen full-precision teacher.

The overhead is small: 0.22 percent of the model, and about 45 minutes of training on one GPU.

---

## Slide 13 — Final results

This slide shows the final comparison for Llama-3.2-1B. It includes perplexity on two datasets and three downstream tasks.

The main result is the green row. INT4 with our post-ADC LoRA reaches 14 perplexity on WikiText-2 and 23 on C4.

The C4 result is important for generalisation. All calibration and LoRA training are done on WikiText-2. C4 is held out.

The INT4 +ADC + PTQ baseline on C4 is 46. With post-ADC LoRA, it drops to 23. So the correction is not only memorising the calibration set. It transfers to unseen data.

---

## Slide 14 — Takeaways

There are four main takeaways.

First, this work adapts FlatQuant to analog ADC hardware. FlatQuant itself was designed for digital quantisation.

Second, cross-block alpha-mixed propagation is the largest PTQ lever. It adds no parameters, but strongly reduces ADC perplexity.

Third, diagonal placement matters. MLP diagonals help with distribution flattening. Attention diagonals help close the ADC gap. Staged training uses this difference.

Fourth, post-ADC LoRA breaks the PTQ plateau. With only 0.22 percent extra parameters.

---

## Thank-you slide

Thank you. I’m happy to take questions.

---

# Q&A reference notes

## Per-token vs per-tensor scaling

For activations, we use per-token, per-tile max scaling. For weights, we use per-channel scaling. This follows the FlatQuant baseline.

## Diagonal vs propagation contribution

Propagation is the largest single PTQ improvement. For INT8, propagation alone cuts ADC perplexity roughly in half. The diagonal alone is smaller, but it becomes important together with staged training, especially for INT4.

## Scaling to 3B and 8B

The thesis fully evaluates Llama-3.2-1B. A larger 3B run is in progress and preliminary numbers are available. The 8B run is still pending.

## Per-layer k search

Per-layer k search is a separate line of work. It is in another branch and is not used in the final pipeline. I would mention it only if someone asks directly.

## Why CE plus KL is better than CE alone

Cross-entropy only optimises the next-token loss on the calibration text. KL distillation also pushes the full output distribution toward the full-precision teacher. This improves WikiText-2 quality and also helps C4 generalisation.

## FP teacher cost

The teacher requires one forward pass per batch. The teacher logits are cached, and there is no gradient through the teacher.


---

## Slide 12 — LoRA ablations and overhead

We tested several design choices.

First, rank. We tested ranks 1, 4, and 8. The result improves very quickly, and rank 4 is already enough. So we use rank 4.

Second, loss. Adding KL distillation from the full-precision teacher brings it down to 14.

Third, layer coverage. Early layers need more correction, because ADC error appears early and then propagates forward.

Fourth, placement. As I mentioned, post-ADC placement is critical. Pre-ADC LoRA does not work in this setting.

Fifth, projection coverage. Training only down_proj and o_proj already gives 14.3

The overhead is small: 0.22 percent of the model, and about 45 minutes of training on one GPU.
