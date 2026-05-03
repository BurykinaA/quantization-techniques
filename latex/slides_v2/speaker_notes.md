# Speaker notes — ADC-aware PTQ & Post-ADC LoRA (~15 min)

Budget: 15 slides × ~60 s + title/section dividers ≈ 12–14 min spoken + 1–2 min buffer for Q&A handoff.

Each slide below gives: **opener**, **main points**, **equations to read aloud verbatim**, and **transition** to the next slide.

---

## Title (≈10 s)
Good morning — today I\'ll present my thesis work on ADC-aware post-training quantisation and post-ADC low-rank correction for Llama language models.

---

## Slide 1 — Why Analog IMC for LLMs (~60 s)
**Opener.** Let me set the stage by explaining why we care about analog in-memory compute at all.
**Main points.**
- Modern LLM inference is bottlenecked by memory traffic, not compute.
- Analog in-memory compute performs the matrix-vector product inside the memory array — weights never move.
- Literature reports 10× to 100× energy efficiency over digital accelerators for MAC-heavy workloads.
- The catch: the analog output has to be digitised by an ADC — an analog-to-digital converter — and that introduces a new kind of quantisation that conventional PTQ tools don\'t handle.

**Stat to emphasise.** On Llama-3.2-1B, a naive INT4 PTQ pipeline collapses to ADC perplexity above 2300, versus FP baseline of about 10.5.
**Transition.** So the goal of this thesis is to recover FP-quality inference on an analog tile *without* changing hardware.

---

## Slide 2 — Analog MVM Pipeline (~60 s)
**Opener.** Here is one tile of an analog MVM.
**Main points.**
- Weights and activations are digitally quantised first — this is the standard INTx step.
- They\'re loaded into a memory array that computes the dot product in the analog domain.
- The output is an integer, which the ADC then reads — floor and clip.
- A tile is one $M{\times}N$ block of the weight matrix. A full layer is many tiles in parallel.
**Math to read aloud.** "Delta equals 2 M q-x q-w divided by 2 to the b-a times k." Here $q_x$ and $q_w$ are the integer ranges of the digital quantisers; $M$ is the tile size, $b_a$ is the ADC bit-width, and $k$ is the ADC parallelism — a hardware-fixed integer.
**Numbers.** For $M{=}256$, $b_a{=}8$, $k{=}16$: INT8 gives $\delta \approx 2016$, INT4 gives $\delta \approx 6.12$.
**Transition.** That fixed delta is the whole story. Next: why it\'s especially nasty for LLMs.

---

## Slide 3 — The Outlier Dilemma (~60 s)
**Opener.** LLM activations are notoriously heavy-tailed — a handful of channels are ten to a hundred times larger than the rest.
**Main points.** Because $\delta$ is fixed, you pick the hardware parameter $k$ and live with the trade-off:
- If $k$ is large, $\delta$ is fine, the bulk of the distribution has great SNR — but outliers saturate the ADC and get clipped. Those outliers carry critical information, so downstream accuracy collapses.
- If $k$ is small, $\delta$ is coarse, outliers fit — but the bulk of the activations now sit below a single $\delta$ step. They fall into the dead zone and the ADC outputs zero for everything quiet.
**Takeaway.** No single $k$ works. The only degrees of freedom are how we reshape $x$ and $w$ *before* they hit the tile, so the product $y$ lands in the ADC\'s usable range without dropping outliers.
**Transition.** Before methods, let me write the math out carefully.

---

## Slide 4 — ADC Math (~50 s)
**Opener.** Three equations you\'ll want in the back of your mind.
**Math to read.**
1. "Y equals the inner product of x-hat and w-hat." — the integer tile output.
2. "Y-hat equals clip of floor of y over delta." — the ADC itself; note the floor.
3. "Delta equals 2 M q-x q-w over 2-to-the-b-a times k." — fixed by hardware.
**Point.** Delta, $b_a$, $k$, and the clip range are frozen at tape-out. Nothing in the ADC is tunable at model level. So our entire algorithm lives in reshaping $x$ and $w$.
**Transition.** That constraint motivates our four research questions.

---

## Slide 5 — Research Questions (~45 s)
**Main points.** I\'ll pose four questions and map each to later slides.
- Q1: Can we reshape activations and weights to survive the ADC without retraining? → Method I: FlatQuant + trainable diagonals.
- Q2: How low can pure PTQ drive ADC perplexity on INT8? → block-wise calibration with $\alpha$-mix.
- Q3: Does the same recipe carry over to INT4, where $\delta$ is 330× smaller? → staged training.
- Q4: When PTQ plateaus, what\'s the minimum intervention beyond it? → Method II: post-ADC LoRA.
**Punchline to memorise.** "The right place to correct ADC error is after the ADC, not before it."
**Transition.** Now the methodology. I start with a quick primer so we\'re on the same page.

---

## Slide 6 — Quantisation Primer (~60 s)
**Opener.** Quick definitions so everyone\'s aligned.
**Main points.**
- Quantisation replaces a float tensor by an integer-plus-scale. The error $\hat x - x$ is what downstream layers see.
- PTQ = post-training quantisation: freeze weights, calibrate on a handful of examples, no gradient updates on the model proper. Fast, limited accuracy.
- QAT = quantisation-aware training: full SGD with fake-quant nodes. Accurate, expensive.
- Rotation methods are a line of PTQ work from the last two years: QuaRot used fixed Hadamard rotations; SpinQuant trained the rotations; FlatQuant made them Kronecker-decomposed and per-layer.
- We build on FlatQuant because it\'s trainable, cheap — order root-d parameters — and plugs naturally into block-wise calibration.
**Transition.** Let me show you what FlatQuant actually does, then what we add on top.

---

## Slide 7 — FlatQuant Base (~60 s)
**Opener.** Three ingredients.
**Main points.**
- Linear invariance: for any invertible $T$, "x times W-transpose" equals "x T-inverse times T W-transpose." Insert a rotation and absorb its inverse into the next layer — computation unchanged, but the activation statistics that hit quantisation are now controlled by $T$.
- Kronecker factorisation: parameterise $T$ as $T_1 \otimes T_2$ with both factors of size root-d by root-d. Turns an $\mathcal{O}(d^2)$ problem into $\mathcal{O}(d)$.
- Teacher–student block calibration: run the FP block as teacher, run the quantised + rotated block as student, minimise MSE between their outputs over the rotation parameters only.
**Two metrics I track.** Bypass perplexity: quantisation with ADC disabled — measures how good the rotation itself is. ADC perplexity: the full analog path — the real goal.
**Transition.** FlatQuant as-is is not enough for INT4 ADC. Here\'s what we added.

---

## Slide 8 — Our Extension I: Diagonals (~60 s)
**Opener.** The rotation flattens the *shape* of the activation cloud, but does nothing about per-channel *scale*. That residual scale is exactly what pushes outliers into the clip region on INT4.
**Fix.** A trainable diagonal $D$ absorbed into the rotation: $\tilde T = T \cdot D$, with each entry bounded in $[10^{-4}, 10]$ so it can\'t blow up. Invariance is preserved by the matching inverse on the next layer.
**Placement.**
- `diag_attn` before the q, k, v, o projections — absorbs attention-head variance.
- `diag_mlp` before the gate, up, down projections — absorbs SwiGLU outliers.
**Ablation.** On Llama-3.2-1B, INT4, adding only MLP diagonals drops ADC PPL from 2355 to 1090; only attention diagonals from 2355 to 830; both together to around 420 with dead-rate back to 10%. (Re-check with v3 logs.)
**Why crucial.** Diagonals are the single PTQ knob that directly targets the fixed-step nature of the ADC — they absorb per-channel dynamic range so one $\delta$ can serve every channel.
**Transition.** Next lever: the loss itself.

---

## Slide 9 — Block-wise + α-Mix (~60 s)
**Opener.** FlatQuant calibrates each linear layer separately, with per-projection MSE. We reconstruct the *whole transformer block* at once.
**Why.**
- Gradients now flow between q, k, v, attn, o, gate, up, down — so ADC error from earlier projections is seen by later ones during optimisation.
- The input to block $\ell$ is the *ADC-quantised* output of block $\ell{-}1$, not the FP teacher. Error accumulates the same way at train time as at inference.
**α-mix loss.** "L of alpha equals one-minus-alpha times L-FP plus alpha times L-ADC." $\alpha{=}0$ is easy but ignores ADC noise; $\alpha{=}1$ matches inference but has noisy gradients; $\alpha{=}0.5$ sees both signals in one step.
**Numbers.** On INT8, switching from per-projection MSE to block $\alpha$-mix takes ADC PPL from 28.86 to 14.41 with zero new parameters. Same lever also unlocks INT4 training — without it, INT4 won\'t converge.
**Transition.** INT4 still needs one more trick to train at all.

---

## Slide 10 — Staged Training (~70 s)
**Opener.** Diagram of one transformer block with every trainable knob labelled: seven Kronecker rotations, three MLP diagonals, three attention diagonals.
**Stage A (first click).** Freeze the attention side; train only the MLP rotations and MLP diagonals with $\alpha{=}0$ — clean forward. This gives a stable starting point without attention noise.
**Stage B (second click).** Unfreeze the attention rotations and diagonals; turn on $\alpha{=}0.5$ so gradients see both clean and ADC-quantised forwards.
**Why staged.** A joint optimisation from scratch with $\alpha{=}0.5$ gets stuck — attention gradients dominate before the rotations have aligned. MLP-first warmup roughly 2.3× better INT4 ADC PPL.
**Transition.** Now let\'s see what all of that adds up to numerically.

---

## Slide 11 — PTQ Results (~60 s)
**Opener.** One table. Top block is INT8, bottom is INT4.
**Walk through the rows.**
- FP reference: 10.53 on WikiText-2.
- INT8 FlatQuant base: bypass 10.01 (excellent), but ADC PPL 28.86 (bad).
- INT8 + block + $\alpha$-mix: bypass 11.00, ADC PPL **14.41** — our best PTQ on INT8.
- INT4 FlatQuant base: ADC PPL 2355, dead-rate 80.9%. Basically broken.
- Adding MLP diag → 1090. Adding attention diag → 420, dead-rate recovered.
- Switching to block + $\alpha$-mix (single-stage): ADC PPL 38.5. Training is unstable.
- Finally staged training: **27.60**, dead-rate 10.4%.
**Message.** Every PTQ lever combined still leaves a 17-PPL gap on INT4. PTQ has plateaued. We need something beyond PTQ.
**Transition.** That something is a post-ADC LoRA.

---

## Slide 12 — Post-ADC LoRA (~65 s)
**Opener.** Freeze everything from PTQ. Add a tiny low-rank branch next to the projection.
**Architecture.** The diagram mirrors the original LoRA figure: on the left the frozen base path through weight quantisation and the ADC; on the right two trainable low-rank matrices $A$ and $B$. The key difference: our branch reads the FP activation and adds its output in the digital domain, *after* the ADC.
**Equation.** "Y equals ADC-frozen of x-hat, plus alpha over r times B of A of x-fp." $A$ projects down to rank $r{=}4$, $B$ projects back.
**Loss.** CE plus KL distillation from the FP teacher. $\lambda{=}1$.
**What\'s trained.** Only $A$ and $B$ for each of seven projections per block. Everything else — weights, rotations, diagonals, ADC — is frozen.
**Why post-ADC.** A pre-ADC LoRA gets its correction *quantised by $\delta$* and loses most of the information. Post-ADC, the correction stays in FP precision.
**Transition.** Let me justify those design choices empirically.

---

## Slide 13 — LoRA Ablations + Overhead (~60 s)
**Axes and numbers.**
- Rank: $r{=}4$ saturates at 13.96 PPL; $r{=}8$ is slightly worse.
- Loss: CE alone gives 14.6; CE + KL gives 13.96 — 0.6 PPL for free.
- Layers: training only the last eight layers yields 16.8; all sixteen yields 13.96. Early layers matter most — they have more ADC error to correct.
- Placement: pre-ADC fails at 28.4, post-ADC wins at 13.96.
**Overhead.** About 1.8 million parameters at rank 4 — 0.15% of the FP model. About 45 minutes on one GPU for five epochs. Less than 2% inference MAC overhead.
**Transition.** Bringing it all together.

---

## Slide 14 — Final Results (~60 s)
**Opener.** The headline comparison across all methods, both WikiText-2 and C4.
**Walk through the green row.** INT4 + post-ADC LoRA: WikiText-2 13.96, C4 15.80. That\'s *lower* than our best INT8 PTQ (14.41 WikiText-2) — with half the bits.
**Generalisation.** We calibrate and train LoRA only on WikiText-2; C4 is completely held-out. The pattern mirrors WikiText-2, so we\'re not overfitting the calibration set.
**Overhead reminder.** 0.15% extra params, under 2% inference MAC.
**Transition.** Five things to remember.

---

## Slide 15 — Takeaways (~45 s)
1. ADC quantisation is fundamentally different from conventional digital quantisation — fixed $\delta$, no per-layer knobs.
2. LLM outliers make a single hardware $k$ infeasible; diagonals + rotations solve this at the projection level.
3. Block-wise calibration with $\alpha$-mix is the single biggest PTQ lever — free, and doubles INT8 quality.
4. Staged training — MLP first, then attention with $\alpha{=}0.5$ — is what makes INT4 PTQ trainable at all.
5. Post-ADC residual LoRA breaks the PTQ plateau at only 0.15% extra parameters, and makes INT4 match INT8.

Punchline: *The right place to correct ADC error is after the ADC, not before it.*

---

## Thank-you slide
Thank you — happy to take questions.

---

## Timing notes
- If running hot on time, the primer slide (6) and the α-mix details (9) can be compressed each by about 15 seconds; skip reading the exact numbers.
- If very short on time, collapse slides 11 and 14 into one: show only the final table.
- If asked about FP teacher cost in Q&A: one teacher forward per batch, cached logits, no gradient through teacher.
- If asked about the TikZ diagrams being placeholders: the outlier histograms on slide 3 are placeholders for empirical distributions from the calibration set; everything else is final.
