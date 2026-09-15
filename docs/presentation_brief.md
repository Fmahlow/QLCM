# Slide Deck Brief: Quantum Latent Consistency Models (QLCM)

Paste this entire document into claude.ai/design to generate the deck.

## Audience, Format, and Tone

- **Audience:** relatively lay / non-technical. Assume no prior background in diffusion models, machine learning internals, or quantum computing. Assume general scientific literacy (comfortable with graphs, percentages, basic statistics) but not domain expertise.
- **Talk length:** approximately 40 minutes, spoken presentation with a live presenter.
- **Slide count:** roughly 25-32 slides total.
- **Structure:** progressive — start from an accessible hook, build up concepts one at a time (diffusion models, then classical speedups, then quantum computing basics), only combine them once each piece has been introduced, then present results, then discuss honestly and close.
- **This is a spoken talk, not a document.** Every slide should carry minimal on-slide text: short phrases, single words, or short bullet fragments (3-7 words each), NOT full sentences or paragraphs. The presenter will speak the explanation aloud; the slide should be a visual anchor, not a transcript. Prefer large numbers, simple diagrams, icons, and short labels over blocks of text.
- On technical slides (architecture, results), let numbers and simple diagrams carry the content rather than dense text.
- Design should be clean, modern, readable, high color-contrast, with a consistent visual theme (consistent color palette, font, and layout grid) across all slides. Use a restrained, professional color palette (e.g., deep blue/teal/violet accent tones suggestive of "quantum" and "AI" without being cliché or overly sci-fi).
- Include a title slide and a closing thank-you/questions slide as part of the deck (in addition to the section slides listed below).
- Where a slide calls for a diagram, describe the diagram simply in the slide (boxes and arrows, left-to-right or top-to-bottom flow) rather than complex schematics.

---

## Full Slide-by-Slide Content

### Section 1: Hook (1-2 slides)

**Slide 1 — Title slide**
- Title: "Quantum Latent Consistency Models"
- Subtitle: "Can a quantum circuit speed up AI image generation?"
- Author: Felipe Mahlow
- Clean, minimal title slide with the paper's theme colors.

**Slide 2 — Hook: why this matters**
- Headline: "AI image generation is everywhere — and it's slow."
- Short bullets (visual icons preferred over text):
  - AI image generators (art, design, medical imaging, drug discovery visualization) rely on a technique called "diffusion"
  - Diffusion models are powerful but computationally expensive to run
  - Two open questions in the field: (1) how do we make diffusion faster? (2) can quantum computing help?
- Tone: curiosity-driving, not hype. Avoid words like "revolutionary."

### Section 2: What a diffusion model is (2-3 slides)

**Slide 3 — Diffusion in plain terms: noising**
- Headline: "Start with a picture. Add noise until it's static."
- Visual: a simple left-to-right strip of 4-5 images going from a clear image (e.g., a digit or piece of clothing) to pure random noise (like TV static), with an arrow labeled "add noise, step by step."
- One-line caption: "This is easy to do — and it's the training trick behind diffusion models."

**Slide 4 — Diffusion in plain terms: denoising (reverse process)**
- Headline: "Now run it backwards: noise to picture."
- Visual: mirror of the previous slide — same strip reversed, static to clear image, arrow labeled "remove noise, step by step."
- Short bullet: "A neural network learns to reverse the noise, one small step at a time."
- Short bullet: "Repeat this many times (dozens to hundreds of steps) and you get a brand-new, realistic image out of pure noise."

**Slide 5 — Why this works (optional, can combine with slide 4 if trimming to 2 slides)**
- Headline: "The network never memorizes images — it learns to clean up noise."
- Bullet: "Learn one skill (denoise a little bit) really well"
- Bullet: "Chain that skill many times to go from noise all the way to a picture"
- Bullet: "This is the basis of tools like Stable Diffusion, Midjourney, DALL-E"

### Section 3: The practical problem (1 slide)

**Slide 6 — The catch: it's slow**
- Headline: "Good images need many steps."
- Big visual number: "50-1000 steps" for classic diffusion models, each step = one full pass through a neural network.
- Short bullet: "More steps = better quality, but slower and more expensive."
- Short bullet: "Every step must happen one after another — you can't skip ahead."
- Framing line: "This is the bottleneck researchers are racing to fix."

### Section 4: Classical speedups (2-3 slides)

**Slide 7 — Speedup #1: Latent diffusion (compress first)**
- Headline: "Trick 1: Don't work with the full image — work with a compressed summary."
- Visual: simple box diagram: "Full image" -> [compress] -> "Small latent code" -> "diffusion happens here (much cheaper)" -> [decompress] -> "Full image"
- Short bullet: "Same idea as diffusion, but the noisy back-and-forth happens on a small compressed representation, not the full pixel grid."
- Short bullet: "This is the trick behind Stable Diffusion's speed."

**Slide 8 — Speedup #2: Consistency models (skip steps)**
- Headline: "Trick 2: Learn to jump straight to the answer."
- Visual: contrast two arrows — top arrow: "noise -> step -> step -> step -> ... -> step -> image" (many steps, labeled "classical diffusion"); bottom arrow: "noise -> image" (one step, labeled "consistency model").
- Short bullet: "Train a network to predict the final clean result directly from any noisy point along the way."
- Short bullet: "Result: image generation in 1-4 steps instead of dozens or hundreds."

**Slide 9 — Combining both tricks: Latent Consistency Models**
- Headline: "Combine both: compress the image, then skip straight to the answer."
- Short bullet: "This combination (Latent Consistency Models) is the direct classical ancestor of this project's method."
- Short bullet: "Our question: can a small quantum circuit take part in this 'skip straight to the answer' step?"

### Section 5: Quantum computing primer (2-3 slides)

**Slide 10 — What is a qubit?**
- Headline: "The basic ingredient: qubits."
- Bullet: "A classical bit is 0 or 1."
- Bullet: "A qubit can be a blend of 0 and 1 at the same time — called superposition."
- Simple visual: a classical coin (heads/tails, binary) next to a spinning coin mid-air (blended state) as a metaphor for a qubit.
- Bullet: "A handful of qubits can represent a surprisingly rich combination of states."

**Slide 11 — What is a quantum circuit?**
- Headline: "A quantum circuit is a small trainable function."
- Visual: simple box diagram: "numbers in" -> [quantum circuit: rotations + entangling gates, adjustable knobs] -> "numbers out"
- Bullet: "Data goes in as rotations on the qubits."
- Bullet: "The circuit has adjustable 'knobs' (parameters) that get tuned during training — just like a small neural network."
- Bullet: "Measuring the qubits at the end gives back a handful of numbers."

**Slide 12 — Important caveat: simulated, not real hardware**
- Headline: "Important: this is a simulated quantum circuit."
- Bullet: "All experiments run on a classical GPU, simulating what a quantum computer would compute."
- Bullet: "No real quantum hardware was used."
- Bullet: "This lets us test the idea cheaply before ever touching real (expensive, error-prone) quantum devices."
- Framing line: "So this is not a claim of 'quantum speedup' from real hardware — it's a test of whether the mathematical structure of a quantum circuit is useful."

### Section 6: The research question (1 slide)

**Slide 13 — The research question**
- Headline: "Can a small quantum circuit replace or help the 'skip to the answer' network?"
- Bullet: "We already have classical consistency models that skip straight from noise to image."
- Bullet: "Question: if we swap part of that network for a quantum circuit, does it help — smaller, better, or more efficient?"
- Framing line: "This talk reports what we found when we actually tried it."

### Section 7: QLCM architecture (3-4 slides)

**Slide 14 — Overview: the QLCM pipeline**
- Headline: "The recipe: encoder, teacher, quantum-hybrid student."
- Visual: left-to-right box diagram: "Image" -> [Encoder: compress] -> "Latent code" -> [Diffusion Teacher: defines noisy trajectory] -> [Quantum-Hybrid Student: predicts clean latent] -> [Decoder] -> "Reconstructed image"
- Bullet: "Encoder: compresses the image into a small vector (classical)."
- Bullet: "Teacher: a classical diffusion model that has already learned to denoise step by step — used only to generate training examples."
- Bullet: "Student: a hybrid classical + quantum network that learns to jump straight to the clean answer, trained by imitating the teacher."

**Slide 15 — How the student is trained ("distillation")**
- Headline: "The student learns by imitating the teacher's trajectory."
- Bullet: "Take a noisy latent, ask the teacher to take one clean-up step."
- Bullet: "Train the student so its prediction agrees at two different noise levels along that same path."
- Bullet: "This is called 'consistency distillation' — a real teacher-student setup, not just training from raw data."

**Slide 16 — Three ways to place the quantum circuit**
- Headline: "Where does the quantum circuit go? Three designs tested."
- Visual: three small side-by-side diagrams:
  1. "QLCM-H (Head)": quantum circuit IS the main prediction engine.
  2. "QLCM-R (Residual)": classical network makes the main prediction; quantum circuit adds a small correction on top.
  3. "QLCM-T (Time)": quantum circuit produces a signal that tells the classical network "how noisy is this," modulating it.
- Short caption: "Same overall pipeline, three different jobs for the quantum piece."

**Slide 17 — Expectation going in**
- Headline: "Our prediction: the Residual design (QLCM-R) should be safest."
- Bullet: "Classical network handles the bulk of the work"
- Bullet: "Quantum circuit only needs to learn a small correction"
- Bullet: "Head design (QLCM-H) is the boldest bet — the quantum circuit alone must carry the whole task"

### Section 8: Experimental setup (1-2 slides)

**Slide 18 — Datasets and setup**
- Headline: "How we tested it."
- Bullet: "Datasets: MNIST (handwritten digits) and Fashion-MNIST (clothing images) — small, well-understood benchmark image sets."
- Bullet: "Classical baseline: a standard (non-quantum) consistency student with a matched number of parameters, for a fair comparison."
- Bullet: "Every configuration run 3 times with different random seeds, to make sure results are consistent and not a fluke."

**Slide 19 — How we measured quality**
- Headline: "Three ways we scored the generated images."
- Bullet: "FID-like score (lower is better): measures how similar generated images are to real ones, using a small custom classifier — NOT the standard 'Inception FID' used in most papers, so these numbers should only be compared within this study, not against other papers."
- Bullet: "Classifier confidence: how confidently a digit/clothing classifier recognizes the generated images."
- Bullet: "Label entropy (diversity): are we generating a good variety of different digits/clothing types, or collapsing to just one or two? Max possible score is 2.303 (perfectly even spread across 10 classes)."

### Section 9: Main results (5-7 slides)

**Slide 20 — Headline result: the quantum residual model doesn't win**
- Big numbers, side-by-side comparison table/graphic:
  - Classical model: 90,640 parameters, FID-like score 346.7 ± 16.0
  - QLCM-R (quantum residual): 139,640 parameters, FID-like score 359.4 ± 3.0
- Headline: "54% more parameters. Slightly worse score."
- Bullet: "Recognizability and diversity: statistically indistinguishable between the two."
- Framing line: "The quantum circuit added cost without adding benefit, at this scale."

**Slide 21 — The quantum-head design fails harder**
- Headline: "Making the quantum circuit the main engine (QLCM-H) backfires."
- Big number: "~10x higher training loss" than the classical or residual designs.
- Bullet: "Diversity collapse: label entropy dropped to 1.05-1.67, vs. ~2.03-2.05 for classical/residual (out of a max of 2.303)."
- Bullet: "In plain terms: the model started generating mostly the same one or two digits, over and over, instead of a healthy variety."
- Bullet: "This confirms a failure mode the team expected going in — good science means checking your predictions, including the ones that say 'this probably won't work.'"

**Slide 22 — Residual and time-conditioning: stable, but no edge**
- Headline: "The 'safer' quantum designs are stable — but don't help."
- Bullet: "Both the Residual (QLCM-R) and Time-conditioning (QLCM-T) placements train smoothly, matching the classical model's training loss."
- Bullet: "Neither shows the diversity collapse seen in the Head design."
- Bullet: "But neither shows any measurable advantage over the plain classical model either."

**Slide 23 — The story of two bugs: how research self-corrects**
- Headline: "Along the way, we found and fixed two real bugs."
- Bullet 1: "Bug #1: the training script wasn't actually consulting the teacher model — it was training from raw noisy data alone, not real 'distillation' as intended."
- Bullet 2: "Bug #2: the fast (multi-step) sampling process was adding noise back incorrectly between steps."
- Framing line: "Early results looked like fast sampling was fundamentally broken. It wasn't — the bugs were."

**Slide 24 — Fixing only one bug wasn't enough**
- Headline: "Lesson: fixing the obvious bug wasn't enough."
- Bullet: "Fixing just the sampling bug (#2) alone still left results degrading."
- Bullet: "Only after fixing the deeper training bug (#1) as well did fast sampling become stable."
- Framing line: "A visible symptom can have an invisible root cause — worth checking both before declaring victory."

**Slide 25 — Fast sampling works, once both bugs are fixed**
- Headline: "After both fixes: fast sampling is stable."
- Bullet: "MNIST, classical model: FID-like scores of 337.8 / 350.3 / 342.5 for 1-step / 2-step / 4-step sampling — essentially flat."
- Bullet: "MNIST, QLCM-R: 355.6 / 363.5 / 355.9 for 1 / 2 / 4 steps — also flat."
- Bullet: "In other words: going from 1 step to 4 steps doesn't meaningfully change quality — the 'few-step' promise holds up once the pipeline is correct."

**Slide 26 — An open problem: Fashion-MNIST mode collapse**
- Headline: "Honest open problem: Fashion-MNIST collapses toward one class."
- Bullet: "Both the classical and QLCM-R models, on Fashion-MNIST, generated a large share of samples as a single garment class ('T-shirt/top') — 337 of 512 and 301 of 512 samples respectively, in one test run."
- Bullet: "Label entropy dropped to about 1.3-1.4 out of a max of 2.303 for both models."
- Bullet: "This happened for both the classical and quantum models — so it's not a quantum-specific problem. Most likely cause: small training dataset and limited training budget."
- Framing line: "We're reporting this honestly rather than hiding or explaining it away."

### Section 10: Discussion (1-2 slides)

**Slide 27 — What this negative result means**
- Headline: "An honest 'no' — and why that matters."
- Bullet: "At this scale, a quantum circuit does not make the model better, smaller, or more efficient."
- Bullet: "Carefully reported negative results matter in a field where hype often outpaces evidence."
- Bullet: "Rigorous negative findings help the whole field avoid chasing effects that aren't real."

**Slide 28 — What this does NOT rule out**
- Headline: "What we did NOT test — and can't rule out."
- Bullet: "Larger quantum circuits (more qubits, more depth)"
- Bullet: "Other ways of placing the quantum circuit in the pipeline"
- Bullet: "Other, larger, or more complex datasets"
- Framing line: "This is a result about one set of designs at one small scale — not a verdict on quantum machine learning in general."

### Section 11: Limitations (1 slide)

**Slide 29 — Limitations**
- Headline: "What to keep in mind about this study."
- Bullet: "Simulator only — no real quantum hardware was used."
- Bullet: "Small quantum circuits: only 4-6 qubits tested."
- Bullet: "Small, simple datasets: MNIST and Fashion-MNIST, not natural photographs."

### Section 12: Future work (1 slide)

**Slide 30 — What's next**
- Headline: "Where this research goes next."
- Bullet: "Use a more standard training-stabilization technique (an 'EMA' target network) for smoother training."
- Bullet: "Diagnose and fix the Fashion-MNIST mode collapse."
- Bullet: "Test with a larger teacher model and larger latent space."
- Bullet: "Scale up to 8 qubits, as originally planned."

### Section 13: Closing (2 slides)

**Slide 31 — Closing takeaway**
- Headline: "The takeaway."
- Big statement: "At today's small scale, quantum circuits don't yet beat classical methods for fast image generation — but a well-run negative result is real scientific progress."
- Bullet: "Two bugs were found and fixed through careful, honest debugging."
- Bullet: "The path forward is clear: bigger circuits, fixed bugs, more datasets."

**Slide 32 — Thank you / Questions**
- Headline: "Thank you."
- Subheadline: "Questions?"
- Contact/author line: Felipe Mahlow

---

## Additional Notes for the Slide Generator

- Do not invent additional statistics beyond what is listed above — all numbers in this brief are exact figures from the source paper and should be reproduced exactly as given (including ± ranges).
- When a slide lists more bullets than would fit comfortably in a spoken-presentation format, prioritize the boldest/headline number or phrase and trim supporting bullets to fit — the priority order within each slide's bullets is top to bottom.
- Keep a consistent iconography style: simple line icons or flat illustrations, not photorealistic images, for concept slides (diffusion, qubits, circuits). Simple bar/line charts or big-number callouts for results slides.
- Avoid clip-art clichés for "quantum" (no glowing atom orbitals, no sci-fi laser effects). Prefer abstract geometric/circuit-style motifs consistent with the rest of the deck's visual theme.
- It is acceptable to merge or split slides slightly to hit the 25-32 slide target and a natural pacing for 40 minutes, as long as all the content points above are represented somewhere in the deck.
