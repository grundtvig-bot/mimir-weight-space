# Decisions and dead ends

Notes for whoever repeats this, mostly so the wrong turns don't get repeated.

## Why the language direction and not "the loop"

The blog's Qwen result is a loop: 93% of answers hit the token limit, a single direction written by the MLP
output columns of layers 15 and 19 switches it off. Before looking for an equivalent in Mimir v1.5 we measured
the failure taxonomy (`src/diagnose.py`): direct answers without reasoning are 2% of Danish answers (v1: 32%),
truncation is 1–2%, repetition loops are ~1%. There is no fat target of that kind.

What *is* large is the share of Danish answers that leave Danish: 24–26% of them, in a model that answers
English problems 92–96% correctly. That looked like the closest analogue of "an obstacle hiding an ability",
and it comes with a decodable direction (AUROC 0.97 held out).

## The sign of the localisation matters, and |contribution| gets it wrong

The first localisation ranked columns by |mean contribution at English positions − mean at Danish positions|.
Ablating those columns (arms e1–e3) changed 8 of 40 answers and no language decision at all — and the fast probe
showed why: the English share of the answer-onset mass went *up* (0.507 → 0.621/0.668). The columns with the
largest |contribution| were the ones pushing *against* English; removing them made the model more English.

Ranking by the signed contribution and then choosing the negative tail (columns that oppose English) with α = −1
— i.e. doubling their anti-English component — moves the probe the right way (0.404) and halves the drift in
generation. Lesson: for a direction edit, decide first which tail you want, then check the sign with a forward
pass before spending a generation run.

## α is not monotone

α = −2 halves less drift (0.20) than α = −1 (0.12), and 256 columns × 16 layers (q4, 6.29M weights) also halves
it but costs accuracy. The chosen arm is small on purpose: 64 columns × 16 layers = 1.57M weights, 0.088%.

## The fast probe is worth building first

`src/probe_decision.py` is one forward pass per problem (100 problems in about a minute) and reports the
probability mass on English- vs Danish-leaning tokens at the first answer position. Every arm above was screened
with it before a generation run, which is what made a 9-arm dose–response affordable on a box that is also
running someone else's training. It is not a substitute for generation — it measures the *inclination*, not the
outcome — but it kills arms fast and it caught the sign error immediately.

## HRM's KV cache is bigger than the model

`num_hidden_layers` for the cache layout is `num_layers_per_stack × H_cycles × (L_cycles + 1)` = 128 slots for
Mimir, not 16. Batch 24 at 4k context allocates ~30 GB and OOM'd a neighbouring training job on the same GPU.
Batch 6–8 is the safe regime on a 48 GB card when something else is on the box; check `nvidia-smi` first.

## Dev/test split

Variant 0 of each template fits the direction and chooses the dose; variant 1 measures everything reported as a
result. The drift→accuracy correlation that motivated the edit (0.708 vs 0.908 on variant 0) shrank to 0.920 vs
0.893 on variant 1 — a reminder that 100 problems is a small sample and that this particular split was kind to
the hypothesis the first time.
