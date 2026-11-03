# Experiment log

Every arm, what it edits, how it is measured, and what came out. Numbers are the ones committed under
`results/`. "GSM" is multilingual-gsm-symbolic, variant 1 of every template (100 problems per language, never
used to fit anything); "EuroEval" is the Danish + English suite of EuroEval 18.x.

| # | arm | edit | GSM dan | GSM eng | scala-en | dala | EuroEval (25 m.) |
|--:|---|---|---:|---:|---:|---:|---:|
| 0 | `base` | – | 0.900 | 0.960 | 64.90 | 97.67 | 52.36 |
| 1 | `q3` | 1.57M weights: 64 `down_proj` columns × 16 H layers, anti-English direction doubled | 0.880 | 0.940 | 65.58 | 96.08 | 51.76 |
| 2 | `acc_en_a05` | 1,536 weights: grammaticality direction into the `yes`/`no` head rows | – | – | **66.43** | 97.67 | not run |
| 3 | `combined_q3_acc` | both of the above, 1.58M weights (0.088%) | **0.940** | 0.960 | see §combined | see §combined | see §combined |

## Arm 1 — the language-drift edit

* **Diagnosis** (`src/diagnose.py`): v1.5 already answers Danish problems directly without reasoning on only 2%
  of items (v1: 32%), so the blog's "loop" target has no equivalent here. The largest remaining structure is that
  24–26% of Danish answers leave Danish mid-thought.
* **Localisation** (`src/langdir.py`): direction `u` = mean final state at English-leaning tokens − same at
  Danish-leaning tokens; **AUROC 0.971 held out**. MLP `down_proj` columns ranked by signed contribution; layer 15
  has one column writing 427× the mean.
* **Edit**: the 64 most anti-English columns in all 16 H layers, α = −1 (doubling their anti-English component).
  `specs/q3.json`.
* **Mechanism result**: drift 25% → 14% on held-out Danish, and 23% → 12% on the 40-problem screen; a random
  direction at the same dose leaves it at 23%.
* **Task result**: no gain — Danish 0.900 → 0.880 (2W/4L, p = 0.69), English 0.960 → 0.940; EuroEval −0.60 over
  25 datasets (6W/5T/14L).
* **Why**: the drift is a *symptom*. On held-out data drifting answers are no worse (0.920 vs 0.893); forcing
  Danish by prefilling `Trin 1:` recovers 2 of 24; 11 of 14 Danish errors are right in English on the same
  template.

## Arm 2 — the acceptability edit

* **Diagnosis** (`src/accept.py`, `src/accept_probe.py`): with EuroEval's prompt *including its
  `prompt_prefix`* and the PrefixLM mask, Mimir misses 164 of 512 corrupted sentences on scala-en's train split
  but raises only 36 false alarms. At the position that emits the label, the gold judgement is linearly
  decodable (**AUROC 0.91–0.92 held out**) — the information is there, the decision does not use it.
* **Edit**: `w_no += α·u`, `w_yes −= α·u` in the output head, 1,536 weights. α swept on the 2,048-item test
  split (disjoint from the reported subset): 61.90 (α=0) → **64.03** (0.5) → 63.55 (1) → 63.70 (1.5) → 63.30 (2).
* **Result**: scala-en 64.90 → **66.43** MCC (+1.53), reproduced exactly by the saved checkpoint; dala 97.67
  unchanged (the edit does not touch `ja`/`nej`).
* **dala is not improvable this way**: at 128/128 accepted and 7 misses, any shift along the Danish direction
  lowers MCC (0.947 → 0.918 at α = 0.5 → 0.903 at α = 2).

## Arm 3 — both at once

`specs/combined_q3_acc.json` = arm 1's 16 tensor edits + arm 2's 2 row edits, 1,575,936 weights (0.0882%).
Checkpoint `checkpoints/combined_q3_acc`, saved by the current `src/save_edit.py`.

* GSM-Symbolic held out: **Danish 0.900 → 0.940** (5W/1L, p = 0.22), English 0.960 → 0.960 (3W/3L, p = 1.0),
  Danish drift 25% → 16%. The Danish gain is suggestive, not established at n = 100 — and it is larger than
  either arm alone, which is worth a second look rather than a claim.
* EuroEval: see the table in the README (§combined) and `results/euroeval_combined/`.

## Measurement notes that cost real time

1. **Prompt prefix.** Dropping EuroEval's `prompt_prefix` line makes Mimir answer `ja` to every Danish sentence
   (MCC 0.00 dala). Any in-house reimplementation must include it.
2. **PrefixLM mask.** Without `token_type_ids` the same harness scores dala MCC 0.18 instead of 0.95. vLLM's
   `hrm_text` path reproduces the benchmark numbers, so the mask matters less there than in plain HF eager
   generation — but our own numbers only matched after applying it.
3. **Checkpoint round trip.** `save_pretrained` rewrote `num_hidden_layers` 16 → 128, which alone moved
   scala-en 64.90 → 60.46 and dala 97.67 → 94.52, and produced a table of spurious "gains" for one arm. Fixed by
   copying the published config/tokenizer files back after saving (§8 of the README), verified with an unedited
   re-save.
4. **Sign of a direction edit.** Ranking columns by |contribution| rather than signed contribution pushes the
   model the *wrong* way (measured: answer-onset English share 0.507 → 0.621/0.668 against 0.404 for the correct
   sign).
5. **KV cache.** HRM allocates `num_layers_per_stack × H_cycles × (L_cycles+1)` = 128 cache slots; batch 24 at 4k
   context is ~30 GB. Keep batches at 6–8 when anything else shares the GPU.
