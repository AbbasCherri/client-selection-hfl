# Do not adopt values from proposed_hfl_weights

`proposed_hfl_weights` (120 trials) and `proposed_hfl_weights_leaderboard.csv`
come from the PRE-2026-08-04 tuner, whose objective maximised `macro_f1` — the
reported TEST metric. Those 22 hyperparameters were selected by looking at the
answer, which is the leak the 2026-08 rigor pass exists to remove.

Use the `weights_<method>` studies instead. They score `val_macro_f1` on
tuning seeds 20-22, disjoint from the evaluation seeds 0-19, and each method has
its own study so no baseline inherits a recipe fitted to the proposed method.

Kept rather than deleted so the two can be compared: the val-selected optimum
sits at a materially different point (lr_decay none vs cosine, sel_static_blend
0.12 vs 0.43), which is the evidence that the leak changed the answer.
