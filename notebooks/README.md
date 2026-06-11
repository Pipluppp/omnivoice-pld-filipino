[Best-eval training notebook, LR 1e-5](omnivoice_training_best_eval.py) is the canonical 5000-step best-development-loss checkpoint run and the template the other two controlled runs repeat. Its selected checkpoint (step 4900) is the strongest model in the final comparison.

[Best-eval training notebook, LR 2e-5](omnivoice_training_best_eval_lr_2e_5.py) repeats the same 5000-step best-development-loss checkpointing setup with `LEARNING_RATE = 2e-5`; its selected checkpoint is at step 5000.

[Best-eval training notebook, LR 5e-6](omnivoice_training_best_eval_lr_5e_6.py) repeats the same 5000-step best-development-loss checkpointing setup with `LEARNING_RATE = 5e-6`; its selected checkpoint is at step 5000.

[Evaluation metrics notebook](omnivoice_evaluation_metrics.py) is the first full test-set computation of **WER**, **SIM-o**, and **UTMOS**; it produced the base-model metric row reused by the later evaluation runs. Ran for about 4 hours.

[Fine-tuned checkpoint evaluation notebook](omnivoice_evaluation_metrics_finetunes.py) produced the metric row of the best-eval LR `1e-5` checkpoint (step 4900) while reusing the base model metrics from the first full evaluation run, valid as long as the test split, inference settings, evaluator models, and text normalization stay unchanged.

[New-LR evaluation notebook](omnivoice_evaluation_metrics_new_lr_reruns.py) produced the metric rows of the best-eval LR `2e-5` and LR `5e-6` checkpoints (both step 5000) on the same full test split, again reusing the base row. Its exported metrics, logs, manifest, figure, and audio-example table live in `../eval-2e-5-and-5e-6/`.

Final controlled full test-set metric summary (all fine-tunes: 5000-step run, best-development-loss checkpoint, differing only in learning rate):

| Model | Selection | WER (%) | WER delta vs base | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | pretrained base | 22.55 | 0.00 | 0.602 | 3.64 |
| Best-eval LR 1e-5 | best development-loss checkpoint at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 |
| Best-eval LR 2e-5 | best development-loss checkpoint at step 5000 | 18.83 | -3.72 | 0.583 | 3.61 |
| Best-eval LR 5e-6 | best development-loss checkpoint at step 5000 | 21.96 | -0.59 | 0.605 | 3.60 |

Strongest overall model: the best-eval LR `1e-5` checkpoint (`omnivoice-filipino-full-checkpoint-4900`). It gives the lowest WER while keeping SIM-o slightly above the base model; LR `2e-5` is close in WER but drops SIM-o below base, and LR `5e-6` keeps the best fine-tune SIM-o with only a modest WER gain. The base model keeps the highest UTMOS.
