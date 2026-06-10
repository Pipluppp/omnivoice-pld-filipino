[Training notebook](omnivoice_training.py) we experimented with on 2026-05-18 for a full training run and saved model checkpoint (1000-step)

[Best-eval training notebook, LR 1e-5](omnivoice_training_best_eval.py) is the canonical 5000-step best-development-loss checkpoint run. This matches the strongest current result pattern and should be used as the comparison template.

[Best-eval training notebook, LR 2e-5](omnivoice_training_best_eval_lr_2e_5.py) repeats the same 5000-step best-development-loss checkpointing setup with `LEARNING_RATE = 2e-5`.

[Best-eval training notebook, LR 5e-6](omnivoice_training_best_eval_lr_5e_6.py) repeats the same 5000-step best-development-loss checkpointing setup with `LEARNING_RATE = 5e-6`.

[Evaluation metrics notebook](omnivoice_evaluation_metrics.py) attempting to do a full test set computation of evaluation metrics **WER**, **SIM-o**, and **UTMOS** for both the base and initial finetuned. Resulted in running for 4 hours.

[Fine-tuned checkpoint evaluation notebook](omnivoice_evaluation_metrics_finetunes.py) evaluates later fine-tuned W&B artifacts while reusing the base model metrics from the first full evaluation run, as long as the test split, inference settings, evaluator models, and text normalization stay unchanged.

Current full test-set metric summary:

| Model | Artifact / checkpoint | WER (%) | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: |
| Base OmniVoice | `k2-fsa/OmniVoice` | 22.55 | 0.602 | 3.64 |
| Fine-tuned 1000 | `omnivoice-filipino-full-checkpoint-1000` | 20.07 | 0.610 | 3.60 |
| Fine-tuned 2000 | `omnivoice-filipino-full-checkpoint-2000` | 22.64 | 0.611 | 3.57 |
| Fine-tuned 4900 | `omnivoice-filipino-full-checkpoint-4900` | 18.52 | 0.604 | 3.61 |

Current strongest model: `omnivoice-filipino-full-checkpoint-4900`, selected because it gives the lowest WER while keeping SIM-o slightly above the base model and UTMOS close to the other fine-tuned checkpoints.

The table above mixes final-step checkpoints and a best-eval-loss checkpoint, so it is useful as a project history but not as a clean learning-rate comparison. For the paper discussion, rerun the controlled best-eval sweep at `1e-5`, `2e-5`, and `5e-6`, then compare the selected best checkpoints from those three runs.

