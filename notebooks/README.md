[Training notebook](omnivoice_training.py) we experimented with on 2026-05-18 for a full training run and saved model checkpoint (1000-step)

[Evaluation metrics notebook](omnivoice_evaluation_metrics.py) attempting to do a full test set computation of evaluation metrics **WER**, **SIM-o**, and **UTMOS** for both the base and initial finetuned. Resulted in running for 4 hours.

[Fine-tuned checkpoint evaluation notebook](omnivoice_evaluation_metrics_finetunes.py) evaluates later fine-tuned W&B artifacts while reusing the base model metrics from the first full evaluation run, as long as the test split, inference settings, evaluator models, and text normalization stay unchanged.

Current full test-set metric summary:

| Model | Artifact / checkpoint | WER (%) | SIM-o | UTMOS |
| --- | --- | ---: | ---: | ---: |
| Base OmniVoice | `k2-fsa/OmniVoice` | 22.55 | 0.602 | 3.64 |
| Fine-tuned 1000 | `omnivoice-filipino-full-checkpoint-1000` | 20.07 | 0.610 | 3.60 |
| Fine-tuned 2000 | `omnivoice-filipino-full-checkpoint-2000` | 22.64 | 0.611 | 3.57 |
| Fine-tuned best | `omnivoice-filipino-full-checkpoint-best` | 18.52 | 0.604 | 3.61 |

Best current model: `omnivoice-filipino-full-checkpoint-best`, selected because it gives the lowest WER while keeping SIM-o slightly above the base model and UTMOS close to the other fine-tuned checkpoints.

