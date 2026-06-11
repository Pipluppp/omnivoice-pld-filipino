# 2026-05-19 full train track metrics

> **Superseded note.** The metric tables below mix final-step checkpoints
> (1000-step LR `2e-5`, 2000-step LR `5e-6`) with one best-development-loss
> checkpoint, so they are not a clean learning-rate comparison. They are kept
> as project history. The final controlled four-model comparison is in
> [2026-06-11-controlled-new-lr-eval.md](2026-06-11-controlled-new-lr-eval.md).

> Left out running [kaggle](https://www.kaggle.com/code/pipluppp/omnivoice-pld-full-train-tokenized-dataset-simple?scriptVersionId=320417583) ([wandb](https://wandb.ai/duncanb013-polytechnic-university-of-the-philippines/OmniVoice-PLD-Filipino/runs/omnivoice-fil-eaa6d9c2?nw=nwuserduncanb013)) of the simplified code that does a full train 1000-step with tokenized dataset

Setup the Evaluation metrics notebooks to load finetune models and base, run the metrics and the models needed to compute the metrics, run inference on test samples, check out how it performs, etc.

Finished running [evaluation metrics](../notebooks/omnivoice_evaluation_metrics.py) notebook on the base model and initial finetuned model to get their WER, SIM-o, and UTMOS. We now have a working notebook to get evals. 

So now we have to do training runs of models, then do evals on these finetunes, to get the best.

## Full test-set objective metric results

The first full evaluation used [omnivoice_evaluation_metrics.py](../notebooks/omnivoice_evaluation_metrics.py) for the base model and the initial 1000-step checkpoint. Later runs used [omnivoice_evaluation_metrics_finetunes.py](../notebooks/omnivoice_evaluation_metrics_finetunes.py) to evaluate the 2000-step checkpoint and the 4900-step development-loss-selected checkpoint while reusing the base model metrics from the first full run.

All rows use the same PLD Filipino test split and the same OmniVoice evaluation metric family:

- WER: lower is better.
- SIM-o: higher is better.
- UTMOS: higher is better.

| Model | Checkpoint / run | Learning rate | Steps / selection | WER (%) | WER delta vs base | SIM-o | UTMOS | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base OmniVoice | `k2-fsa/OmniVoice` | N/A | N/A | 22.55 | 0.00 | 0.602 | 3.64 | Baseline. |
| Fine-tuned 1000 | `omnivoice-filipino-full-checkpoint-1000` | `2e-5` | 1000 steps | 20.07 | -2.48 | 0.610 | 3.60 | Improves WER and SIM-o, with a small UTMOS drop. |
| Fine-tuned 2000 | `omnivoice-filipino-full-checkpoint-2000` | `5e-6` | 2000 steps | 22.64 | +0.09 | 0.611 | 3.57 | Highest SIM-o, but WER is slightly worse than base. |
| Fine-tuned 4900 | `omnivoice-filipino-full-checkpoint-4900` | `1e-5` | lowest eval loss at step 4900 | 18.52 | -4.03 | 0.604 | 3.61 | Strongest checkpoint by WER, with SIM-o preserved above base and only a small UTMOS drop. |

Current strongest model: `omnivoice-filipino-full-checkpoint-4900`.

Reason: it gives the largest intelligibility improvement, reducing WER from 22.55% to 18.52% on the full test set. That is a 4.03 absolute point reduction, about 17.9% relative WER reduction. It does not win on SIM-o or UTMOS individually, but SIM-o remains slightly above the base model and UTMOS is close to the other fine-tuned checkpoints. For this study, the selected checkpoint should be chosen by the coherent overall metric pattern, not by a single auxiliary metric.

Raw edit-count summary:

| Model | Insertions | Deletions | Substitutions | Words |
| --- | ---: | ---: | ---: | ---: |
| Base OmniVoice | 2,668 | 640 | 2,858 | 27,342 |
| Fine-tuned 1000 | 2,439 | 598 | 2,452 | 27,354 |
| Fine-tuned 2000 | 3,115 | 586 | 2,492 | 27,354 |
| Fine-tuned 4900 | 2,045 | 603 | 2,417 | 27,354 |

The 4900-step checkpoint mainly improves WER by reducing insertions and substitutions compared with the base model. The 2000-step `5e-6` run reduced deletions and substitutions but increased insertions enough to make its total WER slightly worse than base.

---

So now that we have a way for evals, back to training.

Planned on leaving some training runs on learning rate `1e-5` lower than the initial run for longer steps maybe 2000, 3000, or 5000 (compared to the initial 1000-step run). But not sure if it would help further improve.

And so had to test our a new model checkpoint saving to check every few eval steps, then save the model checkpoint only when the eval loss improves. So that for these long training runs we'll have access to a development-loss-selected finetuned model?

Left the following running:

- [lr=1e-5, steps=5000](https://www.kaggle.com/code/togepiiiiiiiiiiiii/omnivoice-pld-full-train-tokenized-dataset-simple?scriptVersionId=320677078) which uses the new model checkpoint saving only on the lowest eval loss
- [lr=5e-6, steps=2000](https://www.kaggle.com/code/pipluppp/omnivoice-pld-full-train-tokenized-dataset-simple?scriptVersionId=320677343) stills uses old save by certain model steps of 250, saving the last one
