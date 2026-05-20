This is a research project where we finetune OmniVoice voice cloning for the Filipino language using the cleaned Filipino subset of the Philippine Languages Database dataset (`data/pld_filipino_clean/`).

We run code, training, inference, all within Kaggle. And so we paste code scripts into Kaggle and work through it in notebook, and so when we have `.py` files in this project reflecting a certain notebook we are working on in Kaggle, we format them as `.py` but with the cells syntax to split across markdown and code.

Also note our constraints given we run GPU in Kaggle's free T4 GPU.

We have also cloned the actual OmniVoice repository in this directory to refer to it when needed some details. Despite OmniVoice having the whole `.sh` scripts to run training, finetuning, and evals. We adapt it into Kaggle `.py` code as we run everything through there. But still we aim to be guided by the `.md` documentation and guidelines from OmniVoice (eg., `docs\training.md`, `docs\evaluation.md`, `docs\data_preparation.md`)

For running finetuning/training we want to track Kaggle runs using WandB, and so some of our code will have integration of WandB syntax just to have proper tracking of the runs.

When editing Kaggle notebook-style `.py` files, keep markdown cells clean for Kaggle conversion. In `# %% [markdown]` cells, avoid standalone comment-only blank lines like `#` between paragraphs because Kaggle can preserve them as visible extra `#` characters in notebook markdown. Prefer actual blank markdown lines in the source style that converts cleanly, or keep the markdown block concise without filler comment separators.
