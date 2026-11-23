# Model files

The app loads three files from this folder (override the path with the `MODEL_DIR` environment variable):

| File | What it is |
|---|---|
| `model.pt` | trained weights of the **robust** model (≈52 MB) |
| `tokenizer.json` | BPE tokenizer, 1,223 tokens |
| `model_info.json` | model configuration and test metrics |

They are produced by cell 9 of `notebooks/MERLIN_training.ipynb` (saved to `MERLIN/web_model` on Google Drive).
If `HF_SPACE` and `HF_TOKEN` are set in the notebook, that cell uploads them to the Hugging Face Space directly.
