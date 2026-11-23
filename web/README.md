---
title: MERLIN – Image to LaTeX
emoji: ∑
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Turn an image of a math formula into LaTeX code
---

# MERLIN web app

This folder is a self-contained FastAPI app. The block between the `---` lines above is the
configuration for [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker); keep it if you deploy there.
Full documentation is in the [README at the repository root](../README.md).

| File | Role |
|---|---|
| `app.py` | FastAPI backend: `GET /`, `GET /api/info`, `POST /api/predict` |
| `merlin_model.py` | model code – **identical** to `../merlin_model.py` and to the notebook |
| `static/` | front end (`index.html`, `app.js`, `style.css`) and example images |
| `model/` | `model.pt`, `tokenizer.json`, `model_info.json` |
| `test_app.py` | smoke test of the model code and the API (random weights) |

```bash
docker build -t merlin .
docker run --rm -p 7860:7860 merlin        # open http://localhost:7860
```
