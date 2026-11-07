# AGENTS.md

A proof-of-concept HTTP API that serves Cloudflare's Clef-flash model locally. Other projects send screenshots and typed questions to `POST /v1/systemone`. The whole server is `app.py`. `README.md` covers setup, running the server and the request format: read it before running the server or changing what the API accepts.

## Conventions

- **POC scope.** Make the smallest change that works. Handle the failures a caller is likely to hit (bad input returns HTTP 400), and let everything else raise.
- **Generic server.** The API is `POST /v1/systemone` plus `GET /health`. Use-case prompts, like the error-detection request in `README.md`, belong to the calling project and arrive with each request.
- **Workers AI compatible.** Requests and responses match Cloudflare's hosted System One format, so a caller can switch between the hosted and local model by changing the URL.
- **Local weights.** The model loads from the local Hugging Face cache (`MODEL_PATH` in `app.py`). Keep it that way.
- **Google Python style:** https://google.github.io/styleguide/pyguide.html. pylint checks most of it. Check by hand that imports name modules (`import fastapi`, `from PIL import Image`), because pylint doesn't. Names from `typing` and `collections.abc` are the exception.
- **`requirements.txt` holds runtime dependencies.** Keep `torchvision` in it: nothing imports it directly, but the model's image processor needs it.

## Model code

Cloudflare's model code ships with the weights, not in this repo: `joint_schema_model.py` in the `MODEL_PATH` snapshot folder. In it, `systemone()` validates a request and runs the model, and `systemone_answer()` defines the answer shape for each question type. Read it before changing how requests are built or answers are read.

## Checking a change

A change is done when all three hold:

1. **Lint is silent.**

   ```bash
   curl -sSfL https://google.github.io/styleguide/pylintrc -o "${TMPDIR:-/tmp}/google-pylintrc"
   PYTHONPATH=.venv/lib/python3.15/site-packages uvx --python 3.15 pylint \
     --rcfile="${TMPDIR:-/tmp}/google-pylintrc" --jobs=1 app.py
   ```

   `--jobs=1` is required because the pylintrc's parallel mode crashes on Python 3.15. `uvx` keeps pylint out of `.venv`.

2. **The README example answers as before.** Start the server with `.venv/bin/uvicorn app:app --port 8788` and wait for `Application startup complete` (about 15 s). Run the example request from `README.md` with `shot.png` replaced by `test/01.png`. The response should match the one in `README.md` exactly, because the model is deterministic here.

   When you change `MAX_PIXELS` or the example prompt, also run the other screenshots in `test/`. Every one of them shows an error, so each should score `has_error.noul` above 0.5. Then paste the new real response for `test/01.png` into `README.md`.

3. **The server is stopped.** It holds about 20 GB of memory while it runs.
