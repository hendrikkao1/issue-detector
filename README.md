# Clef System One API (POC)

A local HTTP API for Cloudflare's **Clef-flash**, using the 4-bit MLX build [`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit) on Apple Silicon. You send it screenshots and typed questions, and it returns a probability for each answer. Nothing leaves your machine.

`app.py` loads the weights and their loader, `clef_mlx.py`, from `~/.cache/huggingface/hub/models--mlx-community--clef-flash-4bit`, using the snapshot that `refs/main` points to. The server never reaches the network.

## First time only

Run these from the project folder:

```bash
python3 -m venv .venv               # skip if .venv already exists
source .venv/bin/activate
pip install -r requirements.txt
hf download mlx-community/clef-flash-4bit   # 6.2 GB, skip if already downloaded
```

## Start the server

```bash
source .venv/bin/activate           # once per terminal; your prompt then shows (.venv)
uvicorn app:app --port 8788
```

Wait for `Application startup complete`, which takes about 4 seconds. MLX loads the weights on first use, so the first request takes about 5 seconds and later ones about 2. Then:

- Press **Ctrl+C** to stop the server, which frees the ~10 GB of memory it uses.
- Run `deactivate` to leave the virtualenv.
- To skip activating, run `.venv/bin/uvicorn app:app --port 8788` instead.
- To call it from another machine, add `--host 0.0.0.0`. There is no auth.

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/systemone` | Answers your questions about up to 4 images |
| `GET /health` | Returns the model name and snapshot. Use it to check that the server is up |

Interactive docs: http://127.0.0.1:8788/docs

## Example request: does a screenshot show an error?

```bash
IMG=$(base64 < shot.png | tr -d '\n')
curl -s localhost:8788/v1/systemone -H 'content-type: application/json' -d @- <<JSON
{
  "state": "The attached image is a screenshot of a computer or phone screen.\nCount as an ERROR: a browser error page (e.g. \"This site can't be reached\", ERR_*, 404, 500, 502, 503, certificate warnings), an application crash or \"not responding\" dialog, an error or failure dialog, toast or banner, an unhandled exception or stack trace shown to the user, a \"Something went wrong\" / \"An error occurred\" screen, or a failed operation message (upload failed, payment failed, sync failed).\nDo NOT count: normal pages that merely mention the word \"error\" (documentation, articles, code editors, log viewers being read on purpose), neutral helper text under form fields (such as password rules shown before anything is typed), empty states such as \"No results found\", informational banners, or warnings that are not failures.\nCount failed sign-in or authentication messages, such as \"Incorrect username or password\", and form validation errors, such as \"Email is required\" under a field, as an ERROR.",
  "questions": {
    "has_error": {
      "type": "noul",
      "instructions": "Does this screenshot show a software error, as defined in the state?"
    },
    "error_kind": {
      "type": "choice",
      "instructions": "What kind of error does the screenshot show?",
      "criteria": {
        "none": "No error is shown",
        "browser_error_page": "A browser error page: site can't be reached, HTTP 4xx/5xx, certificate or DNS error",
        "app_crash": "An application crashed, froze, or is not responding",
        "error_dialog": "A modal dialog or popup reporting an error or failure",
        "error_banner_or_toast": "An inline banner, toast, or notification reporting an error or failure",
        "stack_trace": "A stack trace, exception, or debug error output shown to the user",
        "failed_operation": "A full screen or page saying an operation failed or something went wrong"
      }
    }
  },
  "images": [{"content_type": "image/png", "base64": "$IMG"}]
}
JSON
```

Response for `test/01.png`, a Chrome "This site can't be reached" page:

```json
{
  "success": true,
  "errors": [],
  "messages": [],
  "result": {
    "model": "clef-flash-4bit",
    "answers": {
      "has_error": {"type": "noul", "noul": 0.9511},
      "error_kind": {
        "type": "choice",
        "choice": "browser_error_page",
        "confidence": 0.9046,
        "probabilities": {
          "none": 0.0127,
          "browser_error_page": 0.9046,
          "app_crash": 0.0127,
          "error_dialog": 0.0295,
          "error_banner_or_toast": 0.0153,
          "stack_trace": 0.0135,
          "failed_operation": 0.0116
        }
      }
    },
    "usage": {"input_tokens": 1128, "output_tokens": 0}
  }
}
```

To use the result:

- `result.answers.has_error.noul` is the probability that the screenshot shows an error. Treat it as an error when it is at least 0.5. To pick a better cut-off, check it on ~100 screenshots you've labeled yourself.
- `result.answers.error_kind.choice` is the most likely kind, and `confidence` is that kind's probability.

## Request format

| Field | |
|---|---|
| `state` | Context for the model, as plain text (any JSON value works). Put your rules here, such as what counts as an error |
| `questions` | An object mapping your own question IDs to question objects. Answers come back under the same IDs |
| `images` | Up to 4 images, each `{"content_type": "image/png", "base64": "..."}` or a data URL `"data:image/png;base64,..."`. Images over ~0.5 MP are downscaled |

Question types:

| `type` | Extra fields | Answer |
|---|---|---|
| `noul` | none | `{"noul": 0.93}`: the probability of "yes" |
| `choice` | `criteria`: `{"option": "description", ...}` | `{"choice", "confidence", "probabilities"}` |
| `score` | `criteria`: `["level 0 description", "level 1 description", ...]` | `{"score", "confidence", "legend", "probabilities"}`. `score` is the expected level, from 0 to n-1 |

The model answers all questions in a request together, so related questions (like `has_error` and `error_kind`) stay consistent with each other.

Bad input returns HTTP 400 with `{"detail": "..."}`.

## Tuning

Settings are constants at the top of `app.py`:

- **Image size:** `MAX_PIXELS` defaults to 0.5 MP, which measured as the sweet spot on 5K screenshots. It gave the same verdicts on every screenshot in `test/` as larger settings did, including a small error dialog and a text-only banner. Higher settings didn't improve any answer and cost more time. On the earlier PyTorch backend it took 2.1 s at 0.5 MP, 3.6 s at 1 MP (the hosted service's size), about 6 s at 2 MP and 15 s at 4 MP. For very small toasts on big screens, you can crop the screenshot into up to 4 tiles and send them as separate images.
- **Prompt length:** about half of the model time is the prompt text. The example's ~615 text tokens take about 0.8 s on their own, so a shorter `state` is the cheapest speedup.

Requests are processed one at a time because the GPU is the bottleneck. Expect about 1.9 seconds per screenshot on an M3 Max, compared with 2.2 s on the earlier PyTorch backend.

When an answer hovers near 0.5 at every resolution, the definition in `state` is usually ambiguous for that case, so add a sentence that settles it. That's how the sign-in sentence in the example came about.
