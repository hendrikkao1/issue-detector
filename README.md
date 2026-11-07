# Clef System One API (POC)

A local HTTP API for Cloudflare's **Clef-flash**. You send it screenshots and typed questions, and it returns a probability for each answer. It runs on Apple Silicon (MPS), NVIDIA (CUDA) or CPU, and nothing leaves your machine.

`app.py` loads the weights from `~/.cache/huggingface/hub/models--Cloudflare--clef-flash`, using the snapshot that `refs/main` points to. Nothing is downloaded.

## First time only

Run these from the project folder:

```bash
python3 -m venv .venv               # skip if .venv already exists
source .venv/bin/activate
pip install -r requirements.txt
```

## Start the server

```bash
source .venv/bin/activate           # once per terminal; your prompt then shows (.venv)
uvicorn app:app --port 8788
```

Wait for `Application startup complete`, which takes about 15 seconds while the model loads. Then:

- Press **Ctrl+C** to stop the server, which frees the ~20 GB of memory it uses.
- Run `deactivate` to leave the virtualenv.
- To skip activating, run `.venv/bin/uvicorn app:app --port 8788` instead.
- To call it from another machine, add `--host 0.0.0.0`. There is no auth.

The `[ERROR] ... is part of ... signature, but not documented` lines at startup are docstring warnings from transformers and can be ignored.

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/systemone` | Answers your questions about up to 4 images |
| `GET /health` | Returns the model snapshot and device (`mps`, `cuda` or `cpu`). Use it to check that the server is up |

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
    "model": "fb4417d31c1d0f9e7e77f1141b40aaede7491e00",
    "answers": {
      "has_error": {"type": "noul", "noul": 0.9533},
      "error_kind": {
        "type": "choice",
        "choice": "browser_error_page",
        "confidence": 0.8998,
        "probabilities": {
          "none": 0.0141,
          "browser_error_page": 0.8998,
          "app_crash": 0.011,
          "error_dialog": 0.0318,
          "error_banner_or_toast": 0.0187,
          "stack_trace": 0.0137,
          "failed_operation": 0.011
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

- **Image size:** `MAX_PIXELS` defaults to 0.5 MP, which measured as the sweet spot on 5K screenshots. It takes about 2 s per screenshot, compared with about 3.4 s at 1 MP (the hosted service's size), and gave the same verdicts on every screenshot in `test/`. Those include a small error dialog and a text-only banner. Higher settings didn't improve any answer and cost more time: about 6 s at 2 MP and 15 s at 4 MP. For very small toasts on big screens, you can crop the screenshot into up to 4 tiles and send them as separate images.
- **MPS issues:** if you see bfloat16 errors, set `DTYPE = torch.float16`.

Requests are processed one at a time because the GPU is the bottleneck. Expect about 2 seconds per screenshot on a Mac.

When an answer hovers near 0.5 at every resolution, the definition in `state` is usually ambiguous for that case, so add a sentence that settles it. That's how the sign-in sentence in the example came about.
