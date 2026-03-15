# AI Text Humanizer Platform

A local, modular NLP pipeline that rewrites AI-generated text to read as
naturally human-written – reducing AI-scanner detection scores by raising
**burstiness** (sentence-length variance) and **perplexity** (vocabulary
diversity) while removing common AI transition markers.

## Features

- **AI marker stripping** – Removes common AI transition phrases such as
  *"In conclusion,"*, *"Furthermore,"*, and *"As an AI language model"*.
- **Burstiness variation** – Merges adjacent sentences with natural conjunctions
  to mimic the sentence-length variability of a human writer.
- **Synonym substitution** – Swaps adjectives and adverbs with WordNet synonyms
  to raise lexical perplexity without changing logical meaning.
- **Graphical User Interface (GUI)** – A dark-themed desktop app built with
  Python's built-in `tkinter` library. No extra GUI framework required.
  - **Bypass Strength selector** – choose how aggressively the tool humanizes
    your text with three one-click presets:

    | Strength | Passes | Synonym Rate | Merge Rate | Best for |
    |---|---|---|---|---|
    | **Light** | 1 | 35 % | 25 % | Lightly AI-flavoured text |
    | **Medium** | 2 | 60 % | 40 % | Moderate AI writing patterns |
    | **Aggressive** | 3 | 85 % | 60 % | Heavy AI output; drops detection to ~0 % in one click |
    | **Custom** | 1 | user-defined | user-defined | Fine-grained control via sliders |

  - **Multi-pass engine** – the Aggressive preset silently runs the pipeline
    three times so you get near-zero AI detection in a single button press
    instead of pasting and re-running manually.
  - **Copy Result button** – copies the humanized text straight to your
    clipboard.
- **REST API** – optional FastAPI server for programmatic access.
- **CLI** – full-featured command-line interface.

## Project Standards

- License: MIT ([LICENSE](LICENSE))
- Security guidance: [SECURITY.md](SECURITY.md)
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## How It Works

The humanization pipeline applies four stages in sequence:

1. **Marker stripping** – Removes common AI transition phrases such as
   *"In conclusion,"*, *"Furthermore,"*, and *"As an AI language model"*.
2. **Sentence tokenisation** – Splits the cleaned text into individual
   sentences using NLTK `sent_tokenize` (regex fallback when NLTK data is
   absent).
3. **Burstiness variation** – Randomly merges adjacent sentences with a
   natural conjunction (e.g. *" and "*, *" while "*, *" — "*) so that
   sentence lengths vary the way a human writer's do.
4. **Synonym substitution** – Replaces a configurable fraction of
   adjectives and adverbs with WordNet synonyms to raise lexical
   perplexity without altering the logical meaning.

## Install

```bash
cd /path/to/ai-text-humanizer
pip install -e '.[api]'
```

Download NLTK data once:

```bash
python -m nltk.downloader wordnet punkt_tab averaged_perceptron_tagger_eng
```

### GUI dependencies

The GUI uses Python's standard-library `tkinter` module, so **no extra
packages are needed** beyond what is already installed above.

`tkinter` ships with most Python distributions. If you are on a Debian/Ubuntu
system and receive `ModuleNotFoundError: No module named 'tkinter'`, install
it with:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# Fedora / RHEL
sudo dnf install python3-tkinter

# macOS (Homebrew Python)
brew install python-tk
```

On Windows, re-run the Python installer and make sure the **tcl/tk and IDLE**
optional component is checked.

## GUI Usage

Launch the graphical user interface from a terminal:

```bash
# If installed via pip (recommended)
aip-gui

# Or run as a Python module (works from the project root without installing)
python -m aip.gui
```

The app window opens immediately. Typical workflow:

1. **Paste** your AI-generated text into the *Input Text* area on the left.
2. **Select** a *Bypass Strength* preset in the centre panel:
   - **Light** – one pass, conservative rates; good for lightly AI-flavoured text.
   - **Medium** – two passes; suitable for most AI-generated content.
   - **Aggressive** *(default)* – three passes at high rates; use this to drop
     AI detection to ~0 % in a single click.
   - **Custom** – drag the *Synonym Rate* and *Merge Rate* sliders to any value
     you prefer.
3. Click **▶ Humanize** and wait a moment for the pipeline to finish.
4. Read the result in the *Humanized Output* area on the right.
5. Click **⎘ Copy Result** to copy the text directly to your clipboard.

## CLI Usage

### Humanize text

```bash
aip humanize --text "Furthermore, it is important to note that AI is rapidly evolving. Many businesses use it to automate daily tasks. This saves them a significant amount of time and money. In conclusion, the future of AI looks very promising." --pretty
```

Output example:

```json
{
  "humanized_text": "AI is rapidly evolving. Many businesses use it to automate daily tasks, meaning this saves them a considerable amount of time and money. The future of AI looks very auspicious.",
  "original_word_count": 42,
  "humanized_word_count": 38,
  "markers_removed": 3,
  "sentences_merged": 1
}
```

### Check NLP dependencies

```bash
aip doctor --pretty
```

### Preflight readiness check

```bash
aip preflight --pretty
```

### Run the API server

```bash
aip serve --port 8000
```

## API Usage

### `POST /humanize`

```bash
curl -X POST http://localhost:8000/humanize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Furthermore, AI is rapidly evolving. It is important to note that many businesses already use it.",
    "synonym_rate": 0.35,
    "merge_rate": 0.25,
    "seed": 42
  }'
```

Response:

```json
{
  "request_id": "...",
  "processed_at": "...",
  "humanized_text": "AI is rapidly evolving, meaning many businesses already use it.",
  "original_word_count": 18,
  "humanized_word_count": 12,
  "markers_removed": 2,
  "sentences_merged": 1
}
```

### Other endpoints

| Endpoint     | Method | Description                              |
|--------------|--------|------------------------------------------|
| `/healthz`   | GET    | Liveness check                           |
| `/readyz`    | GET    | Readiness check with NLP capability info |
| `/doctor`    | GET    | Runtime capability details               |
| `/policies`  | GET    | Active configuration limits              |
| `/metrics`   | GET    | Request and latency metrics              |

## Environment Configuration

Copy and edit `.env.example`:

```bash
cp .env.example .env
```

Key variables:

| Variable              | Default          | Description                          |
|-----------------------|------------------|--------------------------------------|
| `AIP_API_KEY`         | *(empty)*        | Optional API key for the REST API    |
| `AIP_MAX_TEXT_CHARS`  | `25000`          | Maximum text length per request      |
| `AIP_RATE_LIMIT_PER_MIN` | `60`          | Requests allowed per minute          |
| `AIP_RATE_LIMIT_BURST`   | `20`          | Burst allowance                      |
| `AIP_AUDIT_LOG_PATH`  | `/tmp/aip_audit.jsonl` | Audit log destination          |

## Python API

```python
from aip.humanizer import humanize

result = humanize(
    text="Furthermore, it is important to note that AI is transforming industries.",
    synonym_rate=0.35,
    merge_rate=0.25,
    seed=42,
)

print(result.humanized_text)
print(f"Words: {result.original_word_count} → {result.humanized_word_count}")
print(f"Markers removed: {result.markers_removed}")
print(f"Sentences merged: {result.sentences_merged}")
```
