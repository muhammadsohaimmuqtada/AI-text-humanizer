# AI Text Humanizer

A local, offline NLP pipeline that rewrites AI-generated text to read as
naturally human-written — reducing AI-detection scores by attacking
**burstiness**, **perplexity**, **sentence structure**, and **vocabulary
diversity** across 8 transformation stages.

**No API keys needed. No cloud. 100% local.**

## Features

- **8-stage humanization pipeline** — markers, contractions, clause
  reordering, sentence splitting, burstiness, discourse fillers, and
  synonym substitution working together.
- **~90 AI marker patterns** — strips both classic ("Furthermore,",
  "In conclusion,") and modern GPT-era markers ("delves into", "plays a
  crucial role in", "in today's rapidly evolving").
- **Contraction insertion** — converts formal AI phrasing ("do not" →
  "don't", "it is" → "it's") with 50+ patterns.
- **Clause reordering** — moves prepositional phrases to break AI's
  rigid Subject-Verb-Object pattern.
- **Sentence splitting** — breaks long compound sentences into short,
  punchy fragments that characterise human writing.
- **Discourse fillers** — inserts natural human interjections like
  "Honestly,", "Look,", "The thing is," sparingly.
- **Smart synonym substitution** — replaces adjectives/adverbs using
  WordNet with a robust blacklist (~100 words) to prevent bizarre swaps.
- **Burstiness variation** — merges sentences with 16 conjunction
  styles including semicolons, dashes, and conversational connectors.
- **Graphical User Interface (GUI)** — dark-themed desktop app built with
  Python's built-in `tkinter` library. No extra GUI framework required.
  - **Bypass Strength selector** — choose how aggressively the tool
    humanizes with three one-click presets:

    | Strength | Passes | Best for |
    |---|---|---|
    | **Light** | 1 | Lightly AI-flavoured text |
    | **Medium** | 2 | Moderate AI writing patterns |
    | **Aggressive** | 3 | Heavy AI output; drops detection to ~0% |
    | **Custom** | 1 | Fine-grained control via sliders |

  - **Multi-pass engine** — the Aggressive preset silently runs the
    pipeline three times for near-zero AI detection.
  - **Copy Result button** — copies the humanized text to clipboard.
- **REST API** — optional FastAPI server for programmatic access.
- **CLI** — full-featured command-line interface.

## How It Works

The pipeline applies **eight stages** in sequence:

1. **Marker stripping** — Removes ~90 common AI transition phrases
   (e.g. "Furthermore,", "In conclusion,", "delves into", "plays a
   crucial role in").
2. **Sentence tokenisation** — Splits the cleaned text into individual
   sentences using NLTK (regex fallback when NLTK data is absent).
3. **Contraction insertion** — Converts formal phrases to contractions
   ("do not" → "don't", "it is" → "it's", "cannot" → "can't").
4. **Clause reordering** — Moves leading prepositional/adverbial phrases
   to the end of sentences to break AI's monotonous structure.
5. **Sentence splitting** — Breaks long compound sentences at
   coordinating conjunctions into shorter ones.
6. **Burstiness variation** — Randomly merges adjacent sentences with
   varied conjunctions so sentence lengths mimic human writing.
7. **Discourse filler insertion** — Prepends natural human phrases like
   "Honestly,", "Look,", "The thing is," to some sentences.
8. **Synonym substitution** — Replaces a configurable fraction of
   adjectives and adverbs with WordNet synonyms to raise lexical
   perplexity without altering meaning.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/AI-text-humanizer.git
cd AI-text-humanizer
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
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
   - **Light** – one pass, conservative rates; good for lightly
     AI-flavoured text.
   - **Medium** – two passes; suitable for most AI-generated content.
   - **Aggressive** *(default)* – three passes at high rates; use this
     to drop AI detection to ~0% in a single click.
   - **Custom** – drag the *Synonym Rate* and *Merge Rate* sliders to
     any value you prefer.
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

## Python API

```python
from aip.humanizer import humanize

result = humanize(
    text="Furthermore, it is important to note that AI is transforming industries.",
    synonym_rate=0.35,
    merge_rate=0.25,
    contraction_rate=0.65,
    clause_reorder_rate=0.20,
    split_rate=0.30,
    filler_rate=0.08,
    seed=42,
)

print(result.humanized_text)
print(f"Words: {result.original_word_count} → {result.humanized_word_count}")
print(f"Markers removed: {result.markers_removed}")
print(f"Sentences merged: {result.sentences_merged}")
```

All new parameters are optional with sensible defaults — the simplest call
is just `humanize(text="...")`.

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

## Project Standards

- License: MIT ([LICENSE](LICENSE))
- Security guidance: [SECURITY.md](SECURITY.md)
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Running Tests

```bash
pip install -e '.[test]'
python -m pytest tests/ -v
```
