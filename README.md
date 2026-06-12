# MNNIT_BOT — Multimodal Corporate AI Engine

A multimodal AI assistant for MNNIT Allahabad, covering three knowledge domains: **Campus Life**, **Management Consulting**, and **Health & Fitness**. Combines text chat, voice input, OCR document scanning, and lip-reading-based input.

This repo includes **two implementations** of the LLM backend — pick whichever fits your setup.

## Versions

| | `groq_version/` | `ollama_version/` |
|---|---|---|
| Backend | Groq API — `llama-3.3-70b-versatile` | Local Ollama model (e.g. `llama3.2`) |
| Internet required | Yes | No |
| API key needed | Yes (`GROQ_API_KEY`) | No |
| Speed | Fast (cloud GPU) | Depends on local hardware |
| Cost | Free tier limits apply | Free, unlimited |
| Setup | Just an API key | Ollama installed + model pulled locally |

Everything else — domain detection, EasyOCR, gTTS, lip-reading, the knowledge base, and the UI — is identical between the two versions. Only the LLM call differs.

## Features

- **Domain-aware chat**: Auto-detects whether a query relates to MNNIT campus info, consulting frameworks, or fitness/nutrition, and routes to a specialized system prompt.
- **Voice input/output**: Speech recognition (Google STT) for input, gTTS for spoken responses (English/Hindi).
- **OCR scan**: Captures a document via webcam and uses EasyOCR (GPU-accelerated) to extract and summarize text.
- **Biometric lip input**: Experimental lip-motion detection via OpenCV Haar cascades, mapped to a control vocabulary.
- **Admin knowledge panel**: Password-protected interface to inject new facts into the knowledge base (`manual_knowledge.json`).
- **Conversation logging**: Saves daily conversation logs as JSON.

## Project Structure

```
MNNIT_BOT/
├── README.md
├── requirements_groq.txt
├── requirements_ollama.txt
├── groq_version/
│   └── app.py
├── ollama_version/
│   └── app.py
├── chatbot_data/            # Knowledge base (manual_knowledge.json)
├── chatbot_conversations/   # Daily conversation logs
└── chatbot_db/              # Vector DB (if RAG enabled)
```

## Setup

### Option A — Groq (cloud)

```bash
conda create -n torchenv python=3.10
conda activate torchenv
pip install -r requirements_groq.txt
```

Set your API key (don't hardcode it):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Run:

```bash
python groq_version/app.py
```

### Option B — Ollama (fully offline)

```bash
conda create -n torchenv python=3.10
conda activate torchenv
pip install -r requirements_ollama.txt
```

Make sure Ollama is installed and a model is pulled:

```bash
ollama pull llama3.2
ollama serve
```

Run:

```bash
python ollama_version/app.py
```

## Usage

Both versions launch a Gradio interface at `http://127.0.0.1:7860` (or run the ipywidgets UI inline if using a Jupyter notebook).

- **Text chat**: type your query, get a domain-routed response.
- **Voice input**: click the mic button, speak, response is read aloud.
- **OCR scan**: hold a document to the webcam to extract and summarize text.
- **Biometric lip**: experimental — works best with clear frontal face positioning and good lighting.

## Knowledge Domains

| Domain | Badge | Topics |
|---|---|---|
| `nit_allahabad` | MNNIT CAMPUS | Hostels, mess, departments, placements |
| `consultancy` | STRATEGY CONSULTING | MECE, Porter's Five Forces, case frameworks |
| `health_gym` | HEALTH & FITNESS | Protein intake, training, nutrition |

## Notes

- Admin panel requires a password (set via `ADMIN_PASSWORD` in config) to add new knowledge entries.
- GPU acceleration (CUDA) is auto-detected for EasyOCR; falls back to CPU if unavailable.
- Never commit API keys — use environment variables and add `.env` / `*_secret*` to `.gitignore`.

## License

MIT
