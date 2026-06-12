# MNNIT_BOT — Multimodal Corporate AI Engine

A multimodal AI assistant built for MNNIT Allahabad, covering three knowledge domains: **Campus Life**, **Management Consulting**, and **Health & Fitness**. Combines text chat, voice input, OCR document scanning, and lip-reading-based input, powered by the Groq API (LLaMA 3.3 70B).

## Features

- **Domain-aware chat**: Auto-detects whether a query relates to MNNIT campus info, consulting frameworks, or fitness/nutrition, and routes to a specialized system prompt.
- **Voice input/output**: Speech recognition (Google STT) for input, gTTS for spoken responses (English/Hindi).
- **OCR scan**: Captures a document via webcam and uses EasyOCR (GPU-accelerated) to extract and summarize text.
- **Biometric lip input**: Experimental lip-motion detection via OpenCV Haar cascades, mapped to a control vocabulary.
- **Admin knowledge panel**: Password-protected interface to inject new facts into the knowledge base (`manual_knowledge.json`).
- **Conversation logging**: Saves daily conversation logs as JSON.

## Tech Stack

- **LLM**: Groq API — `llama-3.3-70b-versatile`
- **OCR**: EasyOCR (English + Hindi, GPU via CUDA/PyTorch)
- **Vision**: OpenCV (Haar cascade face/mouth detection)
- **Speech**: `speech_recognition`, `sounddevice`, `gTTS`, `pygame`
- **UI**: ipywidgets (Jupyter) / Gradio (web deployment)

## Setup

```bash
# Create environment
conda create -n torchenv python=3.10
conda activate torchenv

# Install dependencies
pip install groq easyocr opencv-python pillow gtts pygame \
    sounddevice scipy speechrecognition ipywidgets torch \
    requests beautifulsoup4 gradio
```

### Configuration

Set your Groq API key:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Or set it directly in the script (not recommended for production).

## Usage

### Jupyter Notebook (local UI)

```bash
jupyter lab
```

Open the notebook and run all cells. The ipywidgets-based console UI will render inline.

### Gradio (web/local URL)

```bash
python app.py
```

Launches a local web interface at `http://127.0.0.1:7860`.

## Project Structure

```
MNNIT_BOT/
├── chatbot_data/
│   └── manual_knowledge.json   # Domain knowledge base
├── chatbot_conversations/      # Daily conversation logs
├── chatbot_db/                 # Vector DB (if RAG enabled)
├── app.py / notebook.ipynb     # Main application
└── README.md
```

## Knowledge Domains

| Domain | Badge | Topics |
|---|---|---|
| `nit_allahabad` | MNNIT CAMPUS | Hostels, mess, departments, placements |
| `consultancy` | STRATEGY CONSULTING | MECE, Porter's Five Forces, case frameworks |
| `health_gym` | HEALTH & FITNESS | Protein intake, training, nutrition |

## Notes

- Admin panel requires a password (set via `ADMIN_PASSWORD` in config) to add new knowledge entries.
- GPU acceleration (CUDA) is auto-detected for EasyOCR; falls back to CPU if unavailable.
- Lip-reading is experimental and works best with clear, frontal face positioning under good lighting.

## License

Specify your license here (e.g., MIT).
