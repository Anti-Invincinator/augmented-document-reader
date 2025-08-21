# CMI Finance Document Reader — PoC

Single FastAPI service that extracts entities from **PDF/DOCX/Chat**:

- **PDF** → LLM (Gemini by default, optional GPT switch)
- **DOCX** → rule-based parser (`python-docx`)
- **Chat/TXT** → general NER model (HF) + light post-rules

Includes a tiny **HTML UI** (`index.html`) to upload a file and choose the doc type.

---

## 1) Quick Start

### Prereqs
- Python 3.10+ (3.11 recommended)
- pip / venv
- API keys (as needed):  
  - `GEMINI_API_KEY` (for Gemini)
  - `OPENAI_API_KEY` (for GPT)

### Install

```bash
# (recommended) create a virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Add API keys to .env file
```txt
# Copy to .env and fill in your keys
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional model overrides
# PDF_PROVIDER=gemini        # or 'openai'
# PDF_MODEL_GEMINI=gemini-1.5-flash
# PDF_MODEL_OPENAI=gpt-4o-mini
```

### Run API
```bash
uvicorn main:app --reload --port 8000
# API: http://127.0.0.1:8000
# Health: http://127.0.0.1:8000/health
```

### Test UI

Open index.html in your browser (double-click the file).
Pick Document type, (for PDF) choose LLM provider, select a file, click Extract.

## 2) Project Structure
```graphql
.
├─ main.py               # Central FastAPI app: /api/extract + /health
├─ pdf_utils.py          # PDF → Gemini or GPT (JSON-only output)
├─ docx_parser.py        # Rule-based DOCX extractor (your module)
├─ chat_parser.py        # HF NER for chat/TXT (your module)
├─ index.html            # Minimal upload UI
├─ requirements.txt
├─ .env.example          # Copy to .env and fill API keys
└─ .gitignore
```

## 3) API
POST /api/extract

Form fields

file: document to analyze

doc_type: pdf | docx | chat

(PDF only) provider: gemini | openai (default: gemini)

(PDF only) model: optional override (e.g., gemini-1.5-flash, gpt-4o-mini)

Response (example)
```json
{
  "ok": true,
  "doc_type": "pdf",
  "provider": "gemini",
  "result": {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "entities": [
      {"label": "INVOICE_NUMBER", "text": "INV-2025-001"},
      {"label": "TOTAL_AMOUNT", "text": "100,000", "currency": "USD"}
    ],
    "raw": "{... raw JSON from LLM ...}",
    "meta": {
      "file_uri": "providers://gemini/file/...",
      "usage": {"prompt_tokens": 1234, "candidates_tokens": 456, "total_tokens": 1690}
    }
  }
}

```

## 5) Notes

Serialization safety: responses only contain JSON-friendly primitives (no SDK objects).

Scanned PDFs: if text extraction is empty, add OCR later (e.g., pytesseract + pdf2image).

Performance: first call to the HF NER model may be slower (model load). Consider pre-warming on startup if needed.

Security: .env is in .gitignore. Commit only .env.example.
