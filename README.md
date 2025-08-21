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
