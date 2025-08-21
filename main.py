import os
import io
import tempfile
from typing import Literal, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# Local modules
from pdf_utils import extract_pdf_entities

# Your uploaded modules (interfaces assumed as previously discussed)
from docx_parser import FinancialDocumentParser
from chat_parser import FinancialNERExtractor


app = FastAPI(title="CMI Finance NER — Unified API", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class ExtractResult(BaseModel):
    ok: bool
    doc_type: Literal["pdf", "docx", "chat"]
    provider: str | None = None     # for pdf (gemini/openai)
    result: dict


def _py(v: Any) -> Any:
    """
    Coerce nested structures to JSON-safe (numpy, bytes, sets, enums, etc.).
    """
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, dict):
        return {str(k): _py(w) for k, w in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_py(x) for x in v]
    # rely on jsonable_encoder for dataclasses/enums
    return v


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/extract", response_model=ExtractResult)
async def extract(
    file: UploadFile = File(...),
    doc_type: Literal["pdf", "docx", "chat"] = Form(...),
    # PDFs only:
    provider: Literal["gemini", "openai"] | None = Form(None),
    model: str | None = Form(None),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")

    if doc_type == "pdf":
        prov = provider or "gemini"
        if prov == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise HTTPException(500, "OPENAI_API_KEY not set.")
        else:
            if "GEMINI_API_KEY" not in os.environ:
                raise HTTPException(500, "GEMINI_API_KEY not set.")

        try:
            out = extract_pdf_entities(data, provider=prov, model=model)
        except Exception as e:
            raise HTTPException(500, f"PDF LLM extraction failed: {e}")

        out = jsonable_encoder(_py(out))
        return ExtractResult(ok=True, doc_type="pdf", provider=prov, result=out)

    elif doc_type == "docx":
        # Persist to temp path for your parser
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            parser = FinancialDocumentParser()
            parsed = parser.parse_document(tmp_path)
        except Exception as e:
            raise HTTPException(500, f"DOCX parse failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        parsed = jsonable_encoder(_py(parsed))
        return ExtractResult(ok=True, doc_type="docx", provider=None, result=parsed)

    elif doc_type == "chat":
        # Decode plain text
        try:
            text = data.decode("utf-8")
        except Exception:
            text = data.decode("latin-1", errors="ignore")

        try:
            extractor = FinancialNERExtractor(model_name="dslim/bert-base-NER", use_gpu=False)
            structured = extractor.extract_financial_entities_structured(text)
            entities = extractor.extract_all_entities(text)

            # Normalize to JSON-safe shape
            entities_json = []
            for e in entities:
                # entity_type may be an Enum
                etype = getattr(e, "entity_type", None)
                etype_val = getattr(etype, "value", str(etype)) if etype is not None else None
                entities_json.append({
                    "text": str(getattr(e, "text", "")),
                    "type": str(etype_val) if etype_val is not None else "UNKNOWN",
                    "start": int(getattr(e, "start_pos", 0)),
                    "end": int(getattr(e, "end_pos", 0)),
                    "confidence": float(getattr(e, "confidence", 0.0)),
                    "metadata": _py(getattr(e, "metadata", {}) or {}),
                })

            out = {
                "structured": _py(structured),
                "entities": entities_json,
                "raw_text_chars": len(text),
            }
        except Exception as e:
            raise HTTPException(500, f"Chat NER failed: {e}")

        out = jsonable_encoder(_py(out))
        return ExtractResult(ok=True, doc_type="chat", provider=None, result=out)

    else:
        raise HTTPException(400, "Unsupported doc_type.")
