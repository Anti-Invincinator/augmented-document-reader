from __future__ import annotations

import os
import json
import tempfile
from typing import Optional, Literal, Dict, Any, List, Tuple

import re

from dotenv import load_dotenv
load_dotenv()


# -------------------------
# Public entry point
# -------------------------

def extract_pdf_entities(
    pdf_bytes: bytes,
    provider: Literal["gemini", "openai"] = "gemini",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extracts financial entities from a PDF using the requested provider.

    Returns a dict with only JSON-serializable primitives.
    """
    if provider == "openai":
        return _extract_with_openai(pdf_bytes, model)
    else:
        return _extract_with_gemini(pdf_bytes, model)


# -------------------------
# Gemini path
# -------------------------

_GEMINI_PROMPT = """You are a financial NER extractor. Read the attached PDF and extract key entities.
Return ONLY valid JSON, no prose, matching the schema:

{
  "entities": [
    { "label": "INVOICE_NUMBER", "text": "..." },
    { "label": "PO_NUMBER", "text": "..." },
    { "label": "VENDOR", "text": "..." },
    { "label": "CUSTOMER", "text": "..." },
    { "label": "TOTAL_AMOUNT", "text": "...", "currency": "USD" },
    { "label": "SUBTOTAL", "text": "...", "currency": "USD" },
    { "label": "TAX_AMOUNT", "text": "...", "currency": "USD" },
    { "label": "DUE_DATE", "text": "YYYY-MM-DD" },
    { "label": "ISSUE_DATE", "text": "YYYY-MM-DD" },
    { "label": "ACCOUNT_NUMBER", "text": "..." },
    { "label": "IBAN", "text": "..." },
    { "label": "BANK_NAME", "text": "..." },
    { "label": "TAX_ID", "text": "..." },
    { "label": "CURRENCY", "text": "USD" },
    { "label": "ADDRESS", "text": "..." }
  ]
}

Rules:
- Include only entities you find; omit others.
- Prefer ISO dates (YYYY-MM-DD).
- Prefer ISO-4217 currency codes when present.
- Keep values concise (no extra commentary).
"""

def _extract_with_gemini(pdf_bytes: bytes, model: Optional[str]) -> Dict[str, Any]:
    """
    Uploads the PDF to Gemini Files API and prompts for structured JSON entities.
    Ensures only primitive types are returned.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (env or .env).")

    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai") from e

    genai.configure(api_key=api_key)
    model_name = model or "gemini-1.5-flash"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        uploaded = genai.upload_file(
            path=tmp_path,
            mime_type="application/pdf",
            display_name=os.path.basename(tmp_path),
        )

        gmodel = genai.GenerativeModel(model_name)
        resp = gmodel.generate_content(
            [uploaded, {"text": _GEMINI_PROMPT}],
            request_options={"timeout": 180},
        )

        # Ask for pure JSON; be defensive on parsing anyway
        text = (getattr(resp, "text", None) or "").strip()
        data = _coerce_json(text)
        entities = _normalize_entities(data)

        # Token usage (primitives only)
        usage = {}
        try:
            um = getattr(resp, "usage_metadata", None)
            if um:
                usage = {
                    "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                    "candidates_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                    "total_tokens": int(getattr(um, "total_token_count", 0) or 0),
                }
        except Exception:
            usage = {}

        return {
            "provider": "gemini",
            "model": model_name,
            "entities": entities,
            "raw": text,  # raw JSON string from the model
            "meta": {
                "file_uri": getattr(uploaded, "uri", None),
                "usage": usage,
            },
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# -------------------------
# OpenAI path
# -------------------------

_OPENAI_SYSTEM = """You are a financial NER extractor. Given a chunk of extracted PDF text, 
return ONLY valid JSON with the schema:

{
  "entities": [
    { "label": "INVOICE_NUMBER", "text": "..." },
    { "label": "PO_NUMBER", "text": "..." },
    { "label": "VENDOR", "text": "..." },
    { "label": "CUSTOMER", "text": "..." },
    { "label": "TOTAL_AMOUNT", "text": "...", "currency": "USD" },
    { "label": "SUBTOTAL", "text": "...", "currency": "USD" },
    { "label": "TAX_AMOUNT", "text": "...", "currency": "USD" },
    { "label": "DUE_DATE", "text": "YYYY-MM-DD" },
    { "label": "ISSUE_DATE", "text": "YYYY-MM-DD" },
    { "label": "ACCOUNT_NUMBER", "text": "..." },
    { "label": "IBAN", "text": "..." },
    { "label": "BANK_NAME", "text": "..." },
    { "label": "TAX_ID", "text": "..." },
    { "label": "CURRENCY", "text": "USD" },
    { "label": "ADDRESS", "text": "..." }
  ]
}

Rules:
- If an entity isn't present, omit it.
- Use ISO dates when possible (YYYY-MM-DD).
- Use ISO-4217 currency codes when present.
- JSON only, no markdown/prose.
"""

def _extract_with_openai(pdf_bytes: bytes, model: Optional[str]) -> Dict[str, Any]:
    """
    Extracts text locally, then prompts OpenAI per chunk; merges entities.
    Returns only JSON-serializable primitives.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    try:
        from openai import OpenAI  # openai==1.x
    except Exception as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e

    text = _extract_text_from_pdf_bytes(pdf_bytes)
    if not text.strip():
        return {
            "provider": "openai",
            "model": model or "gpt-4o-mini",
            "entities": [],
            "raw": "",
            "meta": {"note": "No extractable text found in PDF."},
        }

    chunks = _chunk_text(text, max_chars=8000)
    client = OpenAI(api_key=api_key)
    model_name = model or "gpt-4o-mini"

    all_entities: List[Dict[str, Any]] = []
    raw_responses: List[str] = []

    for idx, chunk in enumerate(chunks):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": _OPENAI_SYSTEM},
                    {"role": "user", "content": f"PDF TEXT CHUNK {idx+1}/{len(chunks)}:\n\n{chunk}"},
                ],
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            content = f'{{"entities": [], "error": "OpenAI call failed for chunk {idx+1}: {e}"}}'

        raw_responses.append(content)
        data = _coerce_json(content)
        ents = _normalize_entities(data)
        all_entities.extend(ents)

    entities = _dedup_entities(all_entities)

    return {
        "provider": "openai",
        "model": model_name,
        "entities": entities,
        "raw": "\n".join(raw_responses),
        "meta": {"chunks": len(chunks), "chars": len(text)},
    }


# -------------------------
# Helpers
# -------------------------

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text using pypdf; fallback to pdfminer.six if needed.
    """
    try:
        from pypdf import PdfReader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            reader = PdfReader(tmp_path)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text() or ""
                pages_text.append(t)
            return "\n".join(pages_text)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        try:
            from io import BytesIO
            from pdfminer.high_level import extract_text
            return extract_text(BytesIO(pdf_bytes))
        except Exception as e:
            raise RuntimeError("Unable to extract text from PDF. Install 'pypdf' or 'pdfminer.six'.") from e


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    overlap = 600
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _coerce_json(text: str) -> Dict[str, Any]:
    """
    Try to parse an LLM response into JSON. Handles code fences and partial blocks.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1]
        cleaned = cleaned.split("```")[0]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return {"entities": data}
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = cleaned[first : last + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return {"entities": data}
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {"entities": []}


def _normalize_entities(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize to a list of {"label": str, "text": str, ...}
    """
    ents_raw = data.get("entities", [])
    entities: List[Dict[str, Any]] = []

    if isinstance(ents_raw, list):
        for item in ents_raw:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("type") or item.get("entity") or "").strip()
                text = str(item.get("text") or item.get("value") or "").strip()
                if not text:
                    continue
                ent: Dict[str, Any] = {"label": label or "UNKNOWN", "text": text}
                for k in ("currency", "value", "score", "confidence", "start", "end"):
                    if k in item and item[k] is not None:
                        ent[k] = item[k]
                entities.append(ent)
    elif isinstance(ents_raw, dict):
        for label, value in ents_raw.items():
            if isinstance(value, list):
                for v in value:
                    v_text = str(v).strip()
                    if v_text:
                        entities.append({"label": str(label), "text": v_text})
            else:
                v_text = str(value).strip()
                if v_text:
                    entities.append({"label": str(label), "text": v_text})
    return entities


def _dedup_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for e in entities:
        label = str(e.get("label", "")).strip().lower()
        text = str(e.get("text", "")).strip().lower()
        key = (label, text)
        if text and key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped
