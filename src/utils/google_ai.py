import time
from typing import Dict, List, Tuple

from google.genai import types


def send_prompt_with_retry(
    client,
    model_name,
    parts,
    system_prompt,
    max_retries,
    response_mime_type="text/plain",
    schema=None,
    temperature=0.0,
):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=parts + [system_prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type=response_mime_type,
                    response_schema=types.Schema(**schema) if schema else None,
                ),
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + (time.time() % 1)
                print(
                    f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"❌ API call failed after {max_retries} attempts: {e}")
    raise last_error


def detect_mime_type(file_path: str) -> str:
    ext = file_path.lower().split(".")[-1]
    mime_types = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "webp": "image/webp",
    }
    return mime_types.get(ext, "application/octet-stream")


def validate_extraction(result: Dict) -> Tuple[float, List[str]]:
    issues = []
    confidence_score = 1.0

    if not result.get("fields"):
        issues.append("No fields extracted")
        confidence_score -= 0.3

    if not result.get("line_items") and not result.get("metadata"):
        issues.append("No data extracted (empty metadata and line_items)")
        confidence_score -= 0.4

    if result.get("confidence") == "low":
        confidence_score -= 0.2
    elif result.get("confidence") == "medium":
        confidence_score -= 0.1

    fields = result.get("fields", [])
    field_names = {f.get("name") for f in fields}

    line_items = result.get("line_items", [])
    if line_items and isinstance(line_items, list) and line_items[0]:
        first_item_keys = set(line_items[0].keys())
        if not field_names.issuperset(first_item_keys):
            issues.append(
                f"Line item keys ({list(first_item_keys - field_names)[:2]}...) not fully defined in schema."
            )

    confidence_score = max(0.0, min(1.0, confidence_score))

    if issues:
        print(f"⚠️ Validation issues ({len(issues)}): {', '.join(issues[:3])}")

    return confidence_score, issues
