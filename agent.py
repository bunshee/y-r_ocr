import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter


class SelfDescribingOCRAgent:
    def __init__(
        self, api_key, model_name="gemini-2.5-flash", max_workers=4, max_retries=3
    ):
        """Initialize OCR agent with improved configuration.

        Args:
            api_key: Google API key
            model_name: Gemini model to use (default: gemini-2.0-flash-exp for better performance)
            max_workers: Maximum parallel workers for processing pages
            max_retries: Maximum retry attempts for API calls
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _send_prompt_with_retry(
        self, parts, system_prompt, response_mime_type="text/plain", schema=None
    ):
        """Helper to send a prompt with retry logic and exponential backoff."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts + [system_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type=response_mime_type,
                        response_schema=types.Schema(**schema) if schema else None,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) + (
                        time.time() % 1
                    )  # Exponential backoff with jitter
                    print(
                        f"⚠️ API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print(f"❌ API call failed after {self.max_retries} attempts: {e}")

        raise last_error

    def extract_data_single_pass(
        self, file_bytes: bytes, mime_type: str, custom_instructions: str = ""
    ) -> Dict:
        """Single-pass extraction that combines schema inference and data extraction.

        This replaces the two-step process (infer_schema + extract_with_inferred_schema)
        with a single API call for 50% latency reduction and better consistency.

        Args:
            file_bytes: Document file as bytes
            mime_type: MIME type of the document
            custom_instructions: Optional custom instructions

        Returns:
            Dictionary with document_type, confidence, metadata, line_items, and fields
        """
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        prompt = f"""You are an expert document analyst. Analyze this document and extract ALL structured data.

**Task:**
1. Identify the document type (invoice, receipt, form, timesheet, etc.)
2. Extract ALL data fields with their values
3. Classify each field as either:
   - "header": Applies to the whole document (e.g., date, invoice number, company name)
   - "line_item": Repeats in rows/records (e.g., item description, quantity, price)

**Instructions:**
- Use lowercase_snake_case for all field names
- For dates: use YYYY-MM-DD format
- For currency: extract as numbers without symbols
- For times: use HH:MM format
- Include header information context for every line item
- If a field is missing, use null

{custom_instructions}

**Output Format:**
Return a JSON object. First, list all the fields in a "fields" array with their metadata.
Then provide the extracted data:
- "metadata": object with header field values (field_name: value pairs)
- "line_items": array of objects, each containing line item field values

Example structure:
{{
  "document_type": "invoice",
  "confidence": "high",
  "fields": [
    {{"name": "invoice_number", "type": "string", "description": "Invoice ID", "context": "header"}},
    {{"name": "item_name", "type": "string", "description": "Product name", "context": "line_item"}}
  ],
  "metadata": {{"invoice_number": "INV-001"}},
  "line_items": [{{"item_name": "Product A"}}]
}}
"""

        # Note: Not using strict schema validation because metadata and line_items
        # have dynamic fields that can't be predefined. Using JSON mode only.
        try:
            response_text = self._send_prompt_with_retry(
                [file_part],
                prompt,
                response_mime_type="application/json",
                schema=None,  # No schema - allow free-form JSON
            )
            result = json.loads(response_text)
            
            # Ensure metadata and line_items exist even if not in schema response
            if "metadata" not in result:
                result["metadata"] = {}
            if "line_items" not in result:
                result["line_items"] = []
            
            print(
                f"📄 Document Type: {result.get('document_type', 'unknown')} "
                f"(Confidence: {result.get('confidence', 'unknown')})"
            )
            print(
                f"🔍 Extracted {len(result.get('fields', []))} fields, "
                f"{len(result.get('metadata', {}))} metadata fields, "
                f"{len(result.get('line_items', []))} line items"
            )
            return result
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text[:500]}...")
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
            }
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
            }

    def infer_schema(self, file_path, custom_instructions=""):
        """
        Step 1: Automatically infer the schema from the document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = (
            "application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"
        )
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        prompt = f"""
You are an expert document analyst. Analyze the attached document and answer:

1. What type of document is this?
2. What are the key data fields? For each field:
   - Name (use lowercase_snake_case)
   - Type (string, number, date, boolean, currency)
   - Description (brief)
   - Context: Is this a 'header' field (applies to the whole document) or a 'line_item' field (repeats in a table)?

{custom_instructions}

Respond in this JSON format:
{{
  "document_type": "string",
  "confidence": "high/medium/low",
  "fields": [
    {{"name": "field_name", "type": "string", "description": "what it means", "context": "header or line_item"}}
  ]
}}
"""
        try:
            response = self._send_prompt(
                [file_part],
                prompt,
                response_mime_type="application/json",
                schema={
                    "type": types.Type.OBJECT,
                    "properties": {
                        "document_type": types.Schema(type=types.Type.STRING),
                        "confidence": types.Schema(
                            type=types.Type.STRING, enum=["high", "medium", "low"]
                        ),
                        "fields": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "name": types.Schema(type=types.Type.STRING),
                                    "type": types.Schema(type=types.Type.STRING),
                                    "description": types.Schema(type=types.Type.STRING),
                                    "context": types.Schema(
                                        type=types.Type.STRING,
                                        enum=["header", "line_item"],
                                    ),
                                },
                            ),
                        ),
                    },
                },
            )
            inferred = json.loads(response)
            print(
                f"📄 Document Type: {inferred['document_type']} (Confidence: {inferred['confidence']})"
            )
            print(f"🔍 Inferred {len(inferred['fields'])} fields.")
            return inferred
        except Exception as e:
            print(
                f"⚠️ Schema inference failed: {e}. Proceeding with flexible extraction."
            )
            return {"document_type": "unknown", "confidence": "low", "fields": []}

    def extract_with_inferred_schema(
        self, file_path, schema_info=None, custom_instructions=""
    ):
        """
        Step 2: Extract data using the inferred schema.
        """
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = "application/pdf" if file_path.endswith(".pdf") else "image/jpeg"
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        if schema_info and schema_info["fields"]:
            field_list = ", ".join(
                [f"`{f['name']}` ({f['type']})" for f in schema_info["fields"]]
            )
            prompt = f"""
Extract all records from this {schema_info["document_type"]}.

Fields to extract: {field_list}

Instructions:
- Return a JSON array of objects with these keys.
- If a field is missing in a row, use null or empty string.
- Normalize dates to YYYY-MM-DD and currency to numbers.
- Format time values to hh:mm.
- Include header information on every row.

{custom_instructions}

Output only the JSON array.
"""
            schema_for_extraction = {
                "type": types.Type.ARRAY,
                "items": types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        f["name"]: types.Schema(
                            type=types.Type.STRING
                            if f["type"] in ["string", "date", "currency"]
                            else types.Type.NUMBER
                        )
                        for f in schema_info["fields"]
                    },
                ),
            }
        else:  # Fallback for flexible extraction
            prompt = f"""
Analyze this document and extract all meaningful structured data as a JSON array.
Instructions:
- Identify repeating records or key-value pairs.
- Return a JSON array of objects with consistent keys.
- Use descriptive field names in snake_case.
- Output only the JSON array.

{custom_instructions}
"""
            schema_for_extraction = None

        try:
            response = self._send_prompt(
                [file_part],
                prompt,
                response_mime_type="application/json",
                schema=schema_for_extraction,
            )
            data = json.loads(response)
            print(f"✅ Extracted {len(data)} records.")
            return data
        except json.JSONDecodeError:
            print("❌ Failed to parse JSON. Raw response:")
            print(response)
            return []

    def normalize_data(self, df, schema_info):
        """Splits flat DataFrame into metadata and line_items."""
        if not schema_info or not schema_info.get("fields"):
            print("⚠️ No schema context available. Returning raw data.")
            return {}, df

        header_fields = [
            f["name"] for f in schema_info["fields"] if f.get("context") == "header"
        ]
        line_item_fields = [
            f["name"] for f in schema_info["fields"] if f.get("context") == "line_item"
        ]

        # If context wasn't provided, fallback
        if not header_fields or not line_item_fields:
            print(
                "ℹ️ Schema did not provide clear context. Using first row for metadata."
            )
            metadata = df.iloc[0].to_dict()
            line_items = (
                df.drop(columns=metadata.keys(), errors="ignore")
                .drop_duplicates()
                .reset_index(drop=True)
            )
            return metadata, line_items

        # Extract metadata from the first non-null row for header fields
        metadata_series = df[header_fields].dropna(how="all").iloc[0]
        metadata = metadata_series.to_dict()

        # Extract line items, dropping any header columns for clarity
        line_items = df[line_item_fields].drop_duplicates().reset_index(drop=True)

        return metadata, line_items

    def _validate_extraction(self, result: Dict) -> Tuple[float, List[str]]:
        """Validate extracted data and compute confidence score.

        Args:
            result: Extraction result dictionary

        Returns:
            Tuple of (confidence_score, list of issues)
        """
        issues = []
        confidence_score = 1.0

        # Check if we have basic structure
        if not result.get("fields"):
            issues.append("No fields extracted")
            confidence_score -= 0.3

        if not result.get("line_items") and not result.get("metadata"):
            issues.append("No data extracted (empty metadata and line_items)")
            confidence_score -= 0.4

        # Check confidence level
        if result.get("confidence") == "low":
            confidence_score -= 0.2
        elif result.get("confidence") == "medium":
            confidence_score -= 0.1

        # Validate field consistency
        fields = result.get("fields", [])
        field_names = {f.get("name") for f in fields}

        # Check if metadata fields are in schema
        metadata = result.get("metadata", {})
        for key in metadata.keys():
            if key not in field_names:
                issues.append(f"Metadata field '{key}' not in schema")

        # Check if line_item fields are in schema
        line_items = result.get("line_items", [])
        if line_items:
            for item in line_items[:3]:  # Check first 3 items
                for key in item.keys():
                    if key not in field_names:
                        issues.append(f"Line item field '{key}' not in schema")
                        break

        confidence_score = max(0.0, min(1.0, confidence_score))

        if issues:
            print(f"⚠️ Validation issues ({len(issues)}): {', '.join(issues[:3])}")

        return confidence_score, issues

    def process(
        self, file_path, custom_instructions="", auto_infer=True, is_page=False
    ):
        """
        Optimized pipeline using single-pass extraction.

        Args:
            file_path: Path to file to process
            custom_instructions: Optional custom extraction instructions
            auto_infer: Use single-pass extraction (legacy parameter, now always True for new method)
            is_page: Internal flag for page-by-page processing

        Returns:
            Tuple of (metadata dict, line_items DataFrame, schema_info dict)
        """
        print(f"🔍 Processing: {file_path}")

        if file_path.lower().endswith(".pdf") and not is_page:
            return self.process_pdf_parallel(file_path, custom_instructions)

        # Read file and detect MIME type
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = self._detect_mime_type(file_path)

        # Single-pass extraction (replaces old two-step process)
        result = self.extract_data_single_pass(
            file_bytes, mime_type, custom_instructions
        )

        # Validate extraction
        confidence_score, issues = self._validate_extraction(result)
        print(f"📊 Extraction confidence: {confidence_score:.2%}")

        # Convert to DataFrame format
        metadata = result.get("metadata", {})
        line_items_data = result.get("line_items", [])

        if not line_items_data:
            print("⚠️ No line items extracted.")
            line_items_df = pd.DataFrame()
        else:
            line_items_df = pd.DataFrame(line_items_data)

            # Parse date fields
            date_fields = [
                f["name"] for f in result.get("fields", []) if f.get("type") == "date"
            ]
            for col in date_fields:
                if col in line_items_df.columns:
                    line_items_df[col] = pd.to_datetime(
                        line_items_df[col], errors="coerce"
                    )

        # Return in format compatible with existing code
        schema_info = {
            "document_type": result.get("document_type", "unknown"),
            "confidence": result.get("confidence", "low"),
            "fields": result.get("fields", []),
            "validation_score": confidence_score,
            "validation_issues": issues,
        }

        print("✅ Processing complete.")
        return metadata, line_items_df, schema_info

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type from file extension."""
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

    def _process_page_in_memory(
        self, page, page_num: int, total_pages: int, custom_instructions: str = ""
    ) -> Tuple:
        """Process a single PDF page in memory without disk I/O.

        Args:
            page: PyPDF page object
            page_num: Page number (1-indexed)
            total_pages: Total number of pages
            custom_instructions: Optional custom instructions

        Returns:
            Tuple of (page_num, metadata, line_items, schema_info)
        """
        print(f"📄 Processing page {page_num}/{total_pages}")

        # Create PDF in memory
        buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(buffer)
        buffer.seek(0)
        file_bytes = buffer.read()

        # Process with single-pass extraction
        result = self.extract_data_single_pass(
            file_bytes, "application/pdf", custom_instructions
        )

        # Validate extraction
        confidence_score, issues = self._validate_extraction(result)

        # Convert to DataFrame format
        metadata = result.get("metadata", {})
        line_items_data = result.get("line_items", [])

        if not line_items_data:
            line_items_df = pd.DataFrame()
        else:
            line_items_df = pd.DataFrame(line_items_data)

            # Parse date fields
            date_fields = [
                f["name"] for f in result.get("fields", []) if f.get("type") == "date"
            ]
            for col in date_fields:
                if col in line_items_df.columns:
                    line_items_df[col] = pd.to_datetime(
                        line_items_df[col], errors="coerce"
                    )

        schema_info = {
            "document_type": result.get("document_type", "unknown"),
            "confidence": result.get("confidence", "low"),
            "fields": result.get("fields", []),
            "validation_score": confidence_score,
            "validation_issues": issues,
        }

        return page_num, metadata, line_items_df, schema_info

    def process_pdf_parallel(self, file_path: str, custom_instructions: str = ""):
        """Process PDF pages in parallel for significantly better performance.

        Uses ThreadPoolExecutor to process multiple pages concurrently.
        Processes pages in-memory without disk I/O.
        Returns results in page order (sorted), not completion order.

        Args:
            file_path: Path to PDF file
            custom_instructions: Optional custom instructions

        Yields:
            Tuple of (page_num, metadata, line_items, schema_info) for each page in order
        """
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        print(
            f"📚 Processing {total_pages} pages in parallel (max {self.max_workers} workers)..."
        )
        start_time = time.time()

        # Process pages in parallel with ordered output
        # Buffer to hold completed pages until their turn
        results_buffer = {}
        next_page_to_yield = 1
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pages for processing
            futures = {
                executor.submit(
                    self._process_page_in_memory,
                    page,
                    i + 1,
                    total_pages,
                    custom_instructions,
                ): i + 1
                for i, page in enumerate(reader.pages)
            }

            # Process results as they complete, but yield in order
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    result = future.result()
                    results_buffer[page_num] = result
                except Exception as e:
                    print(f"❌ Error processing page {page_num}: {e}")
                    # Store empty result for failed page
                    results_buffer[page_num] = (
                        page_num,
                        {},
                        pd.DataFrame(),
                        {
                            "document_type": "unknown",
                            "confidence": "low",
                            "fields": [],
                            "validation_score": 0.0,
                            "validation_issues": [str(e)],
                        },
                    )
                
                # Yield any pages that are now ready in sequence
                while next_page_to_yield in results_buffer:
                    yield results_buffer[next_page_to_yield]
                    del results_buffer[next_page_to_yield]
                    next_page_to_yield += 1

        elapsed = time.time() - start_time
        print(
            f"✅ Completed {total_pages} pages in {elapsed:.2f}s ({elapsed / total_pages:.2f}s per page)"
        )

    def process_pdf_page_by_page(
        self, file_path, custom_instructions="", auto_infer=True
    ):
        """Legacy method - now redirects to parallel processing.

        Kept for backward compatibility.
        """
        return self.process_pdf_parallel(file_path, custom_instructions)
