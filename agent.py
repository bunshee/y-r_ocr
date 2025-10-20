import json
import os

import pandas as pd
from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter


class SelfDescribingOCRAgent:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _send_prompt(
        self, parts, system_prompt, response_mime_type="text/plain", schema=None
    ):
        """Helper to send a prompt and return response text."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts + [system_prompt],
            config=types.GenerateContentConfig(
                response_mime_type=response_mime_type,
                response_schema=types.Schema(**schema) if schema else None,
            ),
        )
        return response.text.strip()

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

    def process(
        self, file_path, custom_instructions="", auto_infer=True, is_page=False
    ):
        """
        Full pipeline: infer schema -> extract data -> normalize.
        """
        print(f"🔍 Processing: {file_path}")

        if file_path.lower().endswith(".pdf") and not is_page:
            return self.process_pdf_page_by_page(
                file_path, custom_instructions, auto_infer
            )

        # Step 1: Infer schema
        schema_info = (
            self.infer_schema(file_path, custom_instructions) if auto_infer else None
        )

        # Step 2: Extract data
        data = self.extract_with_inferred_schema(
            file_path, schema_info, custom_instructions
        )
        df = pd.DataFrame(data)

        if df.empty:
            print("⚠️ No data extracted. Exiting process.")
            return {}, pd.DataFrame(), schema_info

        # Step 3: Parse dates intelligently
        if schema_info:
            date_fields = [
                f["name"] for f in schema_info["fields"] if f["type"] == "date"
            ]
            for col in date_fields:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Step 4: Normalize data
        metadata, line_items = self.normalize_data(df, schema_info)

        print("✅ Processing complete.")
        return metadata, line_items, schema_info

    def process_pdf_page_by_page(
        self, file_path, custom_instructions="", auto_infer=True
    ):
        """
        Splits a PDF and processes each page, yielding results.
        """
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            page_path = os.path.abspath(f"temp_page_{i}.pdf")
            with open(page_path, "wb") as f:
                writer.write(f)

            print(f"📄 Processing page {i + 1}/{len(reader.pages)}")
            metadata, line_items, schema_info = self.process(
                page_path, custom_instructions, auto_infer, is_page=True
            )
            yield i + 1, metadata, line_items, schema_info
            os.remove(page_path)
