import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
import pytesseract
from google import genai
from google.genai import types
from pdf2image import convert_from_bytes  # REQUIRES external poppler utility

# --- REQUIRED NEW IMPORTS FOR VISUAL ROTATION ---
from PIL import Image
from pypdf import PdfReader, PdfWriter

# ------------------------------------------------


class SelfDescribingOCRAgent:
    def __init__(
        self, api_key, model_name="gemini-2.5-flash", max_workers=4, max_retries=3
    ):
        """Initialize OCR agent with improved configuration."""
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

    # =========================================================================
    # METHOD FOR IMAGE ROTATION CORRECTION (Requires External Dependencies)
    # =========================================================================
    def _detect_and_correct_rotation(self, pdf_page_bytes: bytes) -> Tuple[bytes, str]:
        """
        Renders PDF page to image, detects visual rotation using Tesseract OSD,
        corrects it using PIL, and returns the upright image bytes.
        """
        try:
            # Step 1: Render PDF page to PIL Image (Requires pdf2image/poppler)
            # Render at 200 DPI for a good balance of quality and speed
            images = convert_from_bytes(
                pdf_page_bytes, first_page=1, last_page=1, dpi=200
            )
            img = images[0]

            # Step 2: Detect Rotation using Pytesseract OSD
            osd_data = pytesseract.image_to_osd(img)
            angle_match = re.search(r"(?<=Rotate: )\d+", osd_data)
            angle_to_rotate = int(angle_match.group(0)) if angle_match else 0

            if angle_to_rotate != 0:
                print(
                    f"🔄 Detected visual rotation of {angle_to_rotate}° degrees. Correcting..."
                )

                # Step 3: Rotate Image using PIL
                # Tesseract OSD gives the angle needed for CLOCKWISE rotation to be upright.
                # Use -angle_to_rotate to perform the necessary counter-clockwise (or clockwise) rotation.
                rotated_img = img.rotate(
                    -angle_to_rotate, expand=True, resample=Image.Resampling.BICUBIC
                )
            else:
                rotated_img = img
                print("✅ No visual rotation detected. Image is upright.")

            # Step 4: Convert final upright Image back to PNG bytes
            output = io.BytesIO()
            # Save as PNG as it's a lossless format, ideal for OCR input
            rotated_img.save(output, format="PNG")
            output.seek(0)

            return output.getvalue(), "image/png"

        except Exception as e:
            print(
                f"❌ Image Rotation Correction Failed (Check Poppler/Tesseract dependencies): {e}. Falling back to original PDF bytes."
            )
            # Fallback to original PDF if anything goes wrong in the image pipeline
            return pdf_page_bytes, "application/pdf"

    # =========================================================================
    # EXISTING HELPER METHODS (UNCHANGED)
    # =========================================================================
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
        """Single-pass extraction with optimized table-specific instructions."""
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        prompt = """You are an expert document analyst specializing in **accurate, comprehensive table extraction** from complex documents, including timesheets and forms. Your primary goal is to identify and extract ALL tabular data from the provided document page.

**Extraction Rules:**
1.  **Identify Main Table:** Scrutinize the entire page to find the main table.
2.  **Header Detection:** Accurately identify the column headers from the table.
3.  **Data Formatting:**
    - Times: Use **HH:MM** format (e.g., 07:30).
    - Missing/Blank Data: Represent as an empty field in the CSV (e.g., ,,).

**Special Instructions:**
- Vertical arrows / lines mean a value should be duplicated from the cell at the start of the arrow down to the cell at the end.
- If a cell contains a range like `value1-value2`, this should be treated as a single data point in one cell, not split unless they are clearly intended for different columns.

**Output Format:**
- You MUST return the extracted data in a single, well-formed CSV format.
- Do NOT include any explanatory text, notes, or apologies before or after the CSV data. The response should contain ONLY the CSV.
"""
        try:
            response_text = self._send_prompt_with_retry(
                [file_part],
                prompt,
                response_mime_type="text/plain",
                schema=None,
            )

            # Clean up the response text
            cleaned_text = response_text.strip()

            # Remove markdown code block markers if present
            if cleaned_text.startswith("```") and "\n" in cleaned_text:
                # Remove the first and last lines (the code block markers)
                cleaned_text = "\n".join(cleaned_text.split("\n")[1:-1])

            # Clean up any remaining backticks
            cleaned_text = cleaned_text.replace("```", "").strip()

            # Debug output
            print(
                f"Cleaned CSV text:\n{cleaned_text[:500]}..."
            )  # Print first 500 chars for debugging

            # Try to read as CSV
            try:
                df = pd.read_csv(
                    io.StringIO(cleaned_text), dtype=str, keep_default_na=False
                )

                # Ensure we have a valid DataFrame with columns
                if df.empty or len(df.columns) == 0:
                    raise ValueError("Empty or invalid CSV data")

                # Convert empty strings to None for consistency
                df = df.replace("", None)

            except pd.errors.EmptyDataError:
                print("Warning: Empty CSV data received")
                df = pd.DataFrame(columns=["Error"])
                df.loc[0] = ["No table data could be extracted from this page"]

            # Convert dataframe to list of dicts for line_items
            line_items = df.to_dict(orient="records")

            # Create the result structure
            result = {
                "document_type": "timesheet",  # Default document type
                "confidence": "high" if not df.empty else "low",
                "fields": [{"name": str(col)} for col in df.columns],
                "metadata": {},
                "line_items": line_items,
                "raw_data": df,  # Include the raw DataFrame for debugging
            }

            print(
                f"📄 Extracted {len(result.get('fields', []))} fields, "
                f"{len(result.get('line_items', []))} rows"
            )
            return result

        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
                "error": str(e),
            }

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

    def process(self, file_path, custom_instructions: str = ""):
        """
        Optimized pipeline using single-pass extraction.
        """
        print(f"🔍 Processing: {file_path}")

        if file_path.lower().endswith(".pdf"):
            # PDF is handled by the parallel processor
            return self.process_pdf_parallel(file_path, custom_instructions)

        # Non-PDF files are handled here
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = self._detect_mime_type(file_path)

        # Single-pass extraction
        result = self.extract_data_single_pass(
            file_bytes, mime_type, custom_instructions
        )

        # Validate extraction
        confidence_score, issues = self._validate_extraction(result)
        print(f"📊 Extraction confidence: {confidence_score:.2%}")

        # Convert to DataFrame format
        metadata = result.get("metadata", {})
        line_items_data = result.get("line_items", [])

        line_items_df = pd.DataFrame(line_items_data)

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

    def _process_page_in_memory(
        self, page, page_num: int, total_pages: int, custom_instructions: str = ""
    ) -> Tuple:
        """Process a single PDF page in memory with image-based rotation correction."""
        print(f"📄 Processing page {page_num}/{total_pages}")

        # 1. Get initial PDF page bytes for the corrector
        buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(buffer)
        buffer.seek(0)
        pdf_page_bytes = buffer.read()

        # 2. Perform Image Rotation Correction
        # This is the step that guarantees an upright image is sent to the model, IF DEPENDENCIES WORK.
        corrected_bytes, file_mime_type = self._detect_and_correct_rotation(
            pdf_page_bytes
        )

        if file_mime_type == "image/png":
            print(f"🖼️ Page {page_num} successfully converted to upright PNG image.")
        else:
            # This path is taken if the image rotation pipeline failed (expected based on previous error)
            print(f"⚠️ Page {page_num} sent as original PDF (MIME: {file_mime_type}).")

        # 3. Add **CRITICAL VISUAL FAILSAFE** instruction
        rotation_instruction = (
            "**VISUAL ROTATION FAILSAFE:** The image rotation step may have failed. If this page's content "
            "is visually rotated (e.g., 90 or 270 degrees), you **MUST** mentally rotate the image "
            "to the correct, upright orientation (0 degrees) and extract the data based on that corrected view. "
            "**DO NOT** output the data rotated or transposed."
        )

        # 4. Combine instructions
        page_instructions = (
            f"Page {page_num} of {total_pages}. "
            f"{rotation_instruction} "
            f"{custom_instructions}"
        )

        # Process with single-pass extraction, using the corrected bytes and MIME type
        result = self.extract_data_single_pass(
            corrected_bytes, file_mime_type, page_instructions
        )

        # Validate extraction
        confidence_score, issues = self._validate_extraction(result)

        # Convert to DataFrame format
        metadata = result.get("metadata", {})
        line_items_data = result.get("line_items", [])

        line_items_df = pd.DataFrame(line_items_data)

        schema_info = {
            "document_type": result.get("document_type", "unknown"),
            "confidence": result.get("confidence", "low"),
            "fields": result.get("fields", []),
            "validation_score": confidence_score,
            "validation_issues": issues,
        }

        return page_num, metadata, line_items_df, schema_info, corrected_bytes

    def process_pdf_parallel(self, file_path: str, custom_instructions: str = ""):
        """Process PDF pages in parallel for significantly better performance."""
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        print(
            f"📚 Processing {total_pages} pages in parallel (max {self.max_workers} workers)..."
        )
        start_time = time.time()

        results_buffer = {}
        next_page_to_yield = 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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

            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    result = future.result()
                    results_buffer[page_num] = result
                except Exception as e:
                    print(f"❌ Error processing page {page_num}: {e}")
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
                        None,
                    )

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
        """Legacy method retained for compatibility, redirects to parallel processing."""
        return self.process_pdf_parallel(file_path, custom_instructions)

    def _validate_extraction(self, result: Dict) -> Tuple[float, List[str]]:
        """Validate extracted data and compute confidence score."""
        issues = []
        confidence_score = 1.0
        # ... (Validation logic remains the same for consistency) ...

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

        # Validate field consistency (simplified check)
        fields = result.get("fields", [])
        field_names = {f.get("name") for f in fields}

        # Check if line_item fields are in schema
        line_items = result.get("line_items", [])
        if line_items:
            # Check the intersection of fields between schema and first record
            first_item_keys = set(line_items[0].keys())
            if not field_names.issuperset(first_item_keys):
                issues.append(
                    f"Line item keys ({list(first_item_keys - field_names)[:2]}...) not fully defined in schema."
                )

        confidence_score = max(0.0, min(1.0, confidence_score))

        if issues:
            print(f"⚠️ Validation issues ({len(issues)}): {', '.join(issues[:3])}")

        return confidence_score, issues
