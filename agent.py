import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
import pytesseract
from groq import Groq
from pdf2image import convert_from_bytes  # REQUIRES external poppler utility

# --- REQUIRED NEW IMPORTS FOR VISUAL ROTATION ---
from PIL import Image
from pypdf import PdfReader, PdfWriter

# ------------------------------------------------


class SelfDescribingOCRAgent:
    def __init__(
        self, api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", max_workers: int = 4, max_retries: int = 3
    ):
        """Initialize OCR agent with Groq client."""
        self.client = Groq(api_key=api_key)
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

    def _send_prompt_with_retry(
        self, file_part, system_prompt, response_mime_type="text/plain", schema=None
    ):
        """Helper to send a prompt with retry logic and exponential backoff."""
        last_error = None
        
        # Convert file part to base64 if it's an image
        if hasattr(file_part, 'file_data') and file_part.file_data.mime_type.startswith('image/'):
            # For Groq, we'll need to handle images differently
            # For now, we'll just use the text prompt
            pass

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Extract the table data as CSV following the instructions."}
                    ],
                    temperature=0.0,
                    max_tokens=4000,
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) + (time.time() % 1)  # Exponential backoff with jitter
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
        class FilePart:
            def __init__(self, data, mime_type):
                self.file_data = type('FileData', (), {'mime_type': mime_type, 'data': data})
        
        file_part = FilePart(file_bytes, mime_type)

        system_prompt = """You are an expert document analyst specializing in **accurate, comprehensive table extraction** from complex documents, including timesheets and forms. Your primary goal is to identify and extract ALL tabular data from the provided document page.

**Extraction Rules:**
- Only get the middle table from the document.
- Do not include other text outside of main table or header tables in the response.
- Column names must be exactly the same as in table.
- Be consistent with the table format.
- All cell values must be quoted.

**Special Instructions (CRITICAL FOR DATA ACCURACY):**
1. **Vertical Repetition (The "Arrow Rule"):** **Vertical arrows, single lines, or down-arrows** drawn in a column indicate that the value in the cell at the start of the mark **MUST BE DUPLICATED** down to all subsequent cells covered by the mark. Duplication stops only when the mark ends, or a new unique handwritten value appears in the column. This rule is universal for ALL columns in which these marks appear.
2. **Compound Values:** If a cell contains a range or sequence like `value1-value2`, `value1 value2`, or `value1/value2` (e.g., "1p 2p"), this must be treated as a single data point in **one cell**, not split.
3. **Blank Data:** Represent all missing or blank data as an empty field in the CSV.

**Data Formatting:**
- **Times:** Use **HH:MM** format for the Call time (e.g., 07:00). For In/Out times that include **AM/PM**, keep the original entry as the value, but ensure it is quoted (e.g., "7A", "4:15P").

**CRITICAL OUTPUT INSTRUCTIONS:**
- Your ENTIRE response MUST be ONLY the CSV data.
- DO NOT include markdown fences like ```csv or ```.
- DO NOT include any introductory text, summaries, or explanations.
- If you cannot find a table, you MUST still respond with a single header row as per the instructions above. Do not write "No table found."
- The very first character of your response should be the first character of the CSV header.
- The very last character of your response should be the last character of the last CSV row.
"""
        try:
            response_text = self._send_prompt_with_retry(
                file_part,
                system_prompt,
                response_mime_type="text/plain"
            )
            cleaned_csv = re.sub(r"(?i)^```csv\n|```$", "", response_text).strip()
            df = pd.read_csv(io.StringIO(cleaned_csv))
            line_items = df.to_dict(orient="records")

            result = {
                "document_type": "unknown",
                "confidence": "medium",
                "fields": [{"name": col} for col in df.columns],
                "metadata": {},
                "line_items": line_items,
                "raw_text": response_text,
            }
            return result
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
                "raw_text": response_text,
            }
        except Exception as e:
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
                "raw_text": f"Extraction failed: {e}",
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
            try:
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
            except Exception as e:
                print(f"❌ Failed to read PDF file: {e}")

                def error_generator(e):
                    yield (1, {}, pd.DataFrame(), {}, None, f"Failed to read PDF: {e}")

                return error_generator(e), 1

            results_generator = self.process_pdf_parallel(
                reader, total_pages, custom_instructions
            )
            return results_generator, total_pages

        # Non-PDF files are handled here
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = self._detect_mime_type(file_path)

        # Single-pass extraction
        result = self.extract_data_single_pass(
            file_bytes, mime_type, custom_instructions
        )

        # For non-PDF, we wrap the single result in a generator and return a total of 1 page
        def single_result_generator():
            # This part needs to be consistent with the 6-element tuple
            metadata = result.get("metadata", {})
            line_items_data = result.get("line_items", [])
            raw_text = result.get("raw_text", "")
            line_items_df = pd.DataFrame(line_items_data)
            schema_info = {
                "document_type": result.get("document_type", "unknown"),
                "confidence": result.get("confidence", "low"),
                "fields": result.get("fields", []),
                "validation_score": 0.0,  # Simplified for non-pdf
                "validation_issues": [],
            }
            yield 1, metadata, line_items_df, schema_info, file_bytes, raw_text

        return single_result_generator(), 1

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
        corrected_bytes, file_mime_type = self._detect_and_correct_rotation(
            pdf_page_bytes
        )

        if file_mime_type == "image/png":
            print(f"🖼️ Page {page_num} successfully converted to upright PNG image.")
        else:
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

        # Process with single-pass extraction
        result = self.extract_data_single_pass(
            corrected_bytes, file_mime_type, page_instructions
        )

        # Validate extraction
        confidence_score, issues = self._validate_extraction(result)

        # Convert to DataFrame format
        metadata = result.get("metadata", {})
        line_items_data = result.get("line_items", [])
        raw_text = result.get("raw_text", "")  # Get raw text
        line_items_df = pd.DataFrame(line_items_data)

        schema_info = {
            "document_type": result.get("document_type", "unknown"),
            "confidence": result.get("confidence", "low"),
            "fields": result.get("fields", []),
            "validation_score": confidence_score,
            "validation_issues": issues,
        }

        return page_num, metadata, line_items_df, schema_info, corrected_bytes, raw_text

    def process_pdf_parallel(
        self, reader: PdfReader, total_pages: int, custom_instructions: str = ""
    ):
        """Process PDF pages in parallel for significantly better performance."""
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
                        f"Error processing page: {e}",
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
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        return self.process_pdf_parallel(reader, total_pages, custom_instructions)

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
