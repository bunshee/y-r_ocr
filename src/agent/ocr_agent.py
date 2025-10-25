import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

import pandas as pd
import pytesseract
from google.genai import types
from google.genai.client import Client
from pdf2image import convert_from_bytes
from PIL import Image
from pypdf import PdfReader, PdfWriter

from utils.google_ai import (
    detect_mime_type,
    send_prompt_with_retry,
    validate_extraction,
)


class SelfDescribingOCRAgent:
    def __init__(
        self,
        api_key,
        model_name="gemini-1.5-flash",
        max_workers=4,
        max_retries=3,
        temperature=0.0,
    ):
        self.client = Client(api_key=api_key)
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature

    def _detect_and_correct_rotation(self, pdf_page_bytes: bytes) -> Tuple[bytes, str]:
        try:
            images = convert_from_bytes(
                pdf_page_bytes, first_page=1, last_page=1, dpi=200
            )
            img = images[0]

            osd_data = pytesseract.image_to_osd(img)
            angle_match = re.search(r"(?<=Rotate: )\d+", osd_data)
            angle_to_rotate = int(angle_match.group(0)) if angle_match else 0

            if angle_to_rotate != 0:
                print(
                    f"🔄 Detected visual rotation of {angle_to_rotate}° degrees. Correcting..."
                )
                rotated_img = img.rotate(
                    -angle_to_rotate, expand=True, resample=Image.Resampling.BICUBIC
                )
            else:
                rotated_img = img
                print("✅ No visual rotation detected. Image is upright.")

            output = io.BytesIO()
            rotated_img.save(output, format="PNG")
            output.seek(0)

            return output.getvalue(), "image/png"

        except Exception as e:
            print(
                f"❌ Image Rotation Correction Failed (Check Poppler/Tesseract dependencies): {e}. Falling back to original PDF bytes."
            )
            return pdf_page_bytes, "application/pdf"

    def extract_data_single_pass(
        self, file_bytes: bytes, mime_type: str, custom_instructions: str = ""
    ) -> Dict:
        parts, system_prompt = self._construct_prompt(
            file_bytes, mime_type, custom_instructions
        )
        try:
            response_text = send_prompt_with_retry(
                self.client,
                self.model_name,
                parts,
                system_prompt,
                self.max_retries,
                response_mime_type="text/plain",
                temperature=self.temperature,
            )
            cleaned_csv = re.sub(r"(?i)^```csv\n|```$", "", response_text).strip()
            df = pd.read_csv(io.StringIO(cleaned_csv))

            date_prompt = "From the text, what is the date of the document? Respond with only the date and nothing else."
            date_response = send_prompt_with_retry(
                self.client,
                self.model_name,
                parts,
                date_prompt,
                self.max_retries,
                response_mime_type="text/plain",
            )

            result = {
                "document_type": "timesheet",
                "confidence": "high",
                "fields": [{"name": col} for col in df.columns],
                "metadata": {"date": date_response},
                "line_items": df.to_dict(orient="records"),
                "raw_text": response_text,
            }
            return result
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            print(f"Failed to parse CSV: {e}")
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
                "raw_text": f"CSV parsing failed: {e}",
            }
        except Exception as e:
            print(f"Extraction failed: {e}")
            return {
                "document_type": "unknown",
                "confidence": "low",
                "fields": [],
                "metadata": {},
                "line_items": [],
                "raw_text": f"Extraction failed: {e}",
            }

    def process(self, file_path, custom_instructions: str = ""):
        print(f"🔍 Processing: {file_path}")

        if file_path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
            except Exception as e:
                print(f"❌ Failed to read PDF file: {e}")

                def error_generator(e):
                    yield (
                        1,
                        {},
                        pd.DataFrame(),
                        {},
                        None,
                        f"Failed to read PDF: {e}",
                    )

                return error_generator(e), 1

            results_generator = self.process_pdf_parallel(
                reader, total_pages, custom_instructions
            )
            return results_generator, total_pages

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        mime_type = detect_mime_type(file_path)

        result = self.extract_data_single_pass(
            file_bytes, mime_type, custom_instructions
        )

        def single_result_generator():
            metadata = result.get("metadata", {})
            line_items_data = result.get("line_items", [])
            raw_text = result.get("raw_text", "")
            line_items_df = pd.DataFrame(line_items_data)
            schema_info = {
                "document_type": result.get("document_type", "unknown"),
                "confidence": result.get("confidence", "low"),
                "fields": result.get("fields", []),
                "validation_score": 0.0,
                "validation_issues": [],
            }
            yield 1, metadata, line_items_df, schema_info, file_bytes, raw_text

        return single_result_generator(), 1

    def _process_page_in_memory(
        self, page, page_num: int, total_pages: int, custom_instructions: str = ""
    ) -> Tuple:
        print(f"📄 Processing page {page_num}/{total_pages}")

        buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_page(page)
        writer.write(buffer)
        buffer.seek(0)
        pdf_page_bytes = buffer.read()

        corrected_bytes, file_mime_type = self._detect_and_correct_rotation(
            pdf_page_bytes
        )

        if file_mime_type == "image/png":
            print(f"🖼️ Page {page_num} successfully converted to upright PNG image.")
        else:
            print(f"⚠️ Page {page_num} sent as original PDF (MIME: {file_mime_type}).")

        page_instructions = f"Page {page_num} of {total_pages}. {custom_instructions}"

        result = self.extract_data_single_pass(
            corrected_bytes, file_mime_type, page_instructions
        )

        confidence_score, issues = validate_extraction(result)

        metadata = result.get("metadata", {})
        line_items_data = result.get("line_items", [])
        raw_text = result.get("raw_text", "")
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

    def _construct_prompt(self, file_bytes, mime_type, custom_instructions):
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        prompt = f'''You are an expert document analyst specializing in **accurate, comprehensive table extraction** from complex documents, including timesheets and forms. Your primary goal is to identify and extract ALL tabular data from the provided raw OCR text.

        ## Extraction Rules
        ***

        ### General Structural Rules
        * **Target Table:** Only get the **middle table** from the document.
        * **Context Exclusion:** Do not include other text outside of the main table or header tables in the response.
        * **Headers:** Column names **must be exactly the same** as in the table.
        * **Format Consistency:** Be consistent with the table format.
        * **Quoting:** All cell values **must be quoted**.

        ### Data Formatting
        * **Times:** Use **HH:MM** format for the Call time (e.g., 07:00). For In/Out times that include **AM/PM**.

        {custom_instructions}

        ## CRITICAL OUTPUT INSTRUCTIONS
        * Your ENRE response **MUST be ONLY the CSV data**.
        * **DO NOT** include markdown fences like ```csv or ```.
        * **DO NOT** include any introductory text, summaries, or explanations.
        * If you cannot find a table, you **MUST** still respond with a single header row as per the instructions above. Do not write "No table found."
        * The very first character of your response should be the first character of the CSV header.
        * The very last character of your response should be the last character of the last CSV row.
        '''
        return [file_part], prompt
