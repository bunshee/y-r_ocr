import asyncio
import io
import re
from typing import Dict, List
import pandas as pd
from google import genai
from google.genai import types


class AsyncValidationAgent:
    """Asynchronous agent that validates and corrects OCR output with arrow rule application."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-image"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.validation_results = {}
        
    async def validate_and_correct_async(
        self, 
        page_num: int, 
        image_bytes: bytes, 
        original_df: pd.DataFrame,
        original_raw_text: str
    ) -> Dict:
        """Validate a single page asynchronously and apply arrow rule corrections."""
        
        try:
            # Re-analyze the image with focus on arrows/lines
            corrected_df = await self._reanalyze_with_arrow_detection(
                image_bytes, original_df, original_raw_text
            )
            
            # Apply arrow rule
            final_df = self._apply_arrow_rule(corrected_df)
            
            # Calculate differences
            differences = self._find_differences(original_df, final_df)
            
            return {
                "page_num": page_num,
                "corrected_df": final_df,
                "differences": differences,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "page_num": page_num,
                "corrected_df": original_df,
                "differences": [],
                "status": "error",
                "error": str(e)
            }
    
    async def _reanalyze_with_arrow_detection(
        self, 
        image_bytes: bytes, 
        original_df: pd.DataFrame,
        original_raw_text: str
    ) -> pd.DataFrame:
        """Re-analyze image with specific focus on vertical arrows and lines."""
        
        file_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        
        prompt = f"""Analyze this table image and identify ALL vertical arrows, lines, or checkmarks that indicate value duplication.

**Arrow/Line Rule:**
- When a vertical arrow (↓), line (|), or line with checkmark (✓) is drawn through cells in a column
- The value in the FIRST cell at the top of the arrow/line must be copied to ALL cells below it
- This continues until the arrow/line ends OR a new unique value appears

**Output Instructions:**
1. Identify each column that has vertical arrows/lines
2. For each such column, note:
   - Starting row (where the value is)
   - Ending row (where arrow/line ends)
   - The value to duplicate
3. Return ONLY the corrected CSV data with arrow rules applied
4. DO NOT include markdown fences or explanations

Original extraction for reference:
{original_raw_text[:500]}

Output the complete corrected CSV now:"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[file_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="text/plain",
                ),
            )
            
            response_text = response.text.strip()
            cleaned_csv = re.sub(r"(?i)^```csv\n|```$", "", response_text).strip()
            df = pd.read_csv(io.StringIO(cleaned_csv))
            return df
            
        except Exception as e:
            print(f"⚠️ Re-analysis failed: {e}. Using original data.")
            return original_df
    
    def _apply_arrow_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply arrow rule: copy first non-empty value down until a new value or checkmark."""
        
        if df.empty:
            return df
        
        corrected_df = df.copy()
        
        for col in corrected_df.columns:
            current_value = None
            
            for idx in range(len(corrected_df)):
                cell_value = corrected_df.at[idx, col]
                
                # Check if cell has checkmark indicator
                if pd.notna(cell_value) and str(cell_value).strip() in ['✓', '✔️', '✔', 'checkmark']:
                    # Keep the checkmark and stop duplication for this segment
                    current_value = None
                    continue
                
                # If we find a non-empty value, use it as the current value
                if pd.notna(cell_value) and str(cell_value).strip() and str(cell_value).strip() not in ['', 'nan']:
                    current_value = cell_value
                # If cell is empty but we have a current value, duplicate it
                elif current_value is not None and (pd.isna(cell_value) or str(cell_value).strip() == ''):
                    corrected_df.at[idx, col] = current_value
        
        return corrected_df
    
    def _find_differences(self, original_df: pd.DataFrame, corrected_df: pd.DataFrame) -> List[Dict]:
        """Find differences between original and corrected dataframes."""
        
        differences = []
        
        if original_df.shape != corrected_df.shape:
            differences.append({
                "type": "shape_mismatch",
                "original": original_df.shape,
                "corrected": corrected_df.shape
            })
            return differences
        
        for col in original_df.columns:
            if col not in corrected_df.columns:
                continue
                
            for idx in range(len(original_df)):
                orig_val = original_df.at[idx, col] if idx < len(original_df) else None
                corr_val = corrected_df.at[idx, col] if idx < len(corrected_df) else None
                
                if str(orig_val) != str(corr_val):
                    differences.append({
                        "row": idx,
                        "column": col,
                        "original": orig_val,
                        "corrected": corr_val
                    })
        
        return differences
    
    async def validate_all_pages(
        self, 
        pages_data: List[Dict]
    ) -> Dict[int, Dict]:
        """Validate all pages asynchronously without blocking."""
        
        tasks = []
        for page_data in pages_data:
            task = self.validate_and_correct_async(
                page_data["page_num"],
                page_data["image"],
                page_data["data"],
                page_data.get("raw_text", "")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        validation_results = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Validation error: {result}")
                continue
            validation_results[result["page_num"]] = result
        
        return validation_results
