import os

import pandas as pd
import streamlit as st

from agent.ocr_agent import SelfDescribingOCRAgent
from utils.file_converters import create_zip_archive

st.title("Young & Restless OCR")

if "edited_tables" not in st.session_state:
    st.session_state.edited_tables = {}

api_key = st.text_input("Enter your Google API Key:", type="password")


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


processing_successful = False
date_to_display = None

if uploaded_file is not None and api_key:
    file_key = f"file_{hash(uploaded_file.getvalue())}"

    if file_key not in st.session_state.edited_tables:
        st.session_state.edited_tables[file_key] = {
            "tables": [],
            "date": None,
            "processed": False,
        }

    temp_pdf_path = os.path.abspath("temp.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if not st.session_state.edited_tables[file_key]["processed"]:
        if st.button("Process Document"):
            try:
                agent = SelfDescribingOCRAgent(api_key=api_key)
                results_generator, total_pages = agent.process(temp_pdf_path)

                st.write(f"Found {total_pages} pages. Processing...")
                progress_bar = st.progress(0)

                st.session_state.edited_tables[file_key]["tables"] = []

                for i, (
                    page_num,
                    _,
                    line_items,
                    _,
                    corrected_image_bytes,
                    raw_text,
                ) in enumerate(results_generator):

                    def clean_cell(value):
                        if pd.isna(value) or value == "" or value is None:
                            return ""
                        return str(value).strip()

                    if not isinstance(line_items, pd.DataFrame):
                        line_items = pd.DataFrame()

                    display_df = line_items.map(clean_cell)

                    checkmark_values = [
                        "Checked",
                        "Checkmark",
                        "V",
                        "✓",
                        "✔",
                        "☑",
                        "✅",
                    ]
                    for col in display_df.columns:
                        display_df[col] = display_df[col].apply(
                            lambda x: "" if x in checkmark_values else x
                        )

                    for col in display_df.columns:
                        non_empty_mask = display_df[col] != ""
                        if non_empty_mask.any():
                            display_df[col] = (
                                display_df[col].replace("", pd.NA).ffill().fillna("")
                            )

                    display_df = display_df.loc[:, (display_df != "").any(axis=0)]
                    display_df = display_df[(display_df != "").any(axis=1)]

                    st.session_state.edited_tables[file_key]["tables"].append(
                        {
                            "page_num": page_num,
                            "data": display_df,
                            "image": corrected_image_bytes,
                            "original_columns": line_items.columns.tolist(),
                            "raw_text": raw_text,
                        }
                    )

                    progress_bar.progress(int(((i + 1) / total_pages) * 100))

                st.session_state.edited_tables[file_key]["processed"] = True
                progress_bar.empty()

                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.edited_tables[file_key]["processed"] = False
    else:
        try:
            for table_data in st.session_state.edited_tables[file_key]["tables"]:
                page_num = table_data["page_num"]
                st.subheader(f"Page {page_num}")

                if table_data["image"]:
                    st.image(
                        table_data["image"],
                        caption=f"Rotated Image for Page {table_data['page_num']}",
                        width="stretch",
                    )

                if table_data.get("raw_text"):
                    with st.expander("View Raw Extracted Text"):
                        st.text(table_data["raw_text"])

                if not table_data["data"].empty:
                    try:
                        editor_key = f"editor_{file_key}_{table_data['page_num']}"

                        edited_df = st.data_editor(
                            table_data["data"],
                            key=editor_key,
                            column_config={
                                col: {"default": ""}
                                for col in table_data["data"].columns
                            },
                        )

                        table_data["data"] = edited_df

                    except Exception as e:
                        st.warning(f"Error displaying data editor: {e}")
                        st.dataframe(table_data["data"])
                else:
                    st.info("No data to display for this page.")

            if st.button("Download All Results"):
                tables_with_pages = [
                    (t["page_num"], t["data"])
                    for t in st.session_state.edited_tables[file_key]["tables"]
                    if not t["data"].empty
                ]

                if tables_with_pages:
                    zip_file_buffer = create_zip_archive(
                        tables_with_pages, date_to_display
                    )
                    st.download_button(
                        label="Download All Results (ZIP)",
                        data=zip_file_buffer,
                        file_name="extracted_results.zip",
                        mime="application/zip",
                    )

            processing_successful = True

        except Exception as e:
            st.error(f"An error occurred while displaying results: {e}")
