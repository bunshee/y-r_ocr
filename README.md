# Young & Restless OCR

This project is a Streamlit web application that uses the Google Gemini API to perform Optical Character Recognition (OCR) on PDF files. It can automatically infer the schema of a document, extract the data, and then normalize it into a structured format. The application allows users to upload a PDF file, and then it displays the extracted data in a table. Finally, the user can download the extracted data as a CSV, PDF, or DOCX file.

## Installation

### With `uv` (recommended)

1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

### With `pip`

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set your Google API key as an environment variable:
```bash
export GOOGLE_API_KEY="your-api-key"
```

   **Note:** You can get your API key from the [Google AI Studio](https://aistudio.google.com/).

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL provided by Streamlit.

4. Upload a PDF file and click the "Process Document" button.

5. The extracted data will be displayed in a table.

6. You can then download the extracted data as a CSV, PDF, or DOCX file.

## Dependencies

- google-genai
- pandas
- reportlab
- streamlit
- pypdf
- python-docx
