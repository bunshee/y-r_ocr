import os
import sys
from pathlib import Path

if __name__ == "__main__":
    # Change the working directory to the `src` directory
    os.chdir(Path(__file__).parent / "src")
    # Run the streamlit app
    os.system("streamlit run app.py")
