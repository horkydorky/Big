import pandas as pd
from pathlib import Path

def load_sales_data():
    """
    Loads and cleans the sales data from the raw CSV file located in the project root.
    This function encapsulates all the cleaning steps from the EDA notebook.
    """
    try:
        # Get the full path to the directory of this file (dataset.py)
        # e.g., .../BUSINESS_INSIGHTS_GENERATOR/business_insights_generator/
        script_dir = Path(__file__).resolve().parent
        
        # Go up ONE level to get to the project root
        project_root = script_dir.parent
        
        # --- THIS IS THE CORRECT PATH FOR A FILE IN THE ROOT ---
        file_path = project_root / "output.csv"

    except Exception:
        # A simple fallback path
        file_path = Path("output.csv")

    if not file_path.exists():
        # Raise an error that Streamlit can display
        raise FileNotFoundError(f"Sales data file not found. Looked for: {file_path.resolve()}")

    df = pd.read_csv(file_path, low_memory=False)

    # Perform cleaning
    if 'Unnamed: ' in df.columns:
        # Find all unnamed columns and drop them
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in col]
        df.drop(columns=unnamed_cols, inplace=True)
        
    df.dropna(subset=['Amount'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Handle potential non-numeric postal codes before converting
    if 'ship-postal-code' in df.columns:
        df.dropna(subset=['ship-postal-code'], inplace=True)
        df['ship-postal-code'] = df['ship-postal-code'].astype(int).astype(str)

    df.dropna(subset=['Date'], inplace=True)

    return df