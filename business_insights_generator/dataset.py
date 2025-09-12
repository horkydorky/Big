import pandas as pd
from pathlib import Path

def load_sales_data():
    """
    Loads the sales data from the CSV file in the project root.
    """
    try:
        # Get the directory of this script (.../Big/business_insights_generator/)
        script_dir = Path(__file__).resolve().parent
        # Go up TWO levels to get to the project root
        project_root = script_dir.parents[0]
        # Build the path to the file in the root
        file_path = project_root /"output.csv"
    except Exception:
        file_path = Path("output.csv")

    if not file_path.exists():
        raise FileNotFoundError(f"Sales data file not found. Looked for: {file_path.resolve()}")

    # ... (the rest of the cleaning code is the same)
    df = pd.read_csv(file_path, low_memory=False)
    if 'Unnamed: ' in df.columns:
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in col]
        df.drop(columns=unnamed_cols, inplace=True)
    df.dropna(subset=['Amount'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'ship-postal-code' in df.columns:
        df.dropna(subset=['ship-postal-code'], inplace=True)
        df['ship-postal-code'] = df['ship-postal-code'].astype(int).astype(str)
    df.dropna(subset=['Date'], inplace=True)
    return df

def load_review_data():
    """
    Loads the customer review data from the CSV file in the project root.
    """
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parents[0]
        # Build the path to the file in the root
        file_path = project_root / "customer_Reviews.csv" # Using the name from your screenshot
    except Exception:
        file_path = Path("customer_Reviews.csv")

    if not file_path.exists():
        raise FileNotFoundError(f"Review data file not found. Looked for: {file_path.resolve()}")

    df_reviews = pd.read_csv(file_path)
    columns_to_keep = ['asins', 'reviews.rating', 'reviews.title', 'reviews.text']
    if not all(col in df_reviews.columns for col in columns_to_keep):
        raise ValueError("The customer_Review.csv file is missing required columns.")
        
    df_reviews_clean = df_reviews[columns_to_keep].copy()
    df_reviews_clean.dropna(inplace=True)
    df_reviews_clean.rename(columns={
        'asins': 'product_id', 'reviews.rating': 'rating',
        'reviews.title': 'review_title', 'reviews.text': 'review_text'
    }, inplace=True)
    return df_reviews_clean