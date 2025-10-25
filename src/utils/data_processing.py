import pandas as pd


def clean_and_process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    def clean_cell(value):
        if pd.isna(value) or value == "" or value is None:
            return ""
        return str(value).strip()

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    display_df = df.map(clean_cell)

    checkmark_values = ["Checked", "Checkmark", "V", "✓", "✔", "☑", "✅"]
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda x: "" if x in checkmark_values else x
        )

    row_value_counts = (display_df != "").sum(axis=1)
    rows_to_fill = row_value_counts > 2

    for col in display_df.columns:
        non_empty_mask = display_df[col] != ""
        if non_empty_mask.any():
            filled_col = display_df[col].replace("", pd.NA).ffill().fillna("")
            display_df[col] = display_df[col].where(~rows_to_fill, filled_col)

    display_df = display_df.loc[:, (display_df != "").any(axis=0)]
    display_df = display_df[(display_df != "").any(axis=1)]

    return display_df
