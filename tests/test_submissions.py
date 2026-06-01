import os
import pandas as pd


def test_submission_format():
    """Validates that the prediction file meets strict platform criteria before submission."""
    submission_path = "submission.csv"

    # Rule 1: File must exist
    assert os.path.exists(
        submission_path
    ), "Submission file was not found in the root directory!"

    sub_df = pd.read_csv(submission_path)

    # Rule 2: Shape must be exactly 41778 rows by 2 columns
    assert sub_df.shape == (
        41778,
        2,
    ), f"Incorrect dimensions! Expected (41778, 2), got {sub_df.shape}"

    # Rule 3: Target and Index columns must have correct names
    assert (
        "Index" in sub_df.columns
    ), "Missing 'Index' column in submission file."
    assert (
        "demand" in sub_df.columns
    ), "Missing 'demand' column in submission file."

    # Rule 4: Ensure no null values exist in predictions
    assert (
        sub_df["demand"].isnull().sum() == 0
    ), "Submission contains NaN values!"