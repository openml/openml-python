import openml
import pandas as pd
import tempfile
import os


def test_create_dataset_with_parquet(tmp_path):
    df = pd.DataFrame({
        "feature_num": [1, 2, 3],
        "feature_cat": ["x", "y", "z"]
    })
    parquet_path = tmp_path / "test_dataset.parquet"
    df.to_parquet(parquet_path)

    dataset = openml.datasets.functions.create_dataset(
        name="test_dataset_parquet",
        description="Test Parquet Upload",
        creator="qa-test",
        default_target_attribute="feature_num",
        attributes="auto",
        parquet_path=str(parquet_path),
        citation="Test citation",
    )

    assert dataset.name == "test_dataset_parquet"
    assert dataset.data_format == "parquet"
    assert os.path.exists(parquet_path)

