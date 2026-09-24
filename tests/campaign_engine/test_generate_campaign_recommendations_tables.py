import pandas as pd
import pytest

from app.segmentation.customer_intelligence.campaign_engine.pipeline import (
    generate_campaign_recommendations_table as pipeline,
)


def test_generate_campaign_recommendations_table_success(monkeypatch):
    """Test that the pipeline builds recommendations and writes the result."""

    input_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001", "C002"],
            "risk_tier": ["High Risk", "Medium Risk"],
            "segment_label": [
                "Satisfied One-Time Buyers",
                "Satisfied One-Time Buyers",
            ],
            "value_tier": ["High", "Medium"],
            "clv": [500.0, 250.0],
        }
    )

    recommendations_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001", "C002"],
            "risk_tier": ["High Risk", "Medium Risk"],
            "segment_label": [
                "Satisfied One-Time Buyers",
                "Satisfied One-Time Buyers",
            ],
            "value_tier": ["High", "Medium"],
            "campaign_name": [
                "Repeat Purchase Incentive Program",
                "Second Purchase Follow-Up Campaign",
            ],
            "campaign_priority": [2, 4],
            "reason_code": ["RC04", "RC04"],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "build_campaign_input_base",
        lambda: input_df,
    )

    monkeypatch.setattr(
        pipeline,
        "recommend_campaigns",
        lambda df: recommendations_df,
    )

    written = {}

    def fake_to_sql(self, name, con, if_exists, index):
        written["table_name"] = name
        written["if_exists"] = if_exists
        written["index"] = index
        written["data"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)

    result = pipeline.generate_campaign_recommendations_table()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    assert list(result.columns) == pipeline.OUTPUT_COLUMNS

    assert written["table_name"] == "customer_campaign_recommendations"
    assert written["if_exists"] == "replace"
    assert written["index"] is False

    pd.testing.assert_frame_equal(
        written["data"].reset_index(drop=True),
        result.reset_index(drop=True),
    )


def test_generate_campaign_recommendations_table_calls_recommender(
    monkeypatch,
):
    """Test that the merged campaign input is passed to recommend_campaigns."""

    input_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["High Risk"],
            "segment_label": ["Satisfied One-Time Buyers"],
            "value_tier": ["High"],
            "clv": [500.0],
        }
    )

    recommendations_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["High Risk"],
            "segment_label": ["Satisfied One-Time Buyers"],
            "value_tier": ["High"],
            "campaign_name": ["Repeat Purchase Incentive Program"],
            "campaign_priority": [2],
            "reason_code": ["RC04"],
        }
    )

    received = {}

    def fake_recommender(df):
        received["input"] = df.copy()
        return recommendations_df

    monkeypatch.setattr(
        pipeline,
        "build_campaign_input_base",
        lambda: input_df,
    )

    monkeypatch.setattr(
        pipeline,
        "recommend_campaigns",
        fake_recommender,
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        lambda *args, **kwargs: None,
    )

    pipeline.generate_campaign_recommendations_table()

    pd.testing.assert_frame_equal(
        received["input"],
        input_df,
    )


def test_generate_campaign_recommendations_table_checks_required_columns(
    monkeypatch,
):
    
    input_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["High Risk"],
            "segment_label": ["Satisfied One-Time Buyers"],
            "value_tier": ["High"],
        }
    )

    invalid_output = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["High Risk"],
            "segment_label": ["Satisfied One-Time Buyers"],
            "value_tier": ["High"],
            "campaign_name": ["Repeat Purchase Incentive Program"],
            # campaign_priority is intentionally missing
            "reason_code": ["RC04"],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "build_campaign_input_base",
        lambda: input_df,
    )

    monkeypatch.setattr(
        pipeline,
        "recommend_campaigns",
        lambda df: invalid_output,
    )

    with pytest.raises(ValueError, match="missing expected column"):
        pipeline.generate_campaign_recommendations_table()


def test_generate_campaign_recommendations_table_writes_replace_mode(
    monkeypatch,
):
    """Test that the output table is replaced rather than appended."""

    input_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["High Risk"],
            "segment_label": ["Satisfied One-Time Buyers"],
            "value_tier": ["High"],
        }
    )

    recommendations_df = pd.DataFrame(
        {
            "customer_unique_id": ["C001"],
            "risk_tier": ["High Risk"],
            "segment_label": ["Satisfied One-Time Buyers"],
            "value_tier": ["High"],
            "campaign_name": ["Repeat Purchase Incentive Program"],
            "campaign_priority": [2],
            "reason_code": ["RC04"],
        }
    )

    sql_arguments = {}

    def fake_to_sql(self, name, con, if_exists, index):
        sql_arguments["name"] = name
        sql_arguments["if_exists"] = if_exists
        sql_arguments["index"] = index

    monkeypatch.setattr(
        pipeline,
        "build_campaign_input_base",
        lambda: input_df,
    )

    monkeypatch.setattr(
        pipeline,
        "recommend_campaigns",
        lambda df: recommendations_df,
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        fake_to_sql,
    )

    pipeline.generate_campaign_recommendations_table()

    assert sql_arguments["name"] == "customer_campaign_recommendations"
    assert sql_arguments["if_exists"] == "replace"
    assert sql_arguments["index"] is False


def test_print_summary(capsys):
    """Test that the summary function prints the expected sections."""

    output = pd.DataFrame(
        {
            "customer_unique_id": ["C001", "C002"],
            "risk_tier": ["High Risk", "Medium Risk"],
            "segment_label": [
                "Satisfied One-Time Buyers",
                "Satisfied One-Time Buyers",
            ],
            "value_tier": ["High", "Medium"],
            "campaign_name": [
                "Repeat Purchase Incentive Program",
                "Second Purchase Follow-Up Campaign",
            ],
            "campaign_priority": [2, 4],
            "reason_code": ["RC04", "RC04"],
        }
    )

    pipeline._print_summary(output)

    captured = capsys.readouterr()

    assert "customer_campaign_recommendations: first 10 rows" in captured.out
    assert "campaign_priority counts" in captured.out
    assert "reason_code counts" in captured.out
    assert "campaign_name counts" in captured.out
