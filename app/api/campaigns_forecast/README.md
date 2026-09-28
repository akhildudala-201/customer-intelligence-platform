# Campaigns, Churn Correlations & Forecasting API 

FastAPI endpoints :

| Area | What it serves |
|---|---|
| **Campaigns** | Which retention campaign each customer gets, why, and the rulebook behind it |
| **Churn correlations** | How predicted churn varies with delivery delay and review score |
| **Forecasting** | Next-period churn-rate and revenue forecasts with 95% confidence intervals |

This package exposes one `router` (in `router.py`). `app/api/main.py` mounts it together with the other teammates' routers.

## Folder structure

```
app/api/
├── __init__.py
├── main.py                     # connects all routers into one FastAPI app
└── campaigns_forecast/
    ├── __init__.py
    ├── router.py               # endpoints (thin: validation + response shaping)
    ├── repository.py           # all SQL queries
    ├── schemas.py              # Pydantic response/request models (shown in /docs)
    └── README.md
tests/test_campaigns_forecast_api.py
```

## Endpoints

All paths are under `API_PREFIX` (default `/api/v1`).

### Campaigns

| Method | Path | Description |
|---|---|---|
| GET | `/campaigns/active` | KPI tile: number of active campaigns, customers targeted, high-priority customers (priority 1-2), and a per-campaign breakdown |
| GET | `/campaigns/recommendations` | Paginated per-customer campaign, priority and reason code, plus aggregate counts. Filters: `campaign_name`, `risk_tier`, `segment_label`, `value_tier`, `campaign_priority` (1-5). Paging: `page`, `page_size` (max 500) |
| GET | `/campaigns/{campaign_name}/customers` | Customers targeted by one campaign, highest churn probability first. Paging: `page`, `page_size`. 404 if the campaign has no customers |
| GET | `/campaigns/reason-codes` | Reason code to human-readable explanation |
| GET | `/campaigns/rules` | Full rulebook: (risk tier, segment, value tier) to campaign, priority, reason; plus the default rule |
| GET | `/campaigns/by-segment` | Campaigns and customer counts grouped by segment |
| POST | `/campaigns/evaluate` | Suggest a campaign and reason for one customer (see below) |

**`POST /campaigns/evaluate`** takes either:

```json
{ "customer_unique_id": "abc123" }
```
returns the stored recommendation (`source: "database"`), or

```json
{ "risk_tier": "High Risk", "segment_label": "High-Value Satisfied Repeat Buyers", "value_tier": "High" }
```
applies the YAML rulebook (`source: "rules"`, good for what-if checks). If the combination has no explicit rule, the default rule is returned with `matched_explicit_rule: false`. Sending neither returns 422.

### Churn correlations

| Method | Path | Description |
|---|---|---|
| GET | `/churn/delivery-correlation` | Churn by delivery-delay bucket: 7+ days early, on time / up to 7 days early, 1-3 days late, 4-7 days late, 8+ days late |
| GET | `/churn/review-correlation` | Churn by average review score: Poor (<2), Below average (2-3), Average (3-4), Good (4-5), Perfect (5), No review |

Both accept `threshold` (0-1, default `0.5`): a customer counts as "predicted to churn" when `churn_probability >= threshold`. Each bucket returns `customer_count`, `avg_churn_probability` and `predicted_churn_rate_pct`.

### Forecasting

| Method | Path | Description |
|---|---|---|
| GET | `/forecast/churn` | Forecast churn rate (0-1) with 95% CI. Param: `horizon` (1-60, default 6) |
| GET | `/forecast/revenue` | Forecast revenue with 95% CI. Param: `horizon` (1-60, default 6) |
| GET | `/forecast/accuracy` | Holdout MAPE, model used and forecast period range for both forecasts |

Forecast periods and granularity (monthly) come from what `forecasting.py` last wrote; there is no granularity parameter. If `horizon` is larger than the periods available, all available periods are returned (`available_periods` says how many).

## Data these endpoints read

Produced by `scripts/run_pipeline.py`:

| Table | Used by |
|---|---|
| `customer_campaign_recommendations` | all `/campaigns/*` data endpoints |
| `churn_predictions` | `/campaigns/{name}/customers`, churn correlations |
| `customer_features` | churn correlations (`avg_delivery_delay_days`, `avg_review_score`) |
| `forecast_revenue_trend`, `forecast_churn_trend` | `/forecast/*` |
| `app/segmentation/customer_intelligence/campaign_rules_reasoncodes.yaml` | `reason-codes`, `rules`, `evaluate` |

## Setup

1. **Python 3.11+** and a running **MySQL** with the pipeline tables (see above).
2. Install dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` in the project root (copy `.env.example`):
   ```env
   DB_HOST=127.0.0.1
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_mysql_password
   DB_NAME=olist
   API_PREFIX=/api/v1
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
   ```
   `ALLOWED_ORIGINS` is comma-separated: add the URL of every frontend that calls the API.
4. If the tables are missing or empty, run the pipeline:
   ```bash
   python scripts/run_pipeline.py --skip-ingest   # --skip-ingest: raw data already loaded
   ```

## Run

From the **project root** (not from inside `app/`):

```bash
uvicorn app.api.main:app --reload --port 8000
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Warnings such as `Skipping app.api.ml_predictions.router (not created yet)` are normal while other teammates' routers don't exist yet.

Quick check:
```bash
curl http://127.0.0.1:8000/api/v1/campaigns/active
curl "http://127.0.0.1:8000/api/v1/forecast/churn?horizon=6"
```

## Tests

```bash
pytest tests/test_campaigns_forecast_api.py -v -rs
```

- Rulebook tests run without data.
- Database tests read the real tables and check that endpoints agree with each other (e.g. campaign counts add up to the total). They are **skipped** (not failed) if MySQL or a table is unavailable; `-rs` shows why.
- Tests are read-only.

## Response codes

| Code | Meaning |
|---|---|
| 200 | OK |
| 404 | Unknown campaign or customer, or the forecast table has no rows |
| 422 | Invalid input (bad parameter range, `evaluate` with no id or incomplete profile) |
| 503 | A required table is missing or MySQL is unreachable. The message says to run the pipeline; details are in the server log |

## Notes and assumptions

- **Delivery delay** is delivered date minus estimated date, so negative means early. The feature pipeline zero-fills missing values, so customers with no delivery data appear in the "on time / early" bucket.
- **"Predicted to churn"** in the correlation endpoints uses `threshold` (default 0.5). Keep it consistent with how the ML endpoints define it.
- **Priorities 1-2** count as "high priority" in `/campaigns/active` (`HIGH_PRIORITY_MAX` in `router.py`).
- Table names are constants in `repository.py`; all request values go through bound SQL parameters.

## Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError` about missing `DB_*` variables at startup | Create `.env` in the project root and set `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` |
| `ModuleNotFoundError: app` | Run `uvicorn` from the project root, not from inside `app/` |
| 503 on every endpoint | MySQL isn't running, or run `scripts/run_pipeline.py` |
| Frontend request blocked (CORS error in the browser) | Add the frontend URL to `ALLOWED_ORIGINS` and restart uvicorn |
| `'cryptography' package is required` | `pip install cryptography` (MySQL 8 auth) |
| Tests fail on import of `TestClient` | `pip install httpx` |
