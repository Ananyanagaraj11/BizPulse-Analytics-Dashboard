<<<<<<< HEAD
# BizPulse-Analytics-Dashboard
=======
# BizPulse Dashboard

BizPulse is a Streamlit dashboard that demonstrates a complete analytics stack:
data, processing, visualization, and UI. It ships with a sample CSV dataset and
renders KPIs, donut charts, line + bar charts, a US revenue map, filters, and
tables using Plotly.

![BizPulse demo](assets/Main.png)

## Tech
- Python
- Pandas
- Streamlit
- Plotly

## Quick start
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `streamlit run app.py`

## Use real data or a database
You can load a CSV or connect to a database using environment variables.

### CSV
- `CSV_PATH` (default: `data/orders.csv`)

### Database
- `DATABASE_URL` (e.g., `sqlite:///data/orders.db` or a Postgres URL)
- `DB_TABLE` (default: `orders`)
- `SQL_QUERY` (optional, overrides `DB_TABLE`)

Example with SQLite:
- `python scripts/load_csv_to_sqlite.py`
- `set DATABASE_URL=sqlite:///data/orders.db` (Windows PowerShell: `$env:DATABASE_URL="sqlite:///data/orders.db"`)
- `streamlit run app.py`

### Postgres
Example (local Postgres):
- `set DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/bizpulse`
- `python scripts/load_csv_to_postgres.py --db-url %DATABASE_URL%`
- `streamlit run app.py`

## Project structure
- `app.py` - Streamlit app
- `data/orders.csv` - sample dataset
- `docs/IEEE_paper.md` - IEEE-style paper draft
- `docs/publish_checklist.md` - publishing checklist
- `assets/demo.svg` - placeholder demo image

## Deployment
### Docker
Build and run:
- `docker build -t bizpulse .`
- `docker run -p 8501:8501 bizpulse`

### Docker Compose (App + Postgres)
- `docker compose up -d db`
- Load data: `python scripts/load_csv_to_postgres.py --db-url postgresql+psycopg2://bizpulse:bizpulse@localhost:5432/bizpulse`
- `docker compose up --build app`

### Streamlit Community Cloud
Connect the repo and set:
- `Python version`: 3.11+
- `Main file`: `app.py`
- `Requirements`: `requirements.txt`

## Data schema
The sample file `data/orders.csv` includes:
- `order_id` (string)
- `order_date` (YYYY-MM-DD)
- `quarter` (string, e.g., Q1 2025)
- `product_category` (string)
- `product_name` (string)
- `state_code` (US state code)
- `state_name` (string)
- `revenue` (float)
- `orders` (int)
- `rating` (float, 1-5)

>>>>>>> 23ee62f (BizPulse dashboard)
