# BizPulse Analytics Dashboard

[![Live Demo](https://img.shields.io/badge/Live-Demo-2ea44f?style=for-the-badge)](https://bizpulse-analytics-dashboard-5c8cvwu7drn9yymrjydmd4.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-111827?style=for-the-badge&logo=github)](https://github.com/Ananyanagaraj11/BizPulse-Analytics-Dashboard)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ananyanagaraj/)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

A dark-mode, KPI-driven analytics dashboard built with Streamlit and Plotly.
Designed to look like a modern SaaS KPI board with filters, charts, and
actionable tables.

## Links
- Live Demo: https://bizpulse-analytics-dashboard-5c8cvwu7drn9yymrjydmd4.streamlit.app/
- GitHub: https://github.com/Ananyanagaraj11/BizPulse-Analytics-Dashboard
- LinkedIn: https://www.linkedin.com/in/ananyanagaraj/
- License: MIT (see `LICENSE`)

## Screenshots
![Dashboard screenshot 1](assets/Screenshot%202026-01-19%20175801.png)
![Dashboard screenshot 2](assets/Screenshot%202026-01-19%20182350.png)
![Dashboard screenshot 3](assets/Screenshot%202026-01-19%20182401.png)
![Dashboard screenshot 4](assets/Screenshot%202026-01-19%20182413.png)

## Sample exports
- [export.csv](assets/export.csv)
- [2026-01-19T23-24_export.csv](assets/2026-01-19T23-24_export.csv)

## Features
- KPI cards, charts, maps, and a priority action list
- Treemap + scatter + risk bars + ranking mini chart
- Responsive grid layout with dark UI styling
- CSV or database-driven data sources

## Tech
- Python
- Pandas
- Streamlit
- Plotly

## Quick Start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the app:
   - `streamlit run app.py`

## Data Sources
### CSV
- `CSV_PATH` (default: `data/orders.csv`)

### Database
- `DATABASE_URL` (e.g., `sqlite:///data/orders.db` or a Postgres URL)
- `DB_TABLE` (default: `orders`)
- `SQL_QUERY` (optional, overrides `DB_TABLE`)

Example with SQLite:
- `python scripts/load_csv_to_sqlite.py`
- `set DATABASE_URL=sqlite:///data/orders.db` (PowerShell: `$env:DATABASE_URL="sqlite:///data/orders.db"`)
- `streamlit run app.py`

Example with Postgres:
- `set DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/bizpulse`
- `python scripts/load_csv_to_postgres.py --db-url %DATABASE_URL%`
- `streamlit run app.py`

## Project Structure
- `app.py` - Streamlit app
- `data/orders.csv` - sample dataset
- `docs/IEEE_paper.md` - IEEE-style draft
- `docs/publish_checklist.md` - publishing checklist

## Deployment
### Streamlit Community Cloud
Connect the repo and set:
- `Python version`: 3.11+
- `Main file`: `app.py`
- `Requirements`: `requirements.txt`

### Docker
- `docker build -t bizpulse .`
- `docker run -p 8501:8501 bizpulse`

### Docker Compose (App + Postgres)
- `docker compose up -d db`
- Load data: `python scripts/load_csv_to_postgres.py --db-url postgresql+psycopg2://bizpulse:bizpulse@localhost:5432/bizpulse`
- `docker compose up --build app`

## Data Schema
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
