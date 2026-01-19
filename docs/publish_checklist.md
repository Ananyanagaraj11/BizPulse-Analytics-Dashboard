# Publish Checklist

## IEEE-style submission (class project or conference)
- Pick a venue and review page limits, formatting, and deadlines.
- Convert `docs/IEEE_paper.md` to IEEE template (Word or LaTeX).
- Add figures from the dashboard (export charts to PNG/SVG).
- Include methodology and any dataset limitations or ethics notes.
- Add a short related work section if required by the venue.
- Ensure citations are complete and formatted per IEEE style.
- Run spellcheck and verify all links.

## Repository readiness
- Add a short demo GIF or screenshot in `README.md`.
- Provide a sample dataset and data dictionary.
- Add a `LICENSE` file (MIT or Apache-2.0 is common for demos).
- Include a `CONTRIBUTING.md` if the project is collaborative.

## Deployment options
- Streamlit Community Cloud: `streamlit run app.py` with `requirements.txt`.
- Docker (optional): add a `Dockerfile` and expose port 8501.
- On-prem: configure a reverse proxy (Nginx) for authentication.

