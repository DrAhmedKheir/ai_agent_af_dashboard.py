# Deploy Guide

## Streamlit Cloud (fastest)
1. Push these files to GitHub (keep `ai_agent_af_dashboard.py` at repo root).
2. Go to https://share.streamlit.io and connect the repo.
3. Set main file path to `ai_agent_af_dashboard.py` (or rename to `streamlit_app.py`).
4. Add secrets (optional):
   - `OPENAI_API_KEY` for Agent chat.

## Docker
```bash
docker build -t af-agent-dashboard .
docker run -p 8501:8501 --env PORT=8501 af-agent-dashboard
# open http://localhost:8501
```

## Heroku
```bash
heroku create af-agent-dashboard
heroku buildpacks:add heroku/python
git add . && git commit -m "deploy"
git push heroku main
# Or use container registry:
heroku container:push web -a af-agent-dashboard
heroku container:release web -a af-agent-dashboard
```

## GitHub Actions (optional, to build & push Docker image)
Create a repo secret `GHCR_PAT` (Personal Access Token with `packages:write`). Then use the workflow below.
