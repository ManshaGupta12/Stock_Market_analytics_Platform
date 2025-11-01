Slide 1: Problem Statement
- Need: scalable analytics for hundreds of NSE stocks, real-time dashboards, explainable predictions.

Slide 2: Architecture
- PySpark ETL -> Parquet Data Lake -> Grafana -> Gradio + FastAPI -> Cohere for LLM explanations

Slide 3: Demo Flow
- Show ETL run -> show Grafana dashboard -> show Gradio prediction with explanation.

Slide 4: Limitations & Next steps
- Data latency, backtesting, stronger ML models, production data feeds, authentication & compliance.

After everything starts:

Open http://localhost:3000
 → Grafana dashboard

Default login:
Username: admin
Password: admin

Open http://localhost:8000
 → API
(you should see a JSON response or docs page if FastAPI is used)

Open http://localhost:7860
 → Gradio interface (your UI)
