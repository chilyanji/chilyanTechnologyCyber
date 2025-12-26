# Streamlit Web Application Setup

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run 05_app.py
```

The app will open at `http://localhost:8501`

## Features

### 1. URL Checker
- Real-time phishing detection
- Feature extraction visualization
- Risk assessment gauge
- Google Safe Browsing integration

### 2. Detection History
- Track all analyzed URLs
- View classification statistics
- Phishing detection rate
- URL categorization pie chart

### 3. About & Documentation
- Project information
- Technology stack explanation
- Model performance metrics
- Feature explanations

## Deployment Options

### Option 1: Streamlit Cloud (Recommended for Demo)
```bash
git push to GitHub repository
Go to https://streamlit.io/cloud
Connect GitHub repo and select this project
```

### Option 2: Google Cloud Run
```bash
gcloud run deploy phishing-detector \
  --source . \
  --platform managed \
  --region us-central1
```

### Option 3: Heroku
```bash
heroku create phishing-detector
git push heroku main
```

## Environment Variables
Create `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
DEBUG=False
```

## Troubleshooting

**Models not loading?**
- Ensure models/ folder exists with trained models
- Run 03_ml_model_training.py first

**Port already in use?**
```bash
streamlit run 05_app.py --server.port 8502
```

**Memory issues?**
- Reduce dataset size in 01_dataset_eda.py
- Use model quantization for production
