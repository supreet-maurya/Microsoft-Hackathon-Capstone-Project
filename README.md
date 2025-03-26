# Microsoft-Hackathon-Capstone-Project
# 🚀 AI Resume Evaluator Pro

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://resume-evaluator-pro.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An AI-powered resume evaluation system that analyzes multiple resumes against job descriptions using Azure OpenAI's advanced NLP capabilities.
## 🌟 Features

- 📄 Multi-PDF resume processing
- 🔍 AI-powered scoring system (0-100 scale)
- 📊 Interactive comparison dashboard
- 🎯 Job description relevance analysis
- 📈 Visual score breakdowns with Matplotlib
- ⚡ Real-time Azure OpenAI integration

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- [Poppler-utils](https://poppler.freedesktop.org/) (PDF processing)
  ```bash
  # Ubuntu/Debian
  sudo apt-get install poppler-utils
  
  # MacOS
  brew install poppler


Install dependencies:
  pip install -r requirements.txt

⚙️ Configuration
Create .env file:
  ENDPOINT_URL="your-azure-endpoint-url"
  DEPLOYMENT_NAME="gpt-4"
  AZURE_OPENAI_API_KEY="your-api-key"

🚦 Usage
Start the application:
  streamlit run app.py
