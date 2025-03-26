from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import pdf2image
from openai import AzureOpenAI 
from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
# Note: Default values are provided for local testing but should be removed in production
endpoint = os.getenv("ENDPOINT_URL", "xxxxxxxxxxxxx")  #write your ENDPOINT_URL here 
deployment = os.getenv("DEPLOYMENT_NAME", "xxxxxxxxxxxxxxxx")  #write your DEPLOYMENT_NAME here 
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "xxxxxxxxxxxxxxxxxxx") #write your key here 

# Initialize Azure OpenAI client
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",
)

def extract_resume_score(response_text):
    """
    Extracts the composite score from the evaluation response text.
    
    Args:
        response_text (str): The evaluation response text from the AI model
        
    Returns:
        int: The extracted score (0-100) or 0 if not found
    """
    try:
        match = re.search(r"Composite Score:\s*(\d+)/100", response_text)
        return int(match.group(1)) if match else 0
    except Exception as e:
        st.error(f"Score extraction error: {str(e)}")
        return 0

def parse_evaluation_scores(evaluation_text):
    """
    Parses the evaluation text to extract scores for different categories.
    
    Args:
        evaluation_text (str): The evaluation text containing scores
        
    Returns:
        dict: A structured dictionary containing all scores and subscores
    """
    scores = {
        'Relevance': {'total': 0, 'subscores': {}},
        'Content': {'total': 0, 'subscores': {}},
        'Presentation': 0,
        'Bonus': 0
    }
    
    # Patterns for main score categories
    patterns = {
        'Relevance': r"- Relevance: (\d+)/40",
        'Content': r"- Content: (\d+)/30",
        'Presentation': r"- Presentation: (\d+)/20",
        'Bonus': r"- Bonus: (\d+)/10"
    }
    
    # Patterns for subcategories within Relevance and Content
    sub_patterns = {
        'Relevance': {
            'Skills': r"- Skills: (\d+)/20",
            'Experience': r"- Experience: (\d+)/15",
            'Education': r"- Education: (\d+)/5"
        },
        'Content': {
            'Achievements': r"- Achievements: (\d+)/15",
            'Narrative': r"- Narrative: (\d+)/10",
            'Errors': r"- Errors: (\d+)/5"
        }
    }

    # Extract main category scores
    for category, pattern in patterns.items():
        match = re.search(pattern, evaluation_text)
        if match:
            if category in ['Relevance', 'Content']:
                scores[category]['total'] = int(match.group(1))
            else:
                scores[category] = int(match.group(1))

    # Extract subcategory scores
    for category, subs in sub_patterns.items():
        for sub, pattern in subs.items():
            match = re.search(pattern, evaluation_text)
            if match:
                scores[category]['subscores'][sub] = int(match.group(1))
    
    return scores

def create_pie_chart(scores, category):
    """
    Creates a pie chart visualization for score subcategories.
    
    Args:
        scores (dict): The scores dictionary containing subscores
        category (str): The category name (e.g., 'Relevance', 'Content')
        
    Returns:
        matplotlib.figure.Figure: The generated pie chart figure
    """
    if not scores.get('subscores'):
        return None
    
    labels = list(scores['subscores'].keys())
    values = list(scores['subscores'].values())
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, autopct='%1.1f%%',
           startangle=90, colors=plt.cm.Pastel1.colors, 
           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
    ax.set_title(f"{category} Breakdown", fontsize=10, pad=5)
    plt.tight_layout(pad=0.5)
    return fig


def create_score_barchart(ranked_resumes):
    """
    Creates a horizontal bar chart comparing resume scores.
    
    Args:
        ranked_resumes (list): List of tuples containing (resume_name, score)
        
    Returns:
        matplotlib.figure.Figure: The generated bar chart figure
    """
    names = [name for name, _ in ranked_resumes]
    scores = [score for _, score in ranked_resumes]
    
    # Increase figure size for better visibility
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, scores, 
                 color='#1f77b4', 
                 height=0.8)  
    ax.bar_label(bars, padding=5, fontsize=10) 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)  
    ax.invert_yaxis()  # Highest score at top
    ax.set_xlim(0, 100)
    ax.set_xlabel('Score (out of 100)', fontsize=12)
    ax.set_title('Resume Scores Comparison', fontsize=14, pad=15)
    ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout(pad=1.0) 
    return fig


def evaluate_resume(resume_text, job_description, resume_name=""):
    """
    Evaluates a resume against a job description using Azure OpenAI.
    
    Args:
        resume_text (str): The text content of the resume
        job_description (str): The job description text
        resume_name (str): Optional name for the resume
        
    Returns:
        tuple: (evaluation_text, score) where score is 0-100
    """
    if not resume_text.strip():
        return "ERROR: Empty resume content", 0
    if not job_description.strip():
        return "ERROR: Empty job description", 0
    
    # Detailed prompt template for the AI evaluation
    prompt = """
    Role: You are an AI-powered Resume Evaluation Expert with dual capabilities:
    1. Technical Resume Scorer - Assess overall resume quality and relevance.
    2. HR Advisor - Provide actionable feedback for improvement.

    Evaluation Protocol:
    1. Perform comprehensive analysis of:
       - Skills alignment with job requirements
       - Experience duration and relevance
       - Demonstrated achievements/impact
       - Document structure and clarity
    2. Apply strict scoring rubric (100-point scale):

    Scoring Breakdown:
    - Relevance to Job Description (40 points)
      - Technical skills match (20)
      - Experience alignment (15)
      - Education/certifications (5)
    - Content Quality (30 points)
      - Achievement quantification (15)
      - Professional narrative (10)
      - Error-free writing (5)
    - Presentation (20 points)
      - Logical structure (10)
      - Visual organization (5)
      - Conciseness (5)
    - Bonus: Notable Differentiators (10 points)

    Required Output Format:
    ---
    ### Resume Evaluation Report: {resume_name}

    #### 📊 Composite Score: {score}/100
    Detailed Breakdown:
    - Relevance: {relevance_score}/40
      - Skills: {skills_score}/20
      - Experience: {exp_score}/15
      - Education: {edu_score}/5
    - Content: {content_score}/30
      - Achievements: {ach_score}/15
      - Narrative: {narrative_score}/10
      - Errors: {error_score}/5
    - Presentation: {presentation_score}/20
    - Bonus: {bonus_score}/10

    #### 🎯 Top Alignment Indicators
    - [✓] 3-5 most relevant JD matches
    - [✗] Critical missing requirements

    #### ✨ Key Strengths
    - [Bullet points of standout elements with metrics]

    #### ⚠ Improvement Areas
    - [Specific gaps with remediation advice]

    #### 💼 HR Assessment
    [Concise professional opinion on fit and potential]

    ---
    """

    try:
        # Call Azure OpenAI API for evaluation
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a meticulous resume analyst. Only use provided facts."},
                {"role": "user", "content": f"Instructions:{prompt} JD: {job_description}\nResume: {resume_text}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        evaluation = response.choices[0].message.content
        score = extract_resume_score(evaluation)
        
        if score == 0:
            st.warning(f"Potential evaluation error for {resume_name}")
            st.text_area("Debug - Full Response", evaluation, height=200)
        
        return evaluation, score
        
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return f"Evaluation failed for {resume_name}", 0
    
def input_pdf_setup(uploaded_file):
    """
    Extracts text from an uploaded PDF file.
    
    Args:
        uploaded_file (UploadedFile): Streamlit file upload object
        
    Returns:
        str: Extracted text from all pages of the PDF
        
    Raises:
        FileNotFoundError: If no file is uploaded
    """
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    raise FileNotFoundError("No file uploaded")

def generate_comparative_analysis(evaluations):
    """
    Generates a comparative analysis of multiple resume evaluations.
    
    Args:
        evaluations (list): List of evaluation texts
        
    Returns:
        str: Comparative analysis generated by the AI
    """
    analysis_prompt = f"""
    Analyze these resume evaluations and provide:
    1. Top 3 strongest candidates overall
    2. Most common strengths across candidates
    3. Most frequent weaknesses
    4. Key differentiators between top candidates
    
    Evaluations: {evaluations}
    """
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# Streamlit App Configuration
st.set_page_config(page_title="ATS Resume Expert Pro", layout="wide")
st.header("📊 Multi-Resume Evaluation System")

# Main UI Components
input_text = st.text_area("Job Description:", key="input", height=150)
uploaded_files = st.file_uploader("Upload resumes (PDF)...", 
                                 type=["pdf"], 
                                 accept_multiple_files=True)

# Evaluation Process
if st.button("Evaluate Resumes") and uploaded_files:
    evaluations = []
    scores = []
    resume_names = []
    
    with st.spinner("Evaluating resumes..."):
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                resume_name = uploaded_file.name[:-4]  # Remove .pdf extension
                resume_text = input_pdf_setup(uploaded_file)
                evaluation, score = evaluate_resume(resume_text, input_text, resume_name)
                
                evaluations.append(evaluation)
                scores.append(score)
                resume_names.append(resume_name)
                
                # Display individual evaluation results
                with st.expander(f"{resume_name} (Score: {score}/100)"):
                    col1, col2 = st.columns([2, 1], gap="small")
                    
                    with col1:
                        st.markdown(evaluation)
                    
                    with col2:
                        scores_data = parse_evaluation_scores(evaluation)
                        
                        # Display pie charts for subcategories
                        if scores_data['Relevance']['subscores']:
                            fig = create_pie_chart(scores_data['Relevance'], 'Relevance')
                            st.pyplot(fig)
                        
                        if scores_data['Content']['subscores']:
                            fig = create_pie_chart(scores_data['Content'], 'Content')
                            st.pyplot(fig)
                        
                        st.markdown(f"""
                        **Presentation**: {scores_data['Presentation']}/20  
                        **Bonus**: {scores_data['Bonus']}/10
                        """)
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Update progress bar
            progress = int((i + 1) / len(uploaded_files) * 100)
            progress_bar.progress(progress)
        
        # Comparative analysis for multiple resumes
        if len(evaluations) > 1:
            st.divider()
            st.subheader("🏆 Comparative Analysis")
            
            # Sort resumes by score (highest first)
            ranked_resumes = sorted(zip(resume_names, scores), 
                                  key=lambda x: x[1], 
                                  reverse=True)
            
            col_analysis, col_chart = st.columns([2, 1], gap="medium")
            
            with col_analysis:
                st.markdown("### 📋 Resume Ranking")
                for rank, (name, score) in enumerate(ranked_resumes, 1):
                    st.metric(f"{rank}. {name}", f"{score}/100")
                
                # Generate and display comparative analysis
                analysis = generate_comparative_analysis(evaluations)
                st.markdown("### 🔍 Overall Insights")
                st.markdown(analysis)
            
            with col_chart:
                st.markdown("### 📈 Score Comparison")
                fig = create_score_barchart(ranked_resumes)
                st.pyplot(fig)
        
        progress_bar.empty()
        st.success("Evaluation complete!")
