import json
import time
from typing import Any

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Crew, Process, Task, LLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResumeIQ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(91, 33, 182, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 100%,
                rgba(76, 29, 149, 0.10),
                transparent 30%
            ),
            #09070d;

        color: #f4f1f8;
    }


    .main .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }


    header {
        background: transparent !important;
    }


    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {
        color: #faf9ff !important;
        font-size: 4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.055em !important;
        line-height: 1 !important;
        margin-bottom: 0.8rem !important;
    }


    h2 {
        color: #f4f1f8 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }


    h3 {
        color: #eeeaf3 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }


    p {
        color: #9a91a5;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-label {
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }


    .hero-description {
        max-width: 680px;
        color: #948b9e;
        font-size: 1rem;
        line-height: 1.75;
        margin-bottom: 2.5rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #100b17;
        border-left: 1px solid rgba(139, 92, 246, 0.12);
    }


    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.3rem;
    }


    .sidebar-brand {
        color: #f5f3f8;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }


    .sidebar-text {
        color: #82798d;
        font-size: 0.82rem;
        line-height: 1.65;
    }


    .sidebar-heading {
        color: #a78bfa;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-top: 1.8rem;
        margin-bottom: 0.7rem;
    }


    /* ========================================================
       SECTION CARDS
       ======================================================== */

    .input-card {
        background: rgba(18, 13, 27, 0.82);
        border: 1px solid rgba(139, 92, 246, 0.14);
        border-radius: 18px;
        padding: 1.25rem;
        margin-bottom: 0.8rem;
    }


    .input-card-title {
        color: #f0edf5;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }


    .input-card-description {
        color: #82798d;
        font-size: 0.82rem;
        line-height: 1.5;
    }


    /* ========================================================
       TEXT AREAS
       ======================================================== */

    .stTextArea textarea {
        background: #0d0913 !important;
        color: #eeeaf4 !important;

        border: 1px solid rgba(139, 92, 246, 0.16) !important;

        border-radius: 12px !important;

        font-size: 0.9rem !important;

        line-height: 1.6 !important;

        transition: all 0.2s ease;
    }


    .stTextArea textarea:hover {
        border-color: rgba(139, 92, 246, 0.35) !important;
    }


    .stTextArea textarea:focus {
        border-color: #7c3aed !important;

        box-shadow:
            0 0 0 1px rgba(124, 58, 237, 0.45) !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: #0d0913;
        border: 1px dashed rgba(139, 92, 246, 0.28);
        border-radius: 12px;
        padding: 0.45rem;
    }


    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;

        background: #7c3aed !important;

        color: #ffffff !important;

        border: none !important;

        border-radius: 12px !important;

        min-height: 48px;

        font-size: 0.92rem !important;

        font-weight: 700 !important;

        box-shadow:
            0 10px 30px rgba(124, 58, 237, 0.20);

        transition:
            transform 0.2s ease,
            background 0.2s ease,
            box-shadow 0.2s ease;
    }


    .stButton > button:hover {
        background: #8b5cf6 !important;

        transform: translateY(-1px);

        box-shadow:
            0 14px 35px rgba(124, 58, 237, 0.32);
    }


    /* ========================================================
       RESULT CARDS
       ======================================================== */

    .result-card {
        background: rgba(18, 13, 27, 0.86);

        border: 1px solid rgba(139, 92, 246, 0.13);

        border-radius: 18px;

        padding: 1.3rem;

        margin-bottom: 1rem;
    }


    .result-title {
        color: #f1edf5;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }


    .result-description {
        color: #a098a8;
        font-size: 0.88rem;
        line-height: 1.7;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status {
        display: inline-block;

        background: rgba(124, 58, 237, 0.12);

        border: 1px solid rgba(139, 92, 246, 0.28);

        color: #c4b5fd;

        border-radius: 999px;

        padding: 0.4rem 0.8rem;

        font-size: 0.78rem;

        font-weight: 700;
    }


    /* ========================================================
       LIST ITEMS
       ======================================================== */

    .list-item {
        background: #0e0a15;

        border: 1px solid rgba(139, 92, 246, 0.08);

        border-radius: 10px;

        padding: 0.8rem 0.9rem;

        margin-bottom: 0.55rem;

        color: #aaa1b3;

        font-size: 0.86rem;

        line-height: 1.55;
    }


    /* ========================================================
       KEYWORDS
       ======================================================== */

    .keyword {
        display: inline-block;

        background: rgba(124, 58, 237, 0.11);

        border: 1px solid rgba(139, 92, 246, 0.22);

        color: #c4b5fd;

        border-radius: 7px;

        padding: 0.35rem 0.55rem;

        margin: 0.2rem 0.12rem;

        font-size: 0.75rem;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #100b17 !important;

        border: 1px solid rgba(139, 92, 246, 0.12) !important;

        border-radius: 12px !important;

        margin-bottom: 0.55rem;
    }


    [data-testid="stExpander"] summary {
        color: #ddd7e5 !important;
        font-size: 0.87rem !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-line {
        margin-top: 4rem;

        padding-top: 1.5rem;

        border-top: 1px solid rgba(139, 92, 246, 0.10);

        text-align: center;

        color: #5f5768;

        font-size: 0.74rem;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .main .block-container {
            padding-top: 2rem;
        }

        h1 {
            font-size: 3rem !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "ResumeIQ",
        unsafe_allow_html=False,
    )

    st.markdown(
        """
        <div class="sidebar-text">
            AI-assisted resume analysis for comparing your
            experience with a target job description.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "How it works",
        unsafe_allow_html=False,
    )

    st.markdown(
        """
        <div class="sidebar-text">
            1. Add your resume.<br>
            2. Add the target job description.<br>
            3. Run the analysis.<br>
            4. Review the recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Review principles",
        unsafe_allow_html=False,
    )

    st.markdown(
        """
        <div class="sidebar-text">
            The reviewer only uses information contained
            in the provided resume and job description.
            It does not invent qualifications or experience.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Technology",
        unsafe_allow_html=False,
    )

    st.markdown(
        """
        <div class="sidebar-text">
            CrewAI<br>
            Groq<br>
            GPT-OSS-120B
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    "AI Resume Analysis",
    unsafe_allow_html=False,
)

st.markdown(
    """
    <style>
    .hero-label-streamlit {
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use HTML only for the small label.
st.markdown(
    '<div class="hero-label-streamlit">AI Resume Analysis</div>',
    unsafe_allow_html=True,
)

st.title("ResumeIQ")

st.markdown(
    """
    <div class="hero-description">
        Compare your resume with a target job description and
        get structured, evidence-based recommendations for
        improving your application.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:

    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:
        return ""


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file) -> str:

    try:

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:
                    pages.append(text)

            except Exception:
                continue

        return "\n\n".join(pages).strip()

    except Exception as e:

        raise ValueError(
            f"Could not read the PDF: {e}"
        )


# ============================================================
# PARSE AI RESPONSE
# ============================================================

def parse_agent_output(raw_output: Any) -> dict:

    if raw_output is None:

        raise ValueError(
            "The AI returned no output."
        )

    if hasattr(raw_output, "raw"):

        raw_output = raw_output.raw

    text = str(raw_output).strip()

    # Remove markdown code fences if the model adds them.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip().startswith("```")
        ):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")

        end = text.rfind("}")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    text[start:end + 1]
                )

            except Exception:
                pass

    raise ValueError(
        "The AI returned an invalid response format."
    )


# ============================================================
# CREATE AGENT
# ============================================================

def create_resume_agent(api_key: str):

    llm = LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=api_key,
        temperature=0,
        max_tokens=5000,
    )

    agent = Agent(

        role="Professional Resume Reviewer",

        goal=(
            "Compare a candidate's resume against a target job "
            "description and provide accurate, evidence-based "
            "recommendations."
        ),

        backstory=(
            "You are an experienced resume reviewer. You carefully "
            "compare resume evidence with job requirements. "
            "You never fabricate qualifications, experience, "
            "skills, education, certifications, or achievements."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    return agent


# ============================================================
# ANALYZE RESUME
# ============================================================

def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
):

    agent = create_resume_agent(api_key)

    task_description = f"""
Review the resume below against the target job description.

RESUME:
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}

RULES:

1. Never fabricate information.
2. Only use evidence found in the resume.
3. If a requirement is not demonstrated, say:
   "Not demonstrated in the provided resume."
4. Do not make hiring decisions.
5. Do not infer sensitive personal characteristics.
6. Provide practical and specific recommendations.
7. Focus on the relationship between the resume and job.
8. Return ONLY valid JSON.

Use this exact JSON structure:

{{
    "match_level": "Strong Match | Moderate Match | Needs Improvement",

    "summary": "Short overall comparison.",

    "strengths": [
        "Strength 1",
        "Strength 2"
    ],

    "requirement_analysis": [
        {{
            "requirement": "Job requirement",
            "status": "Demonstrated | Partially Demonstrated | Not Demonstrated",
            "evidence": "Evidence from resume"
        }}
    ],

    "missing_or_unclear": [
        "Missing or unclear item"
    ],

    "resume_improvements": [
        "Specific improvement"
    ],

    "ats_keywords": [
        "keyword 1",
        "keyword 2"
    ],

    "priority_actions": [
        "Highest priority action",
        "Second priority action",
        "Third priority action"
    ],

    "disclaimer": "This review compares the provided resume with the provided job description and does not guarantee hiring outcomes."
}}
"""

    task = Task(
        description=task_description,

        expected_output=(
            "Valid JSON matching the requested structure."
        ),

        agent=agent,
    )

    crew = Crew(
        agents=[agent],

        tasks=[task],

        process=Process.sequential,

        verbose=False,
    )

    last_error = None

    for attempt in range(3):

        try:

            result = crew.kickoff()

            return parse_agent_output(result)

        except Exception as e:

            last_error = e

            if attempt < 2:

                time.sleep(
                    2 ** attempt
                )

    raise last_error


# ============================================================
# INPUT SECTION
# ============================================================

left_column, right_column = st.columns(
    2,
    gap="large",
)


# ============================================================
# RESUME INPUT
# ============================================================

with left_column:

    st.subheader("Resume")

    st.caption(
        "Paste your resume or upload a PDF."
    )

    resume_text = st.text_area(
        "Resume content",

        height=310,

        placeholder=(
            "Paste your resume here..."
        ),

        label_visibility="collapsed",
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )


# ============================================================
# JOB INPUT
# ============================================================

with right_column:

    st.subheader("Target Job")

    st.caption(
        "Paste the job description you want to compare against."
    )

    job_description = st.text_area(
        "Target job description",

        height=310,

        placeholder=(
            "Paste the complete job description here..."
        ),

        label_visibility="collapsed",
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

button_left, button_center, button_right = st.columns(
    [1, 2, 1]
)

with button_center:

    analyze_clicked = st.button(
        "Analyze Resume",
        use_container_width=True,
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_clicked:

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key = get_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. Add it in "
            "Streamlit Cloud → Settings → Secrets."
        )

        st.stop()


    # --------------------------------------------------------
    # GET RESUME
    # --------------------------------------------------------

    final_resume = resume_text.strip()


    if uploaded_file is not None:

        try:

            pdf_text = extract_pdf_text(
                uploaded_file
            )

            if pdf_text:

                final_resume = pdf_text

            else:

                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a text-based PDF or paste "
                    "your resume manually."
                )

                st.stop()

        except Exception as e:

            st.error(str(e))

            st.stop()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not final_resume:

        st.warning(
            "Please provide your resume."
        )

        st.stop()


    if not job_description.strip():

        st.warning(
            "Please provide the target job description."
        )

        st.stop()


    if len(final_resume) < 100:

        st.warning(
            "Your resume appears to be too short. "
            "Please provide the complete resume."
        )

        st.stop()


    if len(job_description) < 100:

        st.warning(
            "The job description appears to be too short. "
            "Please provide the complete job description."
        )

        st.stop()


    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing your resume..."
    ):

        try:

            result = analyze_resume(
                final_resume,
                job_description,
                api_key,
            )

        except Exception as e:

            error_text = str(e).lower()


            if (
                "429" in error_text
                or "rate limit" in error_text
            ):

                st.error(
                    "Groq rate limit reached. "
                    "Please wait a moment and try again."
                )


            elif (
                "401" in error_text
                or "authentication" in error_text
            ):

                st.error(
                    "Groq authentication failed. "
                    "Check your GROQ_API_KEY in Streamlit Secrets."
                )


            elif "403" in error_text:

                st.error(
                    "Groq rejected the request. "
                    "Check your API key and model access."
                )


            else:

                st.error(
                    f"Analysis failed:\n\n{e}"
                )

            st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("Resume Review")


    # --------------------------------------------------------
    # MATCH LEVEL
    # --------------------------------------------------------

    match_level = result.get(
        "match_level",
        "Not available",
    )

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-title">
                Overall Alignment
            </div>

            <span class="status">
                {match_level}
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = result.get(
        "summary",
        "No summary was returned.",
    )

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-title">
                Summary
            </div>

            <div class="result-description">
                {summary}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # STRENGTHS + PRIORITY ACTIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2,
        gap="large",
    )


    with col1:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-title">
                    Strengths
                </div>

            """,
            unsafe_allow_html=True,
        )

        strengths = result.get(
            "strengths",
            [],
        )

        if strengths:

            for item in strengths:

                st.markdown(
                    f"""
                    <div class="list-item">
                        {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(
                "No strengths identified."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-title">
                    Priority Actions
                </div>

            """,
            unsafe_allow_html=True,
        )

        actions = result.get(
            "priority_actions",
            [],
        )

        if actions:

            for item in actions:

                st.markdown(
                    f"""
                    <div class="list-item">
                        {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(
                "No priority actions identified."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # REQUIREMENT ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "Requirement Analysis"
    )

    st.caption(
        "How the resume demonstrates the requirements of the target role."
    )


    requirements = result.get(
        "requirement_analysis",
        [],
    )


    if requirements:

        for item in requirements:

            requirement = item.get(
                "requirement",
                "Requirement",
            )

            status = item.get(
                "status",
                "Unknown",
            )

            evidence = item.get(
                "evidence",
                "No evidence provided.",
            )

            with st.expander(
                f"{requirement} — {status}"
            ):

                st.write(evidence)

    else:

        st.info(
            "No requirement analysis was returned."
        )


    # --------------------------------------------------------
    # MISSING / UNCLEAR
    # --------------------------------------------------------

    missing = result.get(
        "missing_or_unclear",
        [],
    )


    if missing:

        st.subheader(
            "Missing or Unclear"
        )

        for item in missing:

            st.markdown(
                f"""
                <div class="list-item">
                    {item}
                </div>
                """,
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # RESUME IMPROVEMENTS
    # --------------------------------------------------------

    improvements = result.get(
        "resume_improvements",
        [],
    )


    if improvements:

        st.subheader(
            "Resume Improvements"
        )

        for item in improvements:

            st.markdown(
                f"""
                <div class="list-item">
                    {item}
                </div>
                """,
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # ATS KEYWORDS
    # --------------------------------------------------------

    keywords = result.get(
        "ats_keywords",
        [],
    )


    if keywords:

        st.subheader(
            "Relevant ATS Keywords"
        )

        keyword_html = ""

        for keyword in keywords:

            keyword_html += (
                f'<span class="keyword">{keyword}</span>'
            )

        st.markdown(
            keyword_html,
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    disclaimer = result.get(
        "disclaimer",
        (
            "This review compares the provided resume "
            "with the provided job description and does "
            "not guarantee hiring outcomes."
        ),
    )

    st.info(disclaimer)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<div class='footer-line'>ResumeIQ · AI-assisted resume analysis</div>",
    unsafe_allow_html=True,
)
