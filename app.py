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
# THEME / STYLING
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
                circle at 20% 0%,
                rgba(91, 33, 182, 0.16),
                transparent 30%
            ),
            #0a0710;

        color: #f5f3fa;
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
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #100b18;
        border-left: 1px solid rgba(139, 92, 246, 0.15);
        border-right: none;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.3rem;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f5f3fa;
        margin-bottom: 0.4rem;
    }

    .sidebar-description {
        color: #8f879d;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    .sidebar-section {
        color: #c4b5fd;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
    }

    .sidebar-item {
        color: #aaa2b7;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-bottom: 0.7rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        margin-bottom: 3rem;
    }

    .hero-label {
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        color: #faf9ff;
        font-size: clamp(2.8rem, 6vw, 4.5rem);
        font-weight: 800;
        letter-spacing: -0.055em;
        line-height: 0.98;
        margin: 0;
    }

    .hero-title span {
        color: #a78bfa;
    }

    .hero-description {
        max-width: 700px;
        margin-top: 1.2rem;
        color: #958ca3;
        font-size: 1rem;
        line-height: 1.75;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-heading {
        color: #f3f0f8;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .section-subheading {
        color: #81798d;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-bottom: 1.1rem;
    }


    /* ========================================================
       INPUT CARDS
       ======================================================== */

    .input-card {
        background: rgba(18, 13, 27, 0.88);
        border: 1px solid rgba(139, 92, 246, 0.14);
        border-radius: 18px;
        padding: 1.35rem;
        height: 100%;
    }


    /* ========================================================
       TEXT AREAS
       ======================================================== */

    .stTextArea textarea {
        background: #0d0914 !important;
        color: #eeeaf5 !important;

        border: 1px solid rgba(139, 92, 246, 0.18) !important;

        border-radius: 12px !important;

        font-size: 0.9rem !important;

        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #7c3aed !important;

        box-shadow:
            0 0 0 1px rgba(124, 58, 237, 0.45) !important;
    }

    .stTextArea label {
        color: #a39aaa !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: #0d0914;
        border: 1px dashed rgba(139, 92, 246, 0.28);
        border-radius: 12px;
        padding: 0.6rem;
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

        transition:
            background 0.2s ease,
            transform 0.2s ease,
            box-shadow 0.2s ease;

        box-shadow:
            0 10px 30px rgba(124, 58, 237, 0.22);
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
        background: rgba(18, 13, 27, 0.9);

        border: 1px solid rgba(139, 92, 246, 0.14);

        border-radius: 18px;

        padding: 1.35rem;

        margin-bottom: 1rem;
    }

    .result-heading {
        color: #f4f1f8;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .result-text {
        color: #aaa1b4;
        font-size: 0.9rem;
        line-height: 1.7;
    }


    /* ========================================================
       STATUS BADGE
       ======================================================== */

    .status-badge {
        display: inline-block;

        padding: 0.4rem 0.8rem;

        border-radius: 999px;

        background: rgba(124, 58, 237, 0.12);

        border: 1px solid rgba(139, 92, 246, 0.3);

        color: #c4b5fd;

        font-size: 0.78rem;

        font-weight: 700;
    }


    /* ========================================================
       LIST ITEMS
       ======================================================== */

    .list-item {
        background: #0e0a15;

        border: 1px solid rgba(139, 92, 246, 0.09);

        border-radius: 10px;

        padding: 0.8rem 0.9rem;

        margin-bottom: 0.55rem;

        color: #b5adbf;

        font-size: 0.87rem;

        line-height: 1.55;
    }


    /* ========================================================
       KEYWORD TAG
       ======================================================== */

    .keyword {
        display: inline-block;

        background: rgba(124, 58, 237, 0.12);

        border: 1px solid rgba(139, 92, 246, 0.2);

        color: #c4b5fd;

        border-radius: 7px;

        padding: 0.35rem 0.55rem;

        margin: 0.2rem 0.15rem;

        font-size: 0.76rem;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #100b17 !important;

        border: 1px solid rgba(139, 92, 246, 0.12) !important;

        border-radius: 12px !important;

        margin-bottom: 0.5rem;
    }

    [data-testid="stExpander"] summary {
        color: #ddd7e5 !important;
        font-size: 0.88rem !important;
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

    .app-footer {
        border-top: 1px solid rgba(139, 92, 246, 0.1);

        margin-top: 4rem;

        padding-top: 1.5rem;

        text-align: center;

        color: #625a6d;

        font-size: 0.75rem;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .main .block-container {
            padding-top: 2rem;
        }

        .hero-title {
            font-size: 3rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# RIGHT SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            ResumeIQ
        </div>

        <div class="sidebar-description">
            AI-assisted resume analysis designed to help you
            understand how closely your resume aligns with a
            target role.
        </div>

        <div class="sidebar-section">
            How it works
        </div>

        <div class="sidebar-item">
            1. Add your resume
        </div>

        <div class="sidebar-item">
            2. Add the target job description
        </div>

        <div class="sidebar-item">
            3. Run the analysis
        </div>

        <div class="sidebar-item">
            4. Review the recommendations
        </div>

        <div class="sidebar-section">
            Review principles
        </div>

        <div class="sidebar-item">
            Resume claims are evaluated only from the
            information you provide.
        </div>

        <div class="sidebar-item">
            The system does not invent qualifications,
            experience, or skills.
        </div>

        <div class="sidebar-item">
            Recommendations are informational and do not
            guarantee hiring outcomes.
        </div>

        <div class="sidebar-section">
            Technology
        </div>

        <div class="sidebar-item">
            CrewAI
        </div>

        <div class="sidebar-item">
            Groq
        </div>

        <div class="sidebar-item">
            GPT-OSS-120B
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-label">
            AI Resume Analysis
        </div>

        <div class="hero-title">
            Resume<span>IQ</span>
        </div>

        <div class="hero-description">
            Compare your resume with a target job description and
            get structured, evidence-based recommendations for
            improving your application.
        </div>

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
# JSON PARSER
# ============================================================

def parse_agent_output(raw_output: Any) -> dict:

    if raw_output is None:
        raise ValueError(
            "The AI returned no output."
        )

    if hasattr(raw_output, "raw"):
        raw_output = raw_output.raw

    text = str(raw_output).strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
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
            "Compare a candidate's resume with a target job "
            "description and provide accurate, evidence-based "
            "recommendations."
        ),

        backstory=(
            "You are an experienced professional resume reviewer. "
            "You carefully compare resume evidence against job "
            "requirements. You never fabricate qualifications, "
            "skills, experience, education, certifications, or "
            "achievements."
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
Review the following resume against the target job description.

RESUME:
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}

Follow these rules:

1. Never fabricate information.
2. Only use evidence present in the resume.
3. If a qualification or requirement is not demonstrated,
   write exactly:
   "Not demonstrated in the provided resume."
4. Do not make hiring decisions.
5. Do not infer sensitive personal characteristics.
6. Provide practical resume recommendations.
7. Focus on the relationship between the resume and job description.
8. Return ONLY valid JSON.

Return this exact structure:

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
            "evidence": "Resume evidence"
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
                time.sleep(2 ** attempt)

    raise last_error


# ============================================================
# INPUT AREA
# ============================================================

left_column, right_column = st.columns(
    2,
    gap="large",
)


# ------------------------------------------------------------
# RESUME
# ------------------------------------------------------------

with left_column:

    st.markdown(
        """
        <div class="input-card">

            <div class="section-heading">
                Resume
            </div>

            <div class="section-subheading">
                Paste your resume or upload a PDF.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
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


# ------------------------------------------------------------
# JOB DESCRIPTION
# ------------------------------------------------------------

with right_column:

    st.markdown(
        """
        <div class="input-card">

            <div class="section-heading">
                Target Job
            </div>

            <div class="section-subheading">
                Paste the job description you want to compare against.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
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
# ACTION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

_, center, _ = st.columns(
    [1, 2, 1]
)

with center:

    analyze_clicked = st.button(
        "Analyze Resume",
        use_container_width=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    api_key = get_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. Add it under "
            "Streamlit Cloud → Settings → Secrets."
        )

        st.stop()


    # --------------------------------------------------------
    # RESUME SOURCE
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
    # RUN AI
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
                    "The Groq rate limit was reached. "
                    "Please wait a moment and try again."
                )


            elif (
                "401" in error_text
                or "authentication" in error_text
            ):

                st.error(
                    "Groq authentication failed. "
                    "Check your GROQ_API_KEY."
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

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">

            <div class="hero-label">
                Analysis Complete
            </div>

            <div class="hero-title" style="font-size: 2.4rem;">
                Resume Review
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


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

            <div class="result-heading">
                Overall alignment
            </div>

            <span class="status-badge">
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

            <div class="result-heading">
                Summary
            </div>

            <div class="result-text">
                {summary}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # STRENGTHS + PRIORITIES
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2,
        gap="large",
    )


    with col1:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-heading">
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

                <div class="result-heading">
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
    # REQUIREMENTS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="result-card">

            <div class="result-heading">
                Requirement Analysis
            </div>

            <div class="result-text">
                Review how the resume demonstrates the requirements
                of the target role.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
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

        st.markdown(
            """
            <div class="result-card">

                <div class="result-heading">
                    Missing or Unclear
                </div>

            """,
            unsafe_allow_html=True,
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

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    improvements = result.get(
        "resume_improvements",
        [],
    )


    if improvements:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-heading">
                    Resume Improvements
                </div>

            """,
            unsafe_allow_html=True,
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

        st.markdown(
            "</div>",
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

        st.markdown(
            """
            <div class="result-card">

                <div class="result-heading">
                    Relevant ATS Keywords
                </div>

            """,
            unsafe_allow_html=True,
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

        st.markdown(
            "</div>",
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
    """
    <div class="app-footer">
        ResumeIQ · AI-assisted resume analysis
    </div>
    """,
    unsafe_allow_html=True,
)
