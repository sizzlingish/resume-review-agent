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
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# THEME / STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(124, 58, 237, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 5%,
                rgba(168, 85, 247, 0.12),
                transparent 25%
            ),
            linear-gradient(
                180deg,
                #090611 0%,
                #0c0815 45%,
                #090611 100%
            );

        color: #f5f3ff;
    }


    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ======================================================
       REMOVE STREAMLIT CHROME
    ====================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ======================================================
       TEXT
    ====================================================== */

    h1, h2, h3, h4 {
        color: #f8f7ff !important;
    }

    p {
        color: #aaa3b8;
    }


    /* ======================================================
       HERO
    ====================================================== */

    .hero-wrapper {
        text-align: center;
        padding: 3rem 1rem 2.5rem;
    }


    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(124, 58, 237, 0.10);

        border: 1px solid rgba(167, 139, 250, 0.25);

        color: #c4b5fd;

        font-size: 0.78rem;
        font-weight: 700;

        letter-spacing: 0.04em;
    }


    .hero-title {
        margin-top: 1rem;

        font-size: clamp(3rem, 7vw, 5.3rem);

        font-weight: 850;

        line-height: 0.95;

        letter-spacing: -0.06em;

        background: linear-gradient(
            135deg,
            #ffffff 20%,
            #ddd6fe 55%,
            #a78bfa 100%
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .hero-subtitle {
        max-width: 650px;

        margin: 1.25rem auto 0;

        font-size: 1rem;

        line-height: 1.7;

        color: #938ba5;
    }


    /* ======================================================
       SECTION LABEL
    ====================================================== */

    .eyebrow {
        font-size: 0.72rem;

        text-transform: uppercase;

        letter-spacing: 0.13em;

        color: #8b5cf6;

        font-weight: 800;

        margin-bottom: 0.45rem;
    }


    .section-heading {
        font-size: 1.15rem;

        font-weight: 750;

        color: #f5f3ff;

        margin-bottom: 0.25rem;
    }


    .section-subheading {
        color: #81798e;

        font-size: 0.84rem;

        margin-bottom: 1.2rem;
    }


    /* ======================================================
       GLASS CARD
    ====================================================== */

    .card {
        background:
            linear-gradient(
                145deg,
                rgba(27, 20, 42, 0.86),
                rgba(16, 11, 26, 0.88)
            );

        border: 1px solid rgba(167, 139, 250, 0.14);

        border-radius: 20px;

        padding: 1.35rem;

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);

        margin-bottom: 1rem;
    }


    .card-header {
        display: flex;

        align-items: center;

        gap: 12px;

        margin-bottom: 1rem;
    }


    .icon-box {
        width: 38px;
        height: 38px;

        display: flex;

        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: rgba(124, 58, 237, 0.13);

        border: 1px solid rgba(167, 139, 250, 0.18);

        font-size: 1rem;
    }


    /* ======================================================
       INPUTS
    ====================================================== */

    .stTextArea textarea {

        background: #0e0a17 !important;

        color: #eeeaff !important;

        border: 1px solid rgba(167, 139, 250, 0.16) !important;

        border-radius: 13px !important;

        font-size: 0.9rem !important;

        line-height: 1.6 !important;

        padding: 1rem !important;
    }


    .stTextArea textarea:hover {

        border-color: rgba(167, 139, 250, 0.30) !important;

    }


    .stTextArea textarea:focus {

        border-color: #8b5cf6 !important;

        box-shadow:
            0 0 0 1px rgba(139, 92, 246, 0.5),
            0 0 25px rgba(124, 58, 237, 0.10) !important;

    }


    /* ======================================================
       FILE UPLOADER
    ====================================================== */

    [data-testid="stFileUploader"] {

        background: rgba(14, 10, 23, 0.65);

        border: 1px dashed rgba(139, 92, 246, 0.30);

        border-radius: 14px;

        padding: 0.4rem;

        margin-top: 0.75rem;
    }


    [data-testid="stFileUploader"]:hover {

        border-color: rgba(167, 139, 250, 0.55);

    }


    /* ======================================================
       PRIMARY BUTTON
    ====================================================== */

    .stButton > button {

        width: 100%;

        min-height: 50px;

        border-radius: 14px !important;

        border: 1px solid rgba(196, 181, 253, 0.15) !important;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #9333ea
            ) !important;

        color: white !important;

        font-weight: 750 !important;

        font-size: 0.95rem !important;

        box-shadow:
            0 12px 30px rgba(124, 58, 237, 0.24);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 18px 38px rgba(124, 58, 237, 0.34);

    }


    /* ======================================================
       RESULT HEADER
    ====================================================== */

    .result-header {

        display: flex;

        align-items: center;

        justify-content: space-between;

        gap: 1rem;

        margin-top: 2.5rem;

        margin-bottom: 1rem;
    }


    .result-title {

        font-size: 1.65rem;

        font-weight: 800;

        color: #f8f7ff;

        letter-spacing: -0.03em;
    }


    .result-label {

        color: #81798e;

        font-size: 0.8rem;
    }


    /* ======================================================
       MATCH BADGES
    ====================================================== */

    .match-strong {

        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(34, 197, 94, 0.10);

        border: 1px solid rgba(34, 197, 94, 0.22);

        color: #86efac;

        font-size: 0.78rem;

        font-weight: 750;
    }


    .match-moderate {

        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(245, 158, 11, 0.10);

        border: 1px solid rgba(245, 158, 11, 0.22);

        color: #fcd34d;

        font-size: 0.78rem;

        font-weight: 750;
    }


    .match-needs {

        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(248, 113, 113, 0.10);

        border: 1px solid rgba(248, 113, 113, 0.22);

        color: #fca5a5;

        font-size: 0.78rem;

        font-weight: 750;
    }


    .match-default {

        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(167, 139, 250, 0.10);

        border: 1px solid rgba(167, 139, 250, 0.22);

        color: #c4b5fd;

        font-size: 0.78rem;

        font-weight: 750;
    }


    /* ======================================================
       SUMMARY
    ====================================================== */

    .summary-card {

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.12),
                rgba(22, 16, 36, 0.75)
            );

        border: 1px solid rgba(167, 139, 250, 0.18);

        border-radius: 20px;

        padding: 1.5rem;

        margin-bottom: 1rem;
    }


    .summary-text {

        color: #d4cee0;

        font-size: 0.96rem;

        line-height: 1.75;
    }


    /* ======================================================
       RESULT ITEMS
    ====================================================== */

    .result-item {

        background: rgba(10, 7, 17, 0.55);

        border: 1px solid rgba(167, 139, 250, 0.10);

        border-radius: 12px;

        padding: 0.75rem 0.85rem;

        margin: 0.45rem 0;

        color: #c8c1d3;

        font-size: 0.88rem;

        line-height: 1.55;
    }


    .result-item:hover {

        border-color: rgba(167, 139, 250, 0.22);

        background: rgba(124, 58, 237, 0.055);

    }


    /* ======================================================
       REQUIREMENT STATUS
    ====================================================== */

    .status-demo {

        color: #86efac;

        font-weight: 700;
    }


    .status-partial {

        color: #fcd34d;

        font-weight: 700;
    }


    .status-missing {

        color: #fca5a5;

        font-weight: 700;
    }


    /* ======================================================
       KEYWORD PILLS
    ====================================================== */

    .keyword-container {

        display: flex;

        flex-wrap: wrap;

        gap: 8px;

        margin-top: 0.7rem;
    }


    .keyword {

        display: inline-block;

        padding: 6px 10px;

        border-radius: 8px;

        background: rgba(124, 58, 237, 0.11);

        border: 1px solid rgba(167, 139, 250, 0.18);

        color: #c4b5fd;

        font-size: 0.78rem;

        font-weight: 600;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .app-footer {

        text-align: center;

        margin-top: 3rem;

        padding-top: 1.5rem;

        border-top: 1px solid rgba(167, 139, 250, 0.08);

        color: #625a6e;

        font-size: 0.75rem;
    }


    /* ======================================================
       MOBILE
    ====================================================== */

    @media (max-width: 768px) {

        .main .block-container {

            padding: 1rem;

        }

        .hero-wrapper {

            padding: 1.5rem 0.5rem 2rem;

        }

        .hero-title {

            font-size: 3rem;

        }

        .result-header {

            flex-direction: column;

            align-items: flex-start;

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
# HELPER FUNCTIONS
# ============================================================

def get_api_key() -> str:

    try:

        return st.secrets["GROQ_API_KEY"]

    except Exception:

        return ""


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


def parse_agent_output(raw_output: Any) -> dict:

    if raw_output is None:

        raise ValueError(
            "The AI returned no output."
        )

    if hasattr(raw_output, "raw"):

        raw_output = raw_output.raw

    text = str(raw_output).strip()

    # Remove markdown fences
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
            "Compare a candidate's resume against a target job "
            "description and provide accurate, evidence-based, "
            "actionable recommendations."
        ),

        backstory=(
            "You are an experienced resume reviewer and technical "
            "recruiter. You carefully compare resumes with job "
            "requirements. You never fabricate qualifications, "
            "experience, education, skills, certifications, or "
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

You are reviewing a resume against a target job description.

====================
RESUME
====================

{resume_text}


====================
JOB DESCRIPTION
====================

{job_description}


====================
IMPORTANT RULES
====================

1. Never fabricate information.

2. Never assume a skill exists unless the resume provides evidence.

3. If something is missing or cannot be verified, say:
"Not demonstrated in the provided resume."

4. Do not make hiring decisions.

5. Do not infer sensitive personal characteristics.

6. Base your analysis only on the provided resume and job description.

7. Give practical and specific recommendations.

8. Return ONLY valid JSON.


====================
REQUIRED JSON FORMAT
====================

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
            "evidence": "Evidence from resume or Not demonstrated in the provided resume."
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
            "Valid JSON matching the exact requested structure."
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
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-wrapper">

        <div class="hero-badge">
            ✦ AI-POWERED RESUME ANALYSIS
        </div>

        <div class="hero-title">
            ResumeIQ
        </div>

        <div class="hero-subtitle">
            Understand how your resume aligns with a target role,
            identify gaps, and get practical improvements backed by
            the information actually present in your resume.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT AREA
# ============================================================

left_column, right_column = st.columns(
    2,
    gap="large",
)


# ============================================================
# RESUME CARD
# ============================================================

with left_column:

    st.markdown(
        """
        <div class="card">

            <div class="card-header">

                <div class="icon-box">
                    📄
                </div>

                <div>

                    <div class="eyebrow">
                        STEP 01
                    </div>

                    <div class="section-heading">
                        Your Resume
                    </div>

                    <div class="section-subheading">
                        Paste your resume or upload a PDF.
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    resume_text = st.text_area(
        "Resume text",
        height=290,
        placeholder=(
            "Paste your resume here...\n\n"
            "Include your experience, skills, education, "
            "projects, certifications, and achievements."
        ),
        label_visibility="collapsed",
    )


    uploaded_file = st.file_uploader(
        "Upload resume PDF",
        type=["pdf"],
        help="Upload a text-based PDF resume.",
    )


# ============================================================
# JOB CARD
# ============================================================

with right_column:

    st.markdown(
        """
        <div class="card">

            <div class="card-header">

                <div class="icon-box">
                    🎯
                </div>

                <div>

                    <div class="eyebrow">
                        STEP 02
                    </div>

                    <div class="section-heading">
                        Target Job
                    </div>

                    <div class="section-subheading">
                        Paste the job description you want to analyze.
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    job_description = st.text_area(
        "Job description",
        height=290,
        placeholder=(
            "Paste the complete job description here...\n\n"
            "Include responsibilities, required skills, "
            "qualifications, and preferred experience."
        ),
        label_visibility="collapsed",
    )


# ============================================================
# ACTION
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

_, center_column, _ = st.columns(
    [1, 2, 1]
)

with center_column:

    analyze_clicked = st.button(
        "✦  Analyze My Resume",
        use_container_width=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    api_key = get_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. Add it in "
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
                    "This PDF does not contain readable text. "
                    "Please use a text-based PDF or paste your "
                    "resume manually."
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
            "Please paste your resume or upload a PDF."
        )

        st.stop()


    if not job_description.strip():

        st.warning(
            "Please paste the target job description."
        )

        st.stop()


    if len(final_resume) < 100:

        st.warning(
            "Your resume appears to be very short. "
            "Please provide more complete resume content."
        )

        st.stop()


    if len(job_description) < 100:

        st.warning(
            "The job description appears to be very short. "
            "Please provide the complete job description."
        )

        st.stop()


    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    with st.status(
        "Analyzing your resume...",
        expanded=True,
    ) as status:

        st.write(
            "Reading your resume..."
        )

        time.sleep(0.3)

        st.write(
            "Comparing experience with job requirements..."
        )

        time.sleep(0.3)

        st.write(
            "Preparing actionable recommendations..."
        )


        try:

            result = analyze_resume(
                final_resume,
                job_description,
                api_key,
            )

            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False,
            )

        except Exception as e:

            status.update(
                label="Analysis failed",
                state="error",
                expanded=True,
            )

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
                    "Please check GROQ_API_KEY in Streamlit Secrets."
                )


            elif "403" in error_text:

                st.error(
                    "Groq rejected the request. "
                    "Check your API key and model access."
                )


            else:

                st.error(
                    f"Something went wrong:\n\n{e}"
                )

            st.stop()


    # ========================================================
    # RESULTS HEADER
    # ========================================================

    match_level = result.get(
        "match_level",
        "Not Available",
    )


    if match_level == "Strong Match":

        badge_class = "match-strong"

    elif match_level == "Moderate Match":

        badge_class = "match-moderate"

    elif match_level == "Needs Improvement":

        badge_class = "match-needs"

    else:

        badge_class = "match-default"


    st.markdown(
        f"""
        <div class="result-header">

            <div>

                <div class="eyebrow">
                    ANALYSIS COMPLETE
                </div>

                <div class="result-title">
                    Your Resume Review
                </div>

            </div>

            <div class="{badge_class}">
                {match_level}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = result.get(
        "summary",
        "No summary was returned.",
    )


    st.markdown(
        f"""
        <div class="summary-card">

            <div class="eyebrow">
                OVERVIEW
            </div>

            <div class="section-heading">
                What the analysis found
            </div>

            <div class="summary-text">
                {summary}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # STRENGTHS + ACTIONS
    # ========================================================

    col1, col2 = st.columns(
        2,
        gap="large",
    )


    # --------------------------------------------------------
    # STRENGTHS
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="eyebrow">
                    YOUR ADVANTAGES
                </div>

                <div class="section-heading">
                    ✦ Strengths
                </div>

                <div class="section-subheading">
                    Relevant evidence already present in your resume.
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
                    <div class="result-item">
                        ✓ &nbsp; {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(
                "No strengths were identified."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # PRIORITY ACTIONS
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="eyebrow">
                    NEXT STEPS
                </div>

                <div class="section-heading">
                    🚀 Priority Actions
                </div>

                <div class="section-subheading">
                    Changes you can focus on first.
                </div>

            """,
            unsafe_allow_html=True,
        )


        actions = result.get(
            "priority_actions",
            [],
        )


        if actions:

            for index, item in enumerate(
                actions,
                start=1,
            ):

                st.markdown(
                    f"""
                    <div class="result-item">
                        <strong>{index:02d}</strong>
                        &nbsp;&nbsp; {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(
                "No priority actions were identified."
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # REQUIREMENT ANALYSIS
    # ========================================================

    st.markdown(
        """
        <div class="card">

            <div class="eyebrow">
                JOB ALIGNMENT
            </div>

            <div class="section-heading">
                🔎 Requirement Analysis
            </div>

            <div class="section-subheading">
                How your resume currently demonstrates the
                requirements of this role.
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
                "Unknown requirement",
            )

            status = item.get(
                "status",
                "Unknown",
            )

            evidence = item.get(
                "evidence",
                "",
            )


            if status == "Demonstrated":

                status_display = (
                    '<span class="status-demo">'
                    '✓ Demonstrated'
                    '</span>'
                )

            elif status == "Partially Demonstrated":

                status_display = (
                    '<span class="status-partial">'
                    '◐ Partially Demonstrated'
                    '</span>'
                )

            else:

                status_display = (
                    '<span class="status-missing">'
                    '× Not Demonstrated'
                    '</span>'
                )


            with st.expander(
                f"{requirement}  ·  {status}"
            ):

                st.markdown(
                    f"""
                    <div style="
                        color:#8f879f;
                        font-size:0.75rem;
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                        margin-bottom:0.45rem;
                    ">
                        STATUS
                    </div>

                    <div style="
                        margin-bottom:1rem;
                    ">
                        {status_display}
                    </div>

                    <div style="
                        color:#8f879f;
                        font-size:0.75rem;
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                        margin-bottom:0.45rem;
                    ">
                        EVIDENCE
                    </div>

                    <div style="
                        color:#c8c1d3;
                        line-height:1.6;
                    ">
                        {evidence}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    else:

        st.info(
            "No requirement analysis was returned."
        )


    # ========================================================
    # MISSING / UNCLEAR
    # ========================================================

    missing = result.get(
        "missing_or_unclear",
        [],
    )


    if missing:

        st.markdown(
            """
            <div class="card">

                <div class="eyebrow">
                    GAPS TO REVIEW
                </div>

                <div class="section-heading">
                    ⚠️ Missing or Unclear
                </div>

                <div class="section-subheading">
                    Areas that are not clearly demonstrated by
                    the provided resume.
                </div>

            """,
            unsafe_allow_html=True,
        )


        for item in missing:

            st.markdown(
                f"""
                <div class="result-item">
                    {item}
                </div>
                """,
                unsafe_allow_html=True,
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # RESUME IMPROVEMENTS
    # ========================================================

    improvements = result.get(
        "resume_improvements",
        [],
    )


    if improvements:

        st.markdown(
            """
            <div class="card">

                <div class="eyebrow">
                    EDIT YOUR RESUME
                </div>

                <div class="section-heading">
                    ✍️ Recommended Improvements
                </div>

                <div class="section-subheading">
                    Practical ways to make your resume clearer
                    and more relevant to the target role.
                </div>

            """,
            unsafe_allow_html=True,
        )


        for item in improvements:

            st.markdown(
                f"""
                <div class="result-item">
                    → &nbsp; {item}
                </div>
                """,
                unsafe_allow_html=True,
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # ATS KEYWORDS
    # ========================================================

    keywords = result.get(
        "ats_keywords",
        [],
    )


    if keywords:

        keyword_html = ""


        for keyword in keywords:

            keyword_html += (
                f'<span class="keyword">'
                f'{keyword}'
                f'</span>'
            )


        st.markdown(
            f"""
            <div class="card">

                <div class="eyebrow">
                    ATS & JOB LANGUAGE
                </div>

                <div class="section-heading">
                    🧠 Relevant Keywords
                </div>

                <div class="section-subheading">
                    Terms appearing in the target role that may
                    be useful when they truthfully describe your
                    experience.
                </div>

                <div class="keyword-container">
                    {keyword_html}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    disclaimer = result.get(
        "disclaimer",
        "This review is informational and does not guarantee hiring outcomes.",
    )


    st.info(
        disclaimer
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        ResumeIQ&nbsp;&nbsp;·&nbsp;&nbsp;
        AI-powered resume feedback&nbsp;&nbsp;·&nbsp;&nbsp;
        Built with CrewAI + Groq
    </div>
    """,
    unsafe_allow_html=True,
)
