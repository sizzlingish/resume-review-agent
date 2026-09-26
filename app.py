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
    page_title="ResumeIQ — AI Resume Review",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(124, 58, 237, 0.18),
                transparent 32%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(168, 85, 247, 0.12),
                transparent 28%
            ),
            #090611;
        color: #f5f3ff;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* ---------- HIDE STREAMLIT DEFAULTS ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* ---------- TYPOGRAPHY ---------- */

    h1, h2, h3 {
        color: #f8f7ff !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown {
        color: #c9c4d8;
    }

    /* ---------- HERO ---------- */

    .hero {
        text-align: center;
        padding: 2.5rem 1rem 2rem 1rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.14);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #c4b5fd;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2.4rem, 5vw, 4.4rem);
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
        background: linear-gradient(
            90deg,
            #ffffff,
            #d8b4fe,
            #a78bfa
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 700px;
        margin: 1rem auto 0 auto;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #aaa3bd;
    }

    /* ---------- CARDS ---------- */

    .glass-card {
        background: rgba(22, 16, 36, 0.78);
        border: 1px solid rgba(167, 139, 250, 0.16);
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow:
            0 18px 50px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f5f3ff;
        margin-bottom: 0.2rem;
    }

    .section-description {
        font-size: 0.88rem;
        color: #928ba5;
        margin-bottom: 1rem;
    }

    /* ---------- INPUTS ---------- */

    .stTextArea textarea,
    .stTextInput input {
        background: rgba(14, 10, 24, 0.85) !important;
        color: #f5f3ff !important;
        border: 1px solid rgba(167, 139, 250, 0.18) !important;
        border-radius: 14px !important;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: rgba(14, 10, 24, 0.65);
        border: 1px dashed rgba(167, 139, 250, 0.35);
        border-radius: 14px;
        padding: 0.5rem;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: white !important;
        background: linear-gradient(
            135deg,
            #7c3aed,
            #9333ea
        ) !important;
        box-shadow:
            0 10px 30px rgba(124, 58, 237, 0.28);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 14px 35px rgba(124, 58, 237, 0.4);
    }

    /* ---------- RESULT HEADER ---------- */

    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 2rem 0 1.2rem 0;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 750;
        color: #f8f7ff;
    }

    .match-badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.16);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #c4b5fd;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: rgba(22, 16, 36, 0.78);
        border: 1px solid rgba(167, 139, 250, 0.14);
        border-radius: 18px;
        padding: 1.2rem;
        min-height: 120px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8f879f;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 750;
        color: #eee9ff;
    }

    /* ---------- LIST ITEMS ---------- */

    .custom-item {
        background: rgba(14, 10, 24, 0.55);
        border: 1px solid rgba(167, 139, 250, 0.1);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin: 0.45rem 0;
        color: #d6d0e2;
        line-height: 1.5;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(167, 139, 250, 0.1);
        color: #696276;
        font-size: 0.8rem;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.2rem;
        }

        .hero {
            padding-top: 1rem;
        }

        .hero-title {
            font-size: 2.5rem;
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
# HELPERS
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
        raise ValueError(f"Could not read the PDF: {e}")


def parse_agent_output(raw_output: Any) -> dict:
    if raw_output is None:
        raise ValueError("The agent returned no output.")

    if hasattr(raw_output, "raw"):
        raw_output = raw_output.raw

    text = str(raw_output).strip()

    # Remove markdown code fences if the model adds them
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

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

    raise ValueError("The AI returned an invalid response format.")


# ============================================================
# CREWAI AGENT
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
            "Analyze a candidate's resume against a target job description "
            "and provide accurate, evidence-based, actionable feedback."
        ),
        backstory=(
            "You are an experienced technical recruiter and resume reviewer. "
            "You carefully compare resumes with job requirements. "
            "You never invent qualifications, experience, skills, education, "
            "certifications, or achievements that are not supported by the resume."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return agent


# ============================================================
# ANALYSIS
# ============================================================

def analyze_resume(resume_text: str, job_description: str, api_key: str):

    agent = create_resume_agent(api_key)

    task_description = f"""
You are reviewing a resume against a target job description.

RESUME:
{resume_text}

TARGET JOB DESCRIPTION:
{job_description}

Analyze the resume carefully.

IMPORTANT RULES:
1. Never fabricate information.
2. Never assume a skill exists unless the resume supports it.
3. If something is not present or cannot be verified, say:
   "Not demonstrated in the provided resume."
4. Do not make hiring decisions.
5. Do not infer sensitive personal characteristics.
6. Focus only on evidence from the resume and requirements from the job description.
7. Give practical recommendations that the candidate can actually use.
8. Keep the response concise but useful.

Return ONLY valid JSON using exactly this structure:

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
        expected_output="Valid JSON matching the requested structure.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # Retry a few times for temporary API failures
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
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ AI-POWERED RESUME ANALYZER
        </div>

        <div class="hero-title">
            ResumeIQ
        </div>

        <div class="hero-subtitle">
            Compare your resume with any job description and discover
            exactly what to improve — with clear, evidence-based AI feedback.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(2, gap="large")


with left:

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                📄 Your Resume
            </div>

            <div class="section-description">
                Paste your resume or upload a PDF.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    resume_input = st.text_area(
        "Paste resume",
        height=300,
        placeholder=(
            "Paste your resume text here...\n\n"
            "Example:\n"
            "Software Engineer with 2 years of experience..."
        ),
        label_visibility="collapsed",
    )

    uploaded_file = st.file_uploader(
        "Or upload your resume as PDF",
        type=["pdf"],
    )


with right:

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                🎯 Target Job
            </div>

            <div class="section-description">
                Paste the job description you want to compare against.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    job_description = st.text_area(
        "Job description",
        height=300,
        placeholder=(
            "Paste the target job description here...\n\n"
            "Example:\n"
            "We are looking for a Python developer..."
        ),
        label_visibility="collapsed",
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# REVIEW BUTTON
# ============================================================

review_clicked = st.button(
    "✦ Analyze My Resume",
    use_container_width=True,
)


# ============================================================
# PROCESS
# ============================================================

if review_clicked:

    api_key = get_api_key()

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud → "
            "Settings → Secrets."
        )
        st.stop()

    # Get resume from PDF if supplied
    final_resume = resume_input.strip()

    if uploaded_file is not None:

        try:
            pdf_text = extract_pdf_text(uploaded_file)

            if pdf_text:
                final_resume = pdf_text

            else:
                st.error(
                    "The PDF does not contain readable text. "
                    "If it is a scanned resume, please paste the text instead."
                )
                st.stop()

        except Exception as e:

            st.error(str(e))
            st.stop()

    if not final_resume:

        st.warning("Please provide your resume first.")
        st.stop()

    if not job_description.strip():

        st.warning("Please provide the target job description.")
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

    # ---------- RUN AGENT ----------

    with st.spinner("Analyzing your resume against the job description..."):

        try:

            result = analyze_resume(
                final_resume,
                job_description,
                api_key,
            )

        except Exception as e:

            error_text = str(e).lower()

            if "429" in error_text or "rate limit" in error_text:

                st.error(
                    "Groq rate limit reached. Please wait a moment and try again."
                )

            elif "401" in error_text or "authentication" in error_text:

                st.error(
                    "Groq authentication failed. Check your GROQ_API_KEY "
                    "in Streamlit Secrets."
                )

            elif "403" in error_text:

                st.error(
                    "Groq rejected the request. Check your API key and "
                    "model access."
                )

            else:

                st.error(
                    f"Something went wrong while analyzing the resume:\n\n{e}"
                )

            st.stop()

    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown(
        """
        <div class="result-header">

            <div class="result-title">
                ✦ Resume Analysis
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    match_level = result.get(
        "match_level",
        "Not available",
    )

    st.markdown(
        f"""
        <div class="match-badge">
            {match_level}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = result.get("summary", "")

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                ✨ Overall Summary
            </div>

        """,
        unsafe_allow_html=True,
    )

    st.markdown(summary)

    st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # STRENGTHS + PRIORITY ACTIONS
    # ========================================================

    col1, col2 = st.columns(2, gap="large")


    with col1:

        st.markdown(
            """
            <div class="glass-card">

                <div class="section-title">
                    💜 Your Strengths
                </div>

            """,
            unsafe_allow_html=True,
        )

        strengths = result.get("strengths", [])

        if strengths:

            for item in strengths:

                st.markdown(
                    f"""
                    <div class="custom-item">
                        ✓ {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.markdown("No strengths identified.")

        st.markdown("</div>", unsafe_allow_html=True)


    with col2:

        st.markdown(
            """
            <div class="glass-card">

                <div class="section-title">
                    🚀 Priority Actions
                </div>

            """,
            unsafe_allow_html=True,
        )

        actions = result.get("priority_actions", [])

        if actions:

            for index, item in enumerate(actions, start=1):

                st.markdown(
                    f"""
                    <div class="custom-item">
                        <strong>{index}.</strong> {item}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.markdown("No priority actions identified.")

        st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # REQUIREMENT ANALYSIS
    # ========================================================

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                🔍 Job Requirement Analysis
            </div>

            <div class="section-description">
                How your resume demonstrates the requirements of the role.
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

            with st.expander(
                f"{requirement} — {status}"
            ):

                st.write(evidence)

    else:

        st.info("No requirement analysis was returned.")


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
            <div class="glass-card">

                <div class="section-title">
                    ⚠️ Missing or Unclear
                </div>

            """,
            unsafe_allow_html=True,
        )

        for item in missing:

            st.markdown(
                f"""
                <div class="custom-item">
                    {item}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


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
            <div class="glass-card">

                <div class="section-title">
                    ✍️ Resume Improvements
                </div>

            """,
            unsafe_allow_html=True,
        )

        for item in improvements:

            st.markdown(
                f"""
                <div class="custom-item">
                    → {item}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # ATS KEYWORDS
    # ========================================================

    keywords = result.get(
        "ats_keywords",
        [],
    )

    if keywords:

        st.markdown(
            """
            <div class="glass-card">

                <div class="section-title">
                    🧠 ATS Keywords
                </div>

                <div class="section-description">
                    Relevant terms found in the job description that
                    may be useful when they are genuinely supported
                    by your experience.
                </div>

            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            " ".join(
                [
                    f"`{keyword}`"
                    for keyword in keywords
                ]
            )
        )

        st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # DISCLAIMER
    # ========================================================

    disclaimer = result.get(
        "disclaimer",
        "This review is informational and does not guarantee hiring outcomes.",
    )

    st.info(disclaimer)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ResumeIQ · AI-powered resume feedback · Built with CrewAI + Groq
    </div>
    """,
    unsafe_allow_html=True,
)
