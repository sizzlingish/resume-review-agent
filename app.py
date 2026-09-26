import json
import time
from typing import Any

import streamlit as st
from pypdf import PdfReader

from crewai import Agent, Crew, Process, Task, LLM


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="📄",
    layout="wide",
)


MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📄 AI Resume Review Agent")

st.markdown(
    """
    Compare a resume with a target job description and receive
    evidence-based, actionable improvement suggestions.

    **Important:** The reviewer only uses information provided in
    the resume. It does not invent qualifications or experience.
    """
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_groq_api_key() -> str | None:
    """
    Safely retrieve the Groq API key from Streamlit secrets.
    """

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return str(api_key).strip()

    except Exception:
        return None


def extract_pdf_text(uploaded_file) -> tuple[str | None, str | None]:
    """
    Extract text from an uploaded PDF.

    Returns:
        (text, error_message)
    """

    try:
        reader = PdfReader(uploaded_file)

        if not reader.pages:
            return None, "The PDF does not contain any pages."

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()

                if page_text:
                    extracted_pages.append(page_text)

            except Exception as page_error:
                return (
                    None,
                    f"Could not extract text from PDF page {page_number}: "
                    f"{page_error}",
                )

        text = "\n\n".join(extracted_pages).strip()

        if not text:
            return (
                None,
                "No readable text was found in the PDF. "
                "The file may be scanned/image-based. "
                "Please paste the resume text instead.",
            )

        return text, None

    except Exception as error:
        return None, f"Could not read the PDF: {error}"


def clean_json_response(raw_response: str) -> dict[str, Any]:
    """
    Convert the agent's response into a Python dictionary.

    The function also handles cases where the LLM surrounds JSON
    with Markdown code fences.
    """

    text = raw_response.strip()

    # Remove Markdown JSON fences if present.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Find the outermost JSON object if the model added extra text.
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace + 1]

    return json.loads(text)


def build_resume_review_agent(groq_api_key: str) -> Agent:
    """
    Create the single CrewAI resume reviewer agent.
    """

    llm = LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=5000,
    )

    return Agent(
        role="Resume and Job Description Matching Specialist",
        goal=(
            "Evaluate how closely a candidate's provided resume matches "
            "a target job description and provide accurate, actionable "
            "recommendations without inventing qualifications."
        ),
        backstory=(
            "You are a careful professional resume reviewer. "
            "You compare evidence in a resume against the requirements "
            "of a job description. You never assume that a candidate "
            "has a skill, qualification, degree, certification, tool, "
            "or experience unless it is explicitly supported by the "
            "provided resume. When information is missing, say that "
            "it is not demonstrated in the provided resume."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def run_resume_review(
    resume_text: str,
    job_description: str,
    groq_api_key: str,
) -> dict[str, Any]:

    agent = build_resume_review_agent(groq_api_key)

    task_description = f"""
You are reviewing a resume against a target job description.

================ RESUME ================
{resume_text}

================ JOB DESCRIPTION ================
{job_description}

================ IMPORTANT RULES ================

1. Use ONLY the information contained in the resume and job description.

2. NEVER invent:
   - skills
   - work experience
   - years of experience
   - degrees
   - certifications
   - job titles
   - achievements
   - technologies
   - responsibilities
   - metrics

3. Distinguish carefully between:
   - explicitly demonstrated
   - partially demonstrated
   - not demonstrated in the provided resume

4. If the resume does not mention something, do NOT claim that
   the candidate does not have it.

5. For missing information, use wording such as:
   "Not demonstrated in the provided resume."

6. Recommendations must be actionable and realistic.

7. Do not recommend adding a qualification unless the candidate
   actually has that qualification. Instead, recommend verifying,
   clarifying, or adding it only if it is genuinely possessed.

8. Do not make hiring decisions.

9. Do not discriminate based on protected characteristics.

10. Do not infer age, gender, ethnicity, religion, nationality,
    disability, marital status, health, or other sensitive traits.

================ OUTPUT ================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "overall_summary": "A short factual summary of the match.",

  "match_level": "Strong / Moderate / Limited / Insufficient evidence",

  "key_strengths": [
    {{
      "requirement": "Requirement from the job description",
      "evidence": "Specific evidence found in the resume"
    }}
  ],

  "requirement_analysis": [
    {{
      "requirement": "Requirement or responsibility from the job description",
      "status": "Demonstrated / Partially demonstrated / Not demonstrated",
      "resume_evidence": "Evidence from resume or 'Not demonstrated in the provided resume.'",
      "recommendation": "What the candidate should clarify, strengthen, or verify."
    }}
  ],

  "missing_or_unclear_information": [
    "Important requirement that is not demonstrated or is unclear"
  ],

  "resume_improvements": [
    {{
      "area": "Experience / Skills / Summary / Projects / Education / Formatting",
      "recommendation": "Specific improvement",
      "example": "A safe example of how to improve wording without inventing facts"
    }}
  ],

  "ats_keywords_to_consider": [
    "Keyword or phrase from the job description that could be naturally included IF supported by the candidate's real experience"
  ],

  "priority_actions": [
    "Most important improvement",
    "Second most important improvement",
    "Third most important improvement"
  ],

  "important_disclaimer": "This review is based only on the supplied resume and job description. Missing information means it was not demonstrated in the supplied resume; it does not establish that the candidate lacks the qualification."
}}

Return no Markdown and no explanation outside the JSON.
"""

    task = Task(
        description=task_description,
        expected_output="A valid JSON object following the exact requested schema.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # Small retry loop for transient API failures.
    last_error = None

    for attempt in range(3):
        try:
            result = crew.kickoff()

            raw_output = str(result.raw)

            return clean_json_response(raw_output)

        except Exception as error:
            last_error = error

            # Don't wait after the final attempt.
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(str(last_error))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("How it works")

    st.markdown(
        """
        **1. Provide your resume**

        Paste the text or upload a PDF.

        **2. Add the target job**

        Paste the job description.

        **3. Review**

        The CrewAI agent compares the two.

        **4. Improve**

        Use the recommendations to strengthen your resume.
        """
    )

    st.divider()

    st.caption(
        "Model: openai/gpt-oss-120b via Groq"
    )

    st.caption(
        "Single-agent CrewAI application"
    )


# ============================================================
# INPUT SECTION
# ============================================================

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("1️⃣ Resume")

    resume_input_method = st.radio(
        "Choose resume input method:",
        ["Paste text", "Upload PDF"],
        horizontal=True,
    )

    resume_text = ""

    if resume_input_method == "Paste text":

        resume_text = st.text_area(
            "Paste your resume here",
            height=400,
            placeholder=(
                "Paste the complete resume text here..."
            ),
        )

    else:

        uploaded_pdf = st.file_uploader(
            "Upload resume PDF",
            type=["pdf"],
            help="Upload a text-based PDF resume.",
        )

        if uploaded_pdf is not None:

            with st.spinner("Extracting resume text..."):
                extracted_text, pdf_error = extract_pdf_text(
                    uploaded_pdf
                )

            if pdf_error:
                st.error(pdf_error)

            else:
                resume_text = extracted_text or ""

                st.success(
                    "Resume text extracted successfully."
                )

                with st.expander("Preview extracted text"):
                    st.text_area(
                        "Extracted resume",
                        value=resume_text,
                        height=300,
                        disabled=True,
                    )


with right_column:
    st.subheader("2️⃣ Target Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=400,
        placeholder=(
            "Paste the complete target job description here..."
        ),
    )


# ============================================================
# REVIEW BUTTON
# ============================================================

st.divider()

review_button = st.button(
    "🔍 Review Resume",
    type="primary",
    use_container_width=True,
)


# ============================================================
# VALIDATION + AGENT EXECUTION
# ============================================================

if review_button:

    # ----------------------------------------
    # Validate API key
    # ----------------------------------------

    groq_api_key = get_groq_api_key()

    if not groq_api_key:
        st.error(
            "Groq API key is missing. Add GROQ_API_KEY to "
            "Streamlit Secrets before running the review."
        )
        st.stop()

    # ----------------------------------------
    # Validate resume
    # ----------------------------------------

    resume_text = resume_text.strip()

    if not resume_text:
        st.warning(
            "Please provide your resume by pasting its text "
            "or uploading a readable PDF."
        )
        st.stop()

    # ----------------------------------------
    # Validate job description
    # ----------------------------------------

    job_description = job_description.strip()

    if not job_description:
        st.warning(
            "Please paste the target job description."
        )
        st.stop()

    # ----------------------------------------
    # Basic size protection
    # ----------------------------------------

    if len(resume_text) < 100:
        st.warning(
            "The resume appears to contain very little text. "
            "Please provide the complete resume."
        )
        st.stop()

    if len(job_description) < 100:
        st.warning(
            "The job description appears to contain very little text. "
            "Please provide the complete job description."
        )
        st.stop()

    # ----------------------------------------
    # Run CrewAI
    # ----------------------------------------

    st.subheader("🤖 Resume Review")

    progress_message = st.empty()

    try:

        progress_message.info(
            "The CrewAI reviewer is analyzing the resume..."
        )

        review = run_resume_review(
            resume_text=resume_text,
            job_description=job_description,
            groq_api_key=groq_api_key,
        )

        progress_message.empty()

    except Exception as error:

        progress_message.empty()

        error_text = str(error).lower()

        # ----------------------------------------
        # Rate limit handling
        # ----------------------------------------

        if (
            "429" in error_text
            or "rate limit" in error_text
            or "too many requests" in error_text
        ):
            st.error(
                "Groq rate limit reached. Please wait a little "
                "while and try again."
            )

        # ----------------------------------------
        # Authentication handling
        # ----------------------------------------

        elif (
            "401" in error_text
            or "authentication" in error_text
            or "api key" in error_text
        ):
            st.error(
                "Groq authentication failed. Please check that "
                "your GROQ_API_KEY is correct."
            )

        # ----------------------------------------
        # Model / request errors
        # ----------------------------------------

        elif (
            "400" in error_text
            or "model" in error_text
            or "bad request" in error_text
        ):
            st.error(
                "The AI request was rejected. Check the model "
                "configuration and try again."
            )

        # ----------------------------------------
        # Everything else
        # ----------------------------------------

        else:
            st.error(
                "The resume review could not be completed."
            )

            with st.expander("Technical details"):
                st.code(str(error))

        st.stop()

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    match_level = review.get(
        "match_level",
        "Not available",
    )

    st.metric(
        "Match assessment",
        match_level,
    )

    st.subheader("Summary")

    st.write(
        review.get(
            "overall_summary",
            "No summary was returned.",
        )
    )

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    st.subheader("💪 Key Strengths")

    strengths = review.get("key_strengths", [])

    if strengths:

        for item in strengths:

            requirement = item.get(
                "requirement",
                "Requirement",
            )

            evidence = item.get(
                "evidence",
                "No evidence provided.",
            )

            st.markdown(
                f"**{requirement}**"
            )

            st.write(evidence)

    else:
        st.info(
            "No specific strengths were identified."
        )

    # --------------------------------------------------------
    # Requirement analysis
    # --------------------------------------------------------

    st.subheader("📋 Requirement Analysis")

    requirements = review.get(
        "requirement_analysis",
        [],
    )

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
            "resume_evidence",
            "No evidence provided.",
        )

        recommendation = item.get(
            "recommendation",
            "No recommendation provided.",
        )

        with st.expander(
            f"{status}: {requirement}"
        ):

            st.markdown("**Resume evidence**")

            st.write(evidence)

            st.markdown("**Recommendation**")

            st.write(recommendation)

    # --------------------------------------------------------
    # Missing / unclear
    # --------------------------------------------------------

    st.subheader(
        "❓ Missing or Unclear Information"
    )

    missing = review.get(
        "missing_or_unclear_information",
        [],
    )

    if missing:

        for item in missing:
            st.markdown(f"- {item}")

    else:
        st.success(
            "No major missing or unclear requirements were identified."
        )

    # --------------------------------------------------------
    # Resume improvements
    # --------------------------------------------------------

    st.subheader(
        "✏️ Resume Improvement Recommendations"
    )

    improvements = review.get(
        "resume_improvements",
        [],
    )

    for item in improvements:

        area = item.get(
            "area",
            "Resume",
        )

        recommendation = item.get(
            "recommendation",
            "",
        )

        example = item.get(
            "example",
            "",
        )

        st.markdown(
            f"### {area}"
        )

        st.write(recommendation)

        if example:
            st.markdown("**Example:**")
            st.info(example)

    # --------------------------------------------------------
    # ATS keywords
    # --------------------------------------------------------

    st.subheader(
        "🔑 ATS Keywords to Consider"
    )

    keywords = review.get(
        "ats_keywords_to_consider",
        [],
    )

    if keywords:

        st.write(
            ", ".join(
                f"`{keyword}`"
                for keyword in keywords
            )
        )

    else:

        st.info(
            "No additional keywords were identified."
        )

    # --------------------------------------------------------
    # Priority actions
    # --------------------------------------------------------

    st.subheader(
        "🚀 Priority Actions"
    )

    priority_actions = review.get(
        "priority_actions",
        [],
    )

    for index, action in enumerate(
        priority_actions,
        start=1,
    ):
        st.markdown(
            f"**{index}.** {action}"
        )

    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    st.divider()

    st.caption(
        review.get(
            "important_disclaimer",
            "This review is based only on the supplied resume "
            "and job description.",
        )
    )
