import json
import time
from typing import Any

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Crew, Process, Task, LLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_api_key() -> str | None:
    """Get the Groq API key from Streamlit Secrets."""

    try:
        key = st.secrets["GROQ_API_KEY"]
        return str(key).strip()

    except Exception:
        return None


def extract_pdf_text(uploaded_file) -> tuple[str | None, str | None]:
    """Extract readable text from an uploaded PDF."""

    try:
        reader = PdfReader(uploaded_file)

        if len(reader.pages) == 0:
            return None, "The PDF does not contain any pages."

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()

                if text:
                    pages.append(text)

            except Exception as error:
                return (
                    None,
                    f"Could not read page {page_number}: {error}",
                )

        extracted_text = "\n\n".join(pages).strip()

        if not extracted_text:
            return (
                None,
                "No readable text was found in this PDF. "
                "It may be a scanned/image-based PDF. "
                "Please paste the resume text instead.",
            )

        return extracted_text, None

    except Exception as error:
        return None, f"Could not read the PDF: {error}"


def parse_agent_output(raw_output: str) -> dict[str, Any]:
    """Convert the agent response into a Python dictionary."""

    text = raw_output.strip()

    # Remove Markdown code fences if the model adds them.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Find JSON object if there is extra text.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("The AI did not return valid JSON.")

    text = text[start:end + 1]

    return json.loads(text)


# ============================================================
# CREWAI AGENT
# ============================================================

def create_resume_agent(api_key: str) -> Agent:

    llm = LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=api_key,
        temperature=0,
        max_tokens=5000,
    )

    agent = Agent(
        role="Resume Review Specialist",

        goal=(
            "Compare a candidate's resume with a target job "
            "description and provide an accurate, evidence-based "
            "review with actionable improvement suggestions."
        ),

        backstory=(
            "You are a careful professional resume reviewer. "
            "You analyze only the information provided in the "
            "resume and job description. You never invent skills, "
            "experience, qualifications, certifications, education, "
            "achievements, technologies, or years of experience."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=3,
    )

    return agent


# ============================================================
# RUN RESUME REVIEW
# ============================================================

def review_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
) -> dict[str, Any]:

    agent = create_resume_agent(api_key)

    task = Task(
        description=f"""
You are reviewing a candidate's resume against a target job description.

================ RESUME ================

{resume_text}

================ JOB DESCRIPTION ================

{job_description}

================ RULES ================

Follow these rules strictly:

1. Use ONLY information contained in the supplied resume
   and job description.

2. NEVER invent or assume:
   - skills
   - work experience
   - years of experience
   - degrees
   - certifications
   - job titles
   - achievements
   - technologies
   - responsibilities
   - projects
   - metrics

3. If a requirement is not mentioned in the resume, say:

   "Not demonstrated in the provided resume."

4. Do NOT say that the candidate definitely lacks a skill
   just because it is not mentioned.

5. Distinguish between:
   - Demonstrated
   - Partially demonstrated
   - Not demonstrated

6. Recommendations should be actionable.

7. Never recommend that the candidate falsely add a skill,
   qualification, experience, certification, or achievement.

8. If suggesting a keyword, say that it should only be added
   if it genuinely describes the candidate's experience.

9. Do not make a hiring decision.

10. Do not infer sensitive personal characteristics.

================ REQUIRED OUTPUT ================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "overall_summary": "Short factual summary.",

    "match_level": "Strong / Moderate / Limited / Insufficient evidence",

    "key_strengths": [
        {{
            "requirement": "Job requirement",
            "evidence": "Evidence from the resume"
        }}
    ],

    "requirement_analysis": [
        {{
            "requirement": "Requirement from job description",
            "status": "Demonstrated / Partially demonstrated / Not demonstrated",
            "resume_evidence": "Evidence from resume or Not demonstrated in the provided resume.",
            "recommendation": "Actionable recommendation"
        }}
    ],

    "missing_or_unclear_information": [
        "Requirement that is missing or unclear"
    ],

    "resume_improvements": [
        {{
            "area": "Summary / Experience / Skills / Projects / Education / Formatting",
            "recommendation": "Specific improvement",
            "example": "Example wording that does not invent information"
        }}
    ],

    "ats_keywords_to_consider": [
        "Relevant keyword from job description"
    ],

    "priority_actions": [
        "Most important improvement",
        "Second important improvement",
        "Third important improvement"
    ],

    "important_disclaimer": "This review is based only on the supplied resume and job description. Not mentioned means not demonstrated in the supplied resume; it does not prove that the candidate lacks the qualification."
}}

Return no Markdown.
Return no explanation outside the JSON.
""",

        expected_output=(
            "A valid JSON object containing the requested resume review."
        ),

        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # Retry a few times for temporary API problems.
    last_error = None

    for attempt in range(3):

        try:

            result = crew.kickoff()

            return parse_agent_output(
                str(result.raw)
            )

        except Exception as error:

            last_error = error

            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(str(last_error))


# ============================================================
# USER INTERFACE
# ============================================================

st.title("📄 AI Resume Review Agent")

st.write(
    """
    Upload or paste a resume and provide a target job description.
    The AI agent will compare them and provide evidence-based,
    actionable recommendations.
    """
)

st.info(
    "The agent does not invent qualifications. "
    "If something is not mentioned in the resume, it is reported "
    "as not demonstrated rather than assumed."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About this app")

    st.write(
        """
        This is a single-agent CrewAI application.

        **Streamlit**
        creates the interface.

        **CrewAI**
        manages the AI agent.

        **Groq**
        provides the language model.

        **GPT-OSS 120B**
        performs the resume analysis.
        """
    )

    st.divider()

    st.caption(
        f"Model: {MODEL_NAME}"
    )


# ============================================================
# INPUTS
# ============================================================

resume_column, job_column = st.columns(2)


# ------------------------------------------------------------
# RESUME
# ------------------------------------------------------------

with resume_column:

    st.subheader("1️⃣ Resume")

    input_method = st.radio(
        "Choose how to provide the resume:",
        [
            "Paste resume text",
            "Upload PDF",
        ],
        horizontal=True,
    )

    resume_text = ""

    if input_method == "Paste resume text":

        resume_text = st.text_area(
            "Resume text",
            height=400,
            placeholder=(
                "Paste the candidate's complete resume here..."
            ),
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload resume PDF",
            type=["pdf"],
        )

        if uploaded_file is not None:

            with st.spinner("Reading PDF..."):

                extracted_text, error = extract_pdf_text(
                    uploaded_file
                )

            if error:

                st.error(error)

            else:

                resume_text = extracted_text

                st.success(
                    "PDF text extracted successfully."
                )

                with st.expander(
                    "Preview extracted resume"
                ):

                    st.text(
                        resume_text
                    )


# ------------------------------------------------------------
# JOB DESCRIPTION
# ------------------------------------------------------------

with job_column:

    st.subheader("2️⃣ Target Job")

    job_description = st.text_area(
        "Job description",
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
# PROCESS REQUEST
# ============================================================

if review_button:

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key = get_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )

        st.stop()

    # --------------------------------------------------------
    # RESUME VALIDATION
    # --------------------------------------------------------

    resume_text = resume_text.strip()

    if not resume_text:

        st.warning(
            "Please provide a resume by pasting the text "
            "or uploading a PDF."
        )

        st.stop()

    # --------------------------------------------------------
    # JOB DESCRIPTION VALIDATION
    # --------------------------------------------------------

    job_description = job_description.strip()

    if not job_description:

        st.warning(
            "Please paste the target job description."
        )

        st.stop()

    # --------------------------------------------------------
    # BASIC INPUT CHECK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Resume Review Agent is analyzing the resume..."
    ):

        try:

            review = review_resume(
                resume_text=resume_text,
                job_description=job_description,
                api_key=api_key,
            )

        except Exception as error:

            error_message = str(error).lower()

            if (
                "429" in error_message
                or "rate limit" in error_message
                or "too many requests" in error_message
            ):

                st.error(
                    "Groq rate limit reached. "
                    "Please wait a little and try again."
                )

            elif (
                "401" in error_message
                or "authentication" in error_message
                or "api key" in error_message
            ):

                st.error(
                    "Groq authentication failed. "
                    "Please check your GROQ_API_KEY in "
                    "Streamlit Secrets."
                )

            elif (
                "403" in error_message
                or "permission" in error_message
            ):

                st.error(
                    "Groq rejected the request because the "
                    "model is not permitted for this API key/project."
                )

            elif (
                "400" in error_message
                or "bad request" in error_message
            ):

                st.error(
                    "Groq rejected the request. "
                    "Please check the model configuration."
                )

            else:

                st.error(
                    "The resume review could not be completed."
                )

                with st.expander(
                    "Technical error"
                ):

                    st.code(
                        str(error)
                    )

            st.stop()

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.success(
        "Resume review completed."
    )

    st.divider()

    # --------------------------------------------------------
    # MATCH LEVEL
    # --------------------------------------------------------

    st.subheader("📊 Overall Assessment")

    match_level = review.get(
        "match_level",
        "Not available",
    )

    st.metric(
        "Match level",
        match_level,
    )

    st.write(
        review.get(
            "overall_summary",
            "No summary available.",
        )
    )

    # --------------------------------------------------------
    # STRENGTHS
    # --------------------------------------------------------

    st.subheader("💪 Key Strengths")

    strengths = review.get(
        "key_strengths",
        [],
    )

    if strengths:

        for strength in strengths:

            st.markdown(
                f"**{strength.get('requirement', 'Requirement')}**"
            )

            st.write(
                strength.get(
                    "evidence",
                    "No evidence provided.",
                )
            )

    else:

        st.info(
            "No specific strengths were identified."
        )

    # --------------------------------------------------------
    # REQUIREMENT ANALYSIS
    # --------------------------------------------------------

    st.subheader("📋 Requirement Analysis")

    requirements = review.get(
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

            with st.expander(
                f"{status} — {requirement}"
            ):

                st.markdown(
                    "**Resume evidence**"
                )

                st.write(
                    item.get(
                        "resume_evidence",
                        "No evidence provided.",
                    )
                )

                st.markdown(
                    "**Recommendation**"
                )

                st.write(
                    item.get(
                        "recommendation",
                        "No recommendation provided.",
                    )
                )

    else:

        st.info(
            "No requirement analysis was returned."
        )

    # --------------------------------------------------------
    # MISSING INFORMATION
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

            st.markdown(
                f"- {item}"
            )

    else:

        st.success(
            "No major missing or unclear requirements identified."
        )

    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    st.subheader(
        "✏️ Resume Improvement Recommendations"
    )

    improvements = review.get(
        "resume_improvements",
        [],
    )

    if improvements:

        for improvement in improvements:

            st.markdown(
                f"### {improvement.get('area', 'Resume')}"
            )

            st.write(
                improvement.get(
                    "recommendation",
                    "",
                )
            )

            example = improvement.get(
                "example",
                "",
            )

            if example:

                st.markdown(
                    "**Example:**"
                )

                st.info(example)

    else:

        st.info(
            "No specific improvement recommendations returned."
        )

    # --------------------------------------------------------
    # ATS KEYWORDS
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
            " ".join(
                f"`{keyword}`"
                for keyword in keywords
            )
        )

        st.caption(
            "Only add a keyword if it genuinely describes "
            "your experience."
        )

    else:

        st.info(
            "No additional keywords identified."
        )

    # --------------------------------------------------------
    # PRIORITY ACTIONS
    # --------------------------------------------------------

    st.subheader(
        "🚀 Priority Actions"
    )

    actions = review.get(
        "priority_actions",
        [],
    )

    if actions:

        for number, action in enumerate(
            actions,
            start=1,
        ):

            st.markdown(
                f"**{number}.** {action}"
            )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        review.get(
            "important_disclaimer",
            "This review is based only on the supplied resume "
            "and job description.",
        )
    )
