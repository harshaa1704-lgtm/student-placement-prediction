"""
Resume Analyzer
---------------
Phase-II module for Student Placement Prediction System.

This module analyzes extracted resume text and generates:
    1. Academic analysis
    2. Technical / coding analysis
    3. Communication analysis
    4. Resume score
    5. Skill gap analysis
    6. Personalized recommendations
    7. Detected skills and keywords

IMPORTANT:
This module does NOT modify the Phase-I XGBoost model.
It works independently on extracted resume text.
"""

import re


# ==========================================================
# SKILL DATABASE
# ==========================================================

PROGRAMMING_LANGUAGES = {
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "r",
    "sql",
    "php",
    "kotlin",
    "swift",
    "go",
    "scala"
}

WEB_TECHNOLOGIES = {
    "html",
    "css",
    "javascript",
    "react",
    "angular",
    "node.js",
    "nodejs",
    "express",
    "bootstrap",
    "tailwind"
}

DATA_SKILLS = {
    "python",
    "r",
    "sql",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "power bi",
    "tableau",
    "excel",
    "statistics",
    "data analysis",
    "data visualization"
}

AI_ML_SKILLS = {
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "artificial intelligence",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "sklearn",
    "xgboost",
    "random forest",
    "nlp",
    "natural language processing",
    "computer vision"
}

DATABASE_SKILLS = {
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",
    "sqlite",
    "sql server",
    "firebase"
}

CLOUD_SKILLS = {
    "aws",
    "azure",
    "google cloud",
    "gcp",
    "docker",
    "kubernetes"
}

TOOLS = {
    "git",
    "github",
    "gitlab",
    "jupyter",
    "vscode",
    "visual studio",
    "streamlit",
    "flask",
    "django"
}

COMMUNICATION_KEYWORDS = {
    "communication",
    "presentation",
    "public speaking",
    "leadership",
    "teamwork",
    "team player",
    "collaboration",
    "volunteer",
    "workshop",
    "seminar",
    "conference",
    "event",
    "organizing",
    "coordinator",
    "mentoring",
    "mentor",
    "interpersonal"
}

PROJECT_KEYWORDS = {
    "project",
    "projects",
    "developed",
    "implemented",
    "designed",
    "built",
    "created",
    "application",
    "system",
    "website",
    "dashboard"
}

INTERNSHIP_KEYWORDS = {
    "internship",
    "intern",
    "trainee",
    "industrial training",
    "work experience"
}

CERTIFICATION_KEYWORDS = {
    "certification",
    "certified",
    "certificate",
    "course",
    "udemy",
    "coursera",
    "nptel",
    "linkedin learning"
}

CODING_KEYWORDS = {
    "coding",
    "programming",
    "dsa",
    "data structures",
    "algorithms",
    "competitive programming",
    "leetcode",
    "hackerrank",
    "codechef",
    "coding problem"
}


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize_text(text):
    """
    Clean and normalize extracted resume text.
    """

    if not text:
        return ""

    text = str(text)

    # Normalize spaces
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# ==========================================================
# KEYWORD DETECTION
# ==========================================================

def find_keywords(text, keywords):
    """
    Find keywords present in resume text.
    """

    text_lower = text.lower()

    found = []

    for keyword in keywords:
        pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
        if re.search(pattern, text_lower):
            found.append(keyword)

    return sorted(found)


# ==========================================================
# COUNT KEYWORD OCCURRENCES
# ==========================================================

def count_keywords(text, keywords):
    """
    Count how many relevant keywords occur in the resume.
    """

    text_lower = text.lower()

    count = 0

    for keyword in keywords:

        pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"

        matches = re.findall(pattern, text_lower)

        count += len(matches)

    return count


# ==========================================================
# EXTRACT EDUCATION
# ==========================================================

def extract_education(text):
    """
    Identify common educational qualifications.
    """

    education_patterns = {
        "M.Sc": r"\bm\.?\s*sc\.?\b",
        "MCA": r"\bmca\b",
        "MBA": r"\bmba\b",
        "M.Tech": r"\bm\.?\s*tech\.?\b",
        "B.Sc": r"\bb\.?\s*sc\.?\b",
        "BCA": r"\bbca\b",
        "B.Tech": r"\bb\.?\s*tech\.?\b",
        "BE": r"\bb\.?\s*e\.?\b",
        "BA": r"\bb\.?\s*a\.?\b",
        "B.Com": r"\bb\.?\s*com\.?\b",
        "HSC": r"\bhsc\b",
        "12th": r"\b12(th)?\b",
        "SSLC": r"\bsslc\b",
        "10th": r"\b10(th)?\b"
    }

    found = []

    text_lower = text.lower()

    for qualification, pattern in education_patterns.items():

        if re.search(pattern, text_lower):
            found.append(qualification)

    return found


# ==========================================================
# EXTRACT CGPA / PERCENTAGE
# ==========================================================

def extract_academic_scores(text):
    """
    Extract CGPA and percentage values when available.
    """

    cgpa_values = []
    percentage_values = []

    # CGPA patterns
    cgpa_patterns = [
        r"cgpa\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        r"gpa\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in cgpa_patterns:

        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        for value in matches:

            try:
                number = float(value)

                if 0 <= number <= 10:
                    cgpa_values.append(number)

            except ValueError:
                pass

    # Percentage patterns
    percentage_patterns = [
        r"(\d+(?:\.\d+)?)\s*%",
        r"percentage\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in percentage_patterns:

        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        for value in matches:

            try:
                number = float(value)

                if 0 <= number <= 100:
                    percentage_values.append(number)

            except ValueError:
                pass

    return {
        "cgpa": cgpa_values,
        "percentage": percentage_values
    }


# ==========================================================
# SECTION DETECTION
# ==========================================================

def detect_sections(text):
    """
    Detect common resume sections.
    """

    text_lower = text.lower()

    sections = {
        "objective": [
            "objective",
            "career objective",
            "profile"
        ],

        "education": [
            "education",
            "academic credentials",
            "academic qualification",
            "educational qualification"
        ],

        "skills": [
            "skills",
            "technical skills",
            "technical knowledge"
        ],

        "projects": [
            "projects",
            "academic projects",
            "project"
        ],

        "experience": [
            "experience",
            "work experience",
            "employment"
        ],

        "internships": [
            "internship",
            "internships"
        ],

        "certifications": [
            "certification",
            "certifications",
            "certificates"
        ],

        "achievements": [
            "achievement",
            "achievements",
            "awards"
        ]
    }

    detected = {}

    for section, keywords in sections.items():

        detected[section] = any(
            keyword in text_lower
            for keyword in keywords
        )

    return detected


# ==========================================================
# ACADEMIC ANALYSIS
# ==========================================================

def analyze_academic(text):
    """
    Analyze academic information.
    """

    education = extract_education(text)
    scores = extract_academic_scores(text)

    score = 40

    if education:
        score += 25

    if scores["cgpa"]:
        score += 20

    if scores["percentage"]:
        score += 15

    score = min(score, 100)

    strengths = []
    improvements = []

    if education:
        strengths.append(
            "Educational qualification is clearly mentioned."
        )
    else:
        improvements.append(
            "Add complete educational qualifications."
        )

    if scores["cgpa"]:
        strengths.append(
            "CGPA/GPA information was detected."
        )

    if scores["percentage"]:
        strengths.append(
            "Academic percentage information was detected."
        )

    if not scores["cgpa"] and not scores["percentage"]:
        improvements.append(
            "Consider adding CGPA or academic percentage."
        )

    return {
        "score": score,
        "education": education,
        "cgpa": scores["cgpa"],
        "percentage": scores["percentage"],
        "strengths": strengths,
        "improvements": improvements
    }


# ==========================================================
# CODING ANALYSIS
# ==========================================================

def analyze_coding(text):
    """
    Analyze technical and coding knowledge.
    """

    programming = find_keywords(
        text,
        PROGRAMMING_LANGUAGES
    )

    web = find_keywords(
        text,
        WEB_TECHNOLOGIES
    )

    data = find_keywords(
        text,
        DATA_SKILLS
    )

    ai_ml = find_keywords(
        text,
        AI_ML_SKILLS
    )

    databases = find_keywords(
        text,
        DATABASE_SKILLS
    )

    cloud = find_keywords(
        text,
        CLOUD_SKILLS
    )

    tools = find_keywords(
        text,
        TOOLS
    )

    coding_terms = find_keywords(
        text,
        CODING_KEYWORDS
    )

    total_skill_groups = sum([
        bool(programming),
        bool(web),
        bool(data),
        bool(ai_ml),
        bool(databases),
        bool(cloud),
        bool(tools),
        bool(coding_terms)
    ])

    score = 20 + (total_skill_groups * 10)

    if len(programming) >= 2:
        score += 10

    if len(ai_ml) >= 2:
        score += 5

    if len(data) >= 3:
        score += 5

    score = min(score, 100)

    strengths = []
    improvements = []

    if programming:
        strengths.append(
            "Programming languages detected: "
            + ", ".join(programming)
        )
    else:
        improvements.append(
            "Add programming languages such as Python, Java or C++."
        )

    if data:
        strengths.append(
            "Data-related skills detected."
        )

    if ai_ml:
        strengths.append(
            "AI/ML knowledge detected."
        )

    if len(coding_terms) > 0:
        strengths.append(
            "Coding/DSA-related activities detected."
        )
    else:
        improvements.append(
            "Add DSA, coding practice or problem-solving experience."
        )

    if not tools:
        improvements.append(
            "Add development tools such as Git and GitHub."
        )

    return {
        "score": score,
        "programming_languages": programming,
        "web_technologies": web,
        "data_skills": data,
        "ai_ml_skills": ai_ml,
        "database_skills": databases,
        "cloud_skills": cloud,
        "tools": tools,
        "coding_keywords": coding_terms,
        "strengths": strengths,
        "improvements": improvements
    }


# ==========================================================
# COMMUNICATION ANALYSIS
# ==========================================================

def analyze_communication(text):
    """
    Estimate communication/soft-skill evidence
    from resume content.
    """

    communication = find_keywords(
        text,
        COMMUNICATION_KEYWORDS
    )

    project_terms = find_keywords(
        text,
        PROJECT_KEYWORDS
    )

    score = 35

    score += min(len(communication) * 8, 40)

    if project_terms:
        score += 10

    if len(text.split()) > 300:
        score += 5

    score = min(score, 100)

    strengths = []
    improvements = []

    if communication:
        strengths.append(
            "Communication and soft-skill evidence detected."
        )

    if "presentation" in communication:
        strengths.append(
            "Presentation experience detected."
        )

    if "leadership" in communication:
        strengths.append(
            "Leadership experience detected."
        )

    if "teamwork" in communication or "collaboration" in communication:
        strengths.append(
            "Teamwork/collaboration experience detected."
        )

    if not communication:
        improvements.append(
            "Add leadership, teamwork, presentation or communication activities."
        )

    return {
        "score": score,
        "communication_keywords": communication,
        "strengths": strengths,
        "improvements": improvements
    }


# ==========================================================
# PROJECT ANALYSIS
# ==========================================================

def analyze_projects(text):
    """
    Estimate project experience.
    """

    text_lower = text.lower()

    project_count = 0

    # Try numbered project patterns
    numbered = re.findall(
        r"(?:project\s*)?(?:\d+[\.\):]|[-])",
        text_lower
    )

    project_count = len(numbered)

    # If numbered detection fails, use project section evidence
    if project_count == 0:

        project_mentions = len(
            re.findall(r"\bprojects?\b", text_lower)
        )

        if project_mentions >= 3:
            project_count = 3

        elif project_mentions == 2:
            project_count = 2

        elif project_mentions == 1:
            project_count = 1

    score = min(40 + project_count * 15, 100)

    strengths = []
    improvements = []

    if project_count > 0:
        strengths.append(
            f"Approximately {project_count} project-related entries detected."
        )
    else:
        improvements.append(
            "Add at least 2–3 technical projects."
        )

    return {
        "score": score,
        "project_count": project_count,
        "strengths": strengths,
        "improvements": improvements
    }


# ==========================================================
# EXPERIENCE ANALYSIS
# ==========================================================

def analyze_experience(text):
    """
    Analyze internship/work experience.
    """

    internship_keywords = find_keywords(
        text,
        INTERNSHIP_KEYWORDS
    )

    certification_keywords = find_keywords(
        text,
        CERTIFICATION_KEYWORDS
    )

    internship_count = count_keywords(
        text,
        {"internship", "intern"}
    )

    score = 35

    if internship_keywords:
        score += 30

    if internship_count >= 2:
        score += 15

    if certification_keywords:
        score += 10

    score = min(score, 100)

    strengths = []
    improvements = []

    if internship_keywords:
        strengths.append(
            "Internship/work experience detected."
        )
    else:
        improvements.append(
            "Consider adding internship or practical industry experience."
        )

    if certification_keywords:
        strengths.append(
            "Certification/course-related information detected."
        )
    else:
        improvements.append(
            "Add relevant certifications or industry courses."
        )

    return {
        "score": score,
        "internship_keywords": internship_keywords,
        "certification_keywords": certification_keywords,
        "strengths": strengths,
        "improvements": improvements
    }


# ==========================================================
# SKILL GAP ANALYSIS
# ==========================================================

def skill_gap_analysis(coding_analysis, communication_analysis, project_analysis):
    """
    Identify missing or weak areas.
    """

    gaps = []

    coding_score = coding_analysis["score"]
    communication_score = communication_analysis["score"]
    project_score = project_analysis["score"]

    if coding_score < 60:
        gaps.append(
            "Technical/Coding Skills"
        )

    if communication_score < 60:
        gaps.append(
            "Communication & Soft Skills"
        )

    if project_score < 65:
        gaps.append(
            "Project Experience"
        )

    if not coding_analysis["programming_languages"]:
        gaps.append(
            "Programming Language"
        )

    if not coding_analysis["coding_keywords"]:
        gaps.append(
            "DSA / Problem Solving"
        )

    if not coding_analysis["tools"]:
        gaps.append(
            "Git / Development Tools"
        )

    if not coding_analysis["data_skills"]:
        gaps.append(
            "Data Analytics"
        )

    if not coding_analysis["ai_ml_skills"]:
        gaps.append(
            "AI / Machine Learning"
        )

    return sorted(set(gaps))


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

def generate_recommendations(
    academic,
    coding,
    communication,
    projects,
    experience,
    skill_gaps
):
    """
    Generate personalized recommendations.
    """

    recommendations = []

    # Coding
    if coding["score"] < 70:
        recommendations.append(
            "Improve coding skills by practising Python, Java or C++ regularly."
        )

    if not coding["coding_keywords"]:
        recommendations.append(
            "Start DSA and problem-solving practice using coding platforms."
        )

    # Data
    if not coding["data_skills"]:
        recommendations.append(
            "Learn SQL, Pandas, NumPy and data visualization for data analytics roles."
        )

    # AI/ML
    if not coding["ai_ml_skills"]:
        recommendations.append(
            "Learn basic Machine Learning concepts and build one ML project."
        )

    # GitHub
    if not coding["tools"]:
        recommendations.append(
            "Create a GitHub profile and upload your academic projects."
        )

    # Projects
    if projects["score"] < 70:
        recommendations.append(
            "Build 2–3 practical projects related to your target job role."
        )

    # Communication
    if communication["score"] < 65:
        recommendations.append(
            "Improve communication through presentations, mock interviews and group discussions."
        )

    # Internship
    if not experience["internship_keywords"]:
        recommendations.append(
            "Consider completing an internship or practical industry project."
        )

    # Certification
    if not experience["certification_keywords"]:
        recommendations.append(
            "Add relevant professional certifications to strengthen your resume."
        )

    # Academic
    if not academic["cgpa"] and not academic["percentage"]:
        recommendations.append(
            "Include your CGPA or academic percentage in the resume."
        )

    # Skill gaps
    if skill_gaps:
        recommendations.append(
            "Priority skill gaps: " + ", ".join(skill_gaps[:5])
        )

    # Limit recommendations
    return recommendations[:10]


# ==========================================================
# OVERALL RESUME SCORE
# ==========================================================

def calculate_resume_score(
    academic,
    coding,
    communication,
    projects,
    experience
):
    """
    Calculate overall resume score.
    """

    score = (
        academic["score"] * 0.20
        + coding["score"] * 0.30
        + communication["score"] * 0.20
        + projects["score"] * 0.15
        + experience["score"] * 0.15
    )

    return round(score, 2)


# ==========================================================
# MAIN ANALYZER
# ==========================================================

def analyze_resume(resume_text):
    """
    Main function used by the Streamlit application.

    Parameters
    ----------
    resume_text : str
        Text extracted from uploaded resume.

    Returns
    -------
    dict
        Complete resume analysis.
    """

    # ------------------------------------------------------
    # Validate input
    # ------------------------------------------------------

    if not resume_text or not str(resume_text).strip():

        return {
            "success": False,
            "message": "No resume text was provided.",
            "resume_score": 0,
            "academic": {},
            "coding": {},
            "communication": {},
            "projects": {},
            "experience": {},
            "skill_gaps": [],
            "recommendations": []
        }

    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------

    text = normalize_text(resume_text)

    # ------------------------------------------------------
    # Individual analyses
    # ------------------------------------------------------

    academic = analyze_academic(text)

    coding = analyze_coding(text)

    communication = analyze_communication(text)

    projects = analyze_projects(text)

    experience = analyze_experience(text)

    # ------------------------------------------------------
    # Skill gaps
    # ------------------------------------------------------

    skill_gaps = skill_gap_analysis(
        coding,
        communication,
        projects
    )

    # ------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------

    recommendations = generate_recommendations(
        academic,
        coding,
        communication,
        projects,
        experience,
        skill_gaps
    )

    # ------------------------------------------------------
    # Overall score
    # ------------------------------------------------------

    resume_score = calculate_resume_score(
        academic,
        coding,
        communication,
        projects,
        experience
    )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    return {
        "success": True,

        "resume_score": resume_score,

        "academic": academic,

        "coding": coding,

        "communication": communication,

        "projects": projects,

        "experience": experience,

        "skill_gaps": skill_gaps,

        "recommendations": recommendations,

        "detected_sections": detect_sections(text),

        "word_count": len(text.split()),

        "character_count": len(text)
    }


# ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    sample_resume = """
    Harsha Priya

    M.Sc Data Analytics
    B.Sc Computer Science

    Skills:
    Python, SQL, Pandas, NumPy, Machine Learning,
    XGBoost, Power BI, Git, GitHub

    Projects:
    Student Placement Prediction System
    Resume Analysis System

    Internship:
    Data Analytics Intern

    Certifications:
    Python Certification
    Machine Learning Certification

    Activities:
    Presentation, teamwork, communication and leadership.
    """

    result = analyze_resume(sample_resume)

    print("\n========================================")
    print("RESUME ANALYSIS")
    print("========================================")

    print("Resume Score:", result["resume_score"])

    print("\nAcademic:")
    print(result["academic"])

    print("\nCoding:")
    print(result["coding"])

    print("\nCommunication:")
    print(result["communication"])

    print("\nProjects:")
    print(result["projects"])

    print("\nExperience:")
    print(result["experience"])

    print("\nSkill Gaps:")
    print(result["skill_gaps"])

    print("\nRecommendations:")

    for recommendation in result["recommendations"]:
        print("-", recommendation)