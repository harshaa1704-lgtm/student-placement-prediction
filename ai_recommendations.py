# ==========================================================
# MAIN AI RECOMMENDATION FUNCTION
# ==========================================================

def generate_recommendations(
    resume_analysis=None,
    communication_analysis=None,
    coding_analysis=None,
    skill_gap_analysis=None
):
    """
    Generate personalized AI-style career recommendations.

    Parameters
    ----------
    resume_analysis : dict
        Resume analysis results.

    communication_analysis : dict
        Communication assessment results.

    coding_analysis : dict
        Coding assessment results.

    skill_gap_analysis : dict
        Skill gap analysis results.

    Returns
    -------
    dict
        Personalized recommendations.
    """

    # ------------------------------------------------------
    # SAFETY: Convert None values to empty dictionaries
    # ------------------------------------------------------

    resume_analysis = resume_analysis or {}
    communication_analysis = communication_analysis or {}
    coding_analysis = coding_analysis or {}
    skill_gap_analysis = skill_gap_analysis or {}

    recommendations = []
    priorities = []
    strengths = []
    career_guidance = []

    # ======================================================
    # 1. RESUME ANALYSIS
    # ======================================================

    resume_score = _get_score(
        resume_analysis,
        [
            "resume_score",
            "score",
            "overall_score"
        ]
    )

    if resume_score is not None:

        if resume_score < 50:
            recommendations.append(
                "Improve your resume structure, content, and presentation. "
                "Add relevant technical skills, projects, certifications, "
                "and measurable achievements."
            )
            priorities.append("High")

        elif resume_score < 70:
            recommendations.append(
                "Your resume has a reasonable foundation, but it can be "
                "strengthened by adding more relevant projects, technical "
                "skills, certifications, and quantified achievements."
            )
            priorities.append("Medium")

        else:
            strengths.append(
                "Your resume has a good overall profile."
            )

    # ======================================================
    # 2. COMMUNICATION ANALYSIS
    # ======================================================

    communication_score = _get_score(
        communication_analysis,
        [
            "communication_score",
            "score",
            "overall_score"
        ]
    )

    if communication_score is not None:

        if communication_score < 50:

            recommendations.append(
                "Focus on improving communication skills through daily "
                "English speaking practice, mock interviews, reading, "
                "and group discussions."
            )

            priorities.append("High")

        elif communication_score < 70:

            recommendations.append(
                "Practice speaking clearly and confidently. "
                "Regular mock interviews and group discussions can "
                "help improve your communication."
            )

            priorities.append("Medium")

        else:

            strengths.append(
                "You demonstrate good communication ability."
            )

    # ======================================================
    # 3. CODING ANALYSIS
    # ======================================================

    coding_score = _get_score(
        coding_analysis,
        [
            "coding_score",
            "score",
            "overall_score"
        ]
    )

    if coding_score is not None:

        if coding_score < 50:

            recommendations.append(
                "Strengthen your programming fundamentals. "
                "Practice Python or another programming language, "
                "data structures, algorithms, and basic problem solving."
            )

            priorities.append("High")

        elif coding_score < 70:

            recommendations.append(
                "Increase your coding practice by solving programming "
                "problems regularly and focusing on data structures "
                "and algorithms."
            )

            priorities.append("Medium")

        else:

            strengths.append(
                "Your coding foundation is strong."
            )

    # ======================================================
    # 4. SKILL GAP ANALYSIS
    # ======================================================

    missing_skills = _extract_missing_skills(skill_gap_analysis)

    if missing_skills:

        for skill in missing_skills[:8]:

            recommendations.append(
                f"Improve your {skill} skills through structured learning, "
                f"hands-on projects, and practical problem solving."
            )

        priorities.append("High")

    # ======================================================
    # 5. PROJECT RECOMMENDATION
    # ======================================================

    recommendations.append(
        "Build at least 2–3 practical projects related to your target "
        "career role and showcase them clearly on your resume and GitHub."
    )

    # ======================================================
    # 6. INTERNSHIP RECOMMENDATION
    # ======================================================

    recommendations.append(
        "Gain practical industry exposure through internships, "
        "freelance projects, hackathons, or real-world applications."
    )

    # ======================================================
    # 7. INTERVIEW PREPARATION
    # ======================================================

    recommendations.append(
        "Practice technical interviews, HR questions, aptitude tests, "
        "and mock interviews regularly to improve placement readiness."
    )

    # ======================================================
    # 8. CAREER GUIDANCE
    # ======================================================

    if coding_score is not None and coding_score >= 70:

        career_guidance.append(
            "You may consider software development, data engineering, "
            "or technical roles that require strong programming skills."
        )

    if communication_score is not None and communication_score >= 70:

        career_guidance.append(
            "Your communication skills can support roles involving "
            "client interaction, teamwork, presentations, and leadership."
        )

    if resume_score is not None and resume_score >= 70:

        career_guidance.append(
            "Your resume profile appears suitable for applying to "
            "internships and entry-level placement opportunities."
        )

    # Default career guidance

    if not career_guidance:

        career_guidance.append(
            "Focus on strengthening your technical skills, communication, "
            "projects, and interview preparation before applying widely."
        )

    # ======================================================
    # 9. REMOVE DUPLICATE RECOMMENDATIONS
    # ======================================================

    recommendations = _remove_duplicates(recommendations)

    strengths = _remove_duplicates(strengths)

    career_guidance = _remove_duplicates(career_guidance)

    # ======================================================
    # 10. PRIORITY CALCULATION
    # ======================================================

    if "High" in priorities:
        overall_priority = "High"

    elif "Medium" in priorities:
        overall_priority = "Medium"

    else:
        overall_priority = "Low"

    # ======================================================
    # 11. FINAL RESULT
    # ======================================================

    return {
        "recommendations": recommendations,
        "strengths": strengths,
        "career_guidance": career_guidance,
        "priority": overall_priority,
        "missing_skills": missing_skills,
        "total_recommendations": len(recommendations)
    }


# ==========================================================
# HELPER FUNCTION — GET SCORE
# ==========================================================

def _get_score(data, possible_keys):
    """
    Safely extract a numerical score from a dictionary.
    """

    if not isinstance(data, dict):
        return None

    for key in possible_keys:

        value = data.get(key)

        if value is not None:

            try:
                return float(value)

            except (ValueError, TypeError):
                continue

    return None


# ==========================================================
# HELPER FUNCTION — EXTRACT MISSING SKILLS
# ==========================================================

def _extract_missing_skills(data):
    """
    Extract missing skills from skill-gap analysis.
    """

    if not isinstance(data, dict):
        return []

    possible_keys = [
        "missing_skills",
        "skill_gaps",
        "gaps",
        "missing"
    ]

    for key in possible_keys:

        value = data.get(key)

        if value is None:
            continue

        if isinstance(value, list):

            return [
                str(skill)
                for skill in value
                if str(skill).strip()
            ]

        if isinstance(value, tuple):

            return [
                str(skill)
                for skill in value
                if str(skill).strip()
            ]

        if isinstance(value, str):

            return [
                skill.strip()
                for skill in value.split(",")
                if skill.strip()
            ]

    return []


# ==========================================================
# HELPER FUNCTION — REMOVE DUPLICATES
# ==========================================================

def _remove_duplicates(items):
    """
    Remove duplicate strings while preserving order.
    """

    result = []

    for item in items:

        if item not in result:
            result.append(item)

    return result