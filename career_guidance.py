"""
Career Guidance
----------------
Combines resume, communication, coding and skill-gap results
to generate an overall career-readiness assessment.
"""


def generate_career_guidance(
    resume_score=0,
    communication_score=0,
    coding_score=0,
    skill_coverage=0,
    placement_probability=0
):
    """
    Generate overall career guidance and readiness level.
    """

    # ------------------------------------------------------
    # Convert values safely
    # ------------------------------------------------------

    resume_score = _safe_score(resume_score)
    communication_score = _safe_score(communication_score)
    coding_score = _safe_score(coding_score)
    skill_coverage = _safe_score(skill_coverage)
    placement_probability = _safe_score(placement_probability)

    # ------------------------------------------------------
    # Overall career readiness
    # ------------------------------------------------------

    overall_score = (
        resume_score * 0.20 +
        communication_score * 0.20 +
        coding_score * 0.25 +
        skill_coverage * 0.20 +
        placement_probability * 0.15
    )

    overall_score = round(overall_score, 2)

    # ------------------------------------------------------
    # Readiness level
    # ------------------------------------------------------

    if overall_score >= 80:
        readiness = "Excellent"

    elif overall_score >= 65:
        readiness = "Good"

    elif overall_score >= 50:
        readiness = "Average"

    else:
        readiness = "Needs Improvement"

    # ------------------------------------------------------
    # Strengths
    # ------------------------------------------------------

    strengths = []

    if resume_score >= 70:
        strengths.append("Strong resume profile")

    if communication_score >= 70:
        strengths.append("Good communication skills")

    if coding_score >= 70:
        strengths.append("Good coding and problem-solving skills")

    if skill_coverage >= 70:
        strengths.append("Good technical skill coverage")

    if placement_probability >= 70:
        strengths.append("High placement probability")

    # ------------------------------------------------------
    # Areas for improvement
    # ------------------------------------------------------

    improvements = []

    if resume_score < 70:
        improvements.append("Improve resume quality")

    if communication_score < 70:
        improvements.append("Improve communication skills")

    if coding_score < 70:
        improvements.append("Practice coding and DSA")

    if skill_coverage < 70:
        improvements.append("Develop missing technical skills")

    if placement_probability < 70:
        improvements.append("Improve overall placement readiness")

    # ------------------------------------------------------
    # Career advice
    # ------------------------------------------------------

    career_advice = []

    if coding_score >= 75 and skill_coverage >= 70:
        career_advice.append(
            "Consider software development, data engineering, "
            "or other technical roles."
        )

    if communication_score >= 75:
        career_advice.append(
            "Your communication skills support roles involving "
            "teamwork, presentations and client interaction."
        )

    if coding_score >= 65 and skill_coverage >= 60:
        career_advice.append(
            "Data analyst, junior data scientist and Python-based "
            "technical roles may be suitable career options."
        )

    if not career_advice:
        career_advice.append(
            "Focus on improving your technical skills, communication, "
            "resume and interview preparation before applying widely."
        )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    return {
        "overall_score": overall_score,
        "readiness_level": readiness,
        "strengths": strengths,
        "areas_for_improvement": improvements,
        "career_advice": career_advice
    }


# ==========================================================
# SAFE SCORE FUNCTION
# ==========================================================

def _safe_score(value):
    """
    Convert a value to a valid score between 0 and 100.
    """

    try:
        value = float(value)
    except (ValueError, TypeError):
        return 0.0

    return max(0.0, min(100.0, value))