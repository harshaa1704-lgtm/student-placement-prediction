def analyze_skill_gap(
    detected_skills=None,
    communication_score=0,
    coding_score=0,
    resume_score=0,
    cgpa=0,
    dsa_problems=0,
    internships=0,
    projects=0
):
    """
    Skill Gap Analysis Module

    Identifies:
    - Strong skills
    - Weak areas
    - Missing skills
    - Improvement priorities
    """

    if detected_skills is None:
        detected_skills = []

    # ------------------------------------------------------
    # Normalize detected skills
    # ------------------------------------------------------

    skills = [
        str(skill).lower().strip()
        for skill in detected_skills
        if skill
    ]

    # ------------------------------------------------------
    # Required technical skills
    # ------------------------------------------------------

    required_skills = [
        "python",
        "sql",
        "machine learning",
        "data analysis",
        "statistics",
        "excel",
        "communication",
        "problem solving",
        "git"
    ]

    # ------------------------------------------------------
    # Find missing skills
    # ------------------------------------------------------

    missing_skills = []

    for skill in required_skills:

        found = False

        for detected in skills:

            if skill in detected or detected in skill:
                found = True
                break

        if not found:
            missing_skills.append(skill)

    # ------------------------------------------------------
    # Technical strengths
    # ------------------------------------------------------

    strong_skills = []

    for skill in required_skills:

        for detected in skills:

            if skill in detected or detected in skill:
                strong_skills.append(skill)
                break

    # ------------------------------------------------------
    # Weak areas
    # ------------------------------------------------------

    weak_areas = []

    if communication_score < 7:
        weak_areas.append("Communication")

    if coding_score < 7:
        weak_areas.append("Coding")

    if resume_score < 70:
        weak_areas.append("Resume Quality")

    if cgpa < 7.5:
        weak_areas.append("Academic Performance")

    if dsa_problems < 100:
        weak_areas.append("Data Structures & Algorithms")

    if internships < 1:
        weak_areas.append("Industry Experience")

    if projects < 2:
        weak_areas.append("Project Experience")

    # ------------------------------------------------------
    # Improvement priorities
    # ------------------------------------------------------

    improvement_priorities = []

    if communication_score < 7:
        improvement_priorities.append(
            "Improve communication and interview speaking skills."
        )

    if coding_score < 7:
        improvement_priorities.append(
            "Practice programming and problem-solving regularly."
        )

    if dsa_problems < 100:
        improvement_priorities.append(
            "Practice Data Structures and Algorithms."
        )

    if internships < 1:
        improvement_priorities.append(
            "Gain practical experience through internships."
        )

    if projects < 2:
        improvement_priorities.append(
            "Build more real-world technical projects."
        )

    if "python" not in skills:
        improvement_priorities.append(
            "Learn Python programming."
        )

    if "sql" not in skills:
        improvement_priorities.append(
            "Improve SQL and database skills."
        )

    if "machine learning" not in skills:
        improvement_priorities.append(
            "Learn Machine Learning fundamentals."
        )

    if "data analysis" not in skills:
        improvement_priorities.append(
            "Develop practical Data Analysis skills."
        )

    if "statistics" not in skills:
        improvement_priorities.append(
            "Strengthen statistics fundamentals."
        )

    if "git" not in skills:
        improvement_priorities.append(
            "Learn Git and version control."
        )

    # ------------------------------------------------------
    # Overall skill gap score
    # ------------------------------------------------------

    total_possible = len(required_skills)

    if total_possible > 0:

        skill_coverage = (
            len(strong_skills) / total_possible
        ) * 100

    else:

        skill_coverage = 0

    skill_coverage = round(skill_coverage, 1)

    # ------------------------------------------------------
    # Skill readiness level
    # ------------------------------------------------------

    if skill_coverage >= 80:

        readiness = "Excellent"

    elif skill_coverage >= 60:

        readiness = "Good"

    elif skill_coverage >= 40:

        readiness = "Average"

    else:

        readiness = "Needs Improvement"

    # ------------------------------------------------------
    # Return results
    # ------------------------------------------------------

    return {
        "Strong Skills": strong_skills,
        "Missing Skills": missing_skills,
        "Weak Areas": weak_areas,
        "Improvement Priorities": improvement_priorities,
        "Skill Coverage": skill_coverage,
        "Readiness Level": readiness
    }