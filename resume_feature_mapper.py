def map_resume_to_features(resume_data):
    """
    Convert extracted resume information into
    values that can be used for placement prediction.

    Missing information is returned as None.
    """

    mapped_data = {

        # Academic
        "CGPA": None,

        # Technical / Experience
        "Projects_Count": resume_data.get("Projects"),
        "Internships": resume_data.get("Internships"),
        "Certifications": resume_data.get("Certifications"),

        # GitHub
        "GitHub_Contributions": None,

        # Hackathons
        "Hackathons_Participated": resume_data.get("Hackathons"),

        # Skills
        "Programming_Skills": resume_data.get(
            "Programming Skills", []
        ),

        # These cannot be safely determined from a resume
        "Gender": None,
        "College_Tier": None,
        "Specialization": None,
        "SSC_Marks": None,
        "HSC_Marks": None,
        "DSA_Problems_Solved": None,
        "Communication_Skills": None,
        "Aptitude_Test_Score": None,
        "LeetCode_Rating": None,
        "AI_ML_Skill_Level": None,
        "System_Design_Knowledge": None,
        "Resume_Score": None,
        "Mock_Interview_Score": None,
        "SoftSkillsRating": None,
        "Study_Hours_Per_Day": None,
        "Coding_Skill_Score": None
    }

    # Convert CGPA if available
    cgpa = resume_data.get("CGPA")

    if cgpa != "Not detected":
        try:
            mapped_data["CGPA"] = float(cgpa)
        except:
            mapped_data["CGPA"] = None

    return mapped_data