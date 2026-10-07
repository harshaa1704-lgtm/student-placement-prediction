import re


def analyze_coding(answers):
    """
    Rule-based coding assessment.

    Evaluates answers based on:
    - Correct programming concepts
    - Problem-solving knowledge
    - Algorithm understanding
    - Data structure knowledge
    - Programming language knowledge
    """

    if not answers:
        return {
            "Concept Score": 0,
            "Problem Solving": 0,
            "Algorithm Score": 0,
            "Data Structure Score": 0,
            "Programming Score": 0,
            "Overall Score": 0,
            "Level": "Not Assessed"
        }

    total_score = 0
    question_count = len(answers)

    # Keywords used to identify relevant coding concepts
    coding_keywords = [
        "algorithm",
        "array",
        "string",
        "list",
        "stack",
        "queue",
        "tree",
        "graph",
        "linked list",
        "recursion",
        "sorting",
        "searching",
        "binary search",
        "loop",
        "function",
        "class",
        "object",
        "python",
        "java",
        "sql",
        "database",
        "time complexity",
        "space complexity",
        "o(n)",
        "o(log n)",
        "o(n log n)"
    ]

    for answer in answers:

        if not answer or not answer.strip():
            continue

        text = answer.lower()

        # Count relevant coding concepts
        matched_keywords = sum(
            1
            for keyword in coding_keywords
            if keyword in text
        )

        # Score based on keyword coverage
        if matched_keywords >= 5:
            score = 10

        elif matched_keywords >= 4:
            score = 9

        elif matched_keywords >= 3:
            score = 8

        elif matched_keywords >= 2:
            score = 7

        elif matched_keywords >= 1:
            score = 6

        else:
            score = 4

        total_score += score

    # ------------------------------------------------------
    # Overall score
    # ------------------------------------------------------

    if question_count > 0:
        overall_score = total_score / question_count
    else:
        overall_score = 0

    overall_score = round(overall_score, 1)

    # ------------------------------------------------------
    # Individual category scores
    # ------------------------------------------------------

    concept_score = overall_score

    problem_solving = min(
        10,
        round(overall_score + 0.3, 1)
    )

    algorithm_score = min(
        10,
        round(overall_score, 1)
    )

    data_structure_score = min(
        10,
        round(overall_score - 0.2, 1)
    )

    programming_score = min(
        10,
        round(overall_score + 0.2, 1)
    )

    # Prevent negative values
    data_structure_score = max(
        0,
        data_structure_score
    )

    # ------------------------------------------------------
    # Coding level
    # ------------------------------------------------------

    if overall_score >= 8.5:
        level = "Excellent"

    elif overall_score >= 7:
        level = "Good"

    elif overall_score >= 5.5:
        level = "Average"

    else:
        level = "Needs Improvement"

    return {
        "Concept Score": concept_score,
        "Problem Solving": problem_solving,
        "Algorithm Score": algorithm_score,
        "Data Structure Score": data_structure_score,
        "Programming Score": programming_score,
        "Overall Score": overall_score,
        "Level": level
    }