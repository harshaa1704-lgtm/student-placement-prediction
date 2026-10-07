import re


def analyze_communication(answer):
    """
    Basic text-based communication assessment.

    Evaluates:
    - Grammar
    - Clarity
    - Vocabulary
    - Relevance
    - Sentence Structure
    """

    if not answer or not answer.strip():
        return {
            "Grammar": 0,
            "Clarity": 0,
            "Vocabulary": 0,
            "Relevance": 0,
            "Sentence Structure": 0,
            "Overall Score": 0,
            "Level": "Not Assessed"
        }

    text = answer.strip()

    # ======================================================
    # BASIC TEXT STATISTICS
    # ======================================================

    words = re.findall(r"\b[\w']+\b", text)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    # ======================================================
    # VOCABULARY
    # ======================================================

    unique_words = set(word.lower() for word in words)

    if word_count == 0:
        vocabulary = 0

    else:
        diversity = len(unique_words) / word_count

        if word_count >= 80 and diversity >= 0.55:
            vocabulary = 9

        elif word_count >= 50 and diversity >= 0.45:
            vocabulary = 8

        elif word_count >= 30 and diversity >= 0.35:
            vocabulary = 7

        elif word_count >= 15:
            vocabulary = 6

        else:
            vocabulary = 5

    # ======================================================
    # SENTENCE STRUCTURE
    # ======================================================

    if sentence_count >= 5:
        sentence_structure = 9

    elif sentence_count >= 4:
        sentence_structure = 8

    elif sentence_count >= 3:
        sentence_structure = 7

    elif sentence_count >= 2:
        sentence_structure = 6

    else:
        sentence_structure = 5

    # ======================================================
    # CLARITY
    # ======================================================

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0
    )

    if 8 <= average_sentence_length <= 25:
        clarity = 9

    elif 6 <= average_sentence_length <= 30:
        clarity = 8

    elif 4 <= average_sentence_length <= 35:
        clarity = 7

    else:
        clarity = 5

    # ======================================================
    # GRAMMAR INDICATORS
    # ======================================================

    grammar_score = 8

    # Excessive punctuation
    if re.search(r"[!?]{2,}", text):
        grammar_score -= 1

    # Lowercase "i"
    if re.search(r"\bi\b", text):
        grammar_score -= 1

    # Very long response with very few sentences
    if word_count > 80 and sentence_count <= 2:
        grammar_score -= 1

    grammar_score = max(1, min(10, grammar_score))

    # ======================================================
    # RELEVANCE
    # ======================================================

    relevant_keywords = [
        "education",
        "student",
        "project",
        "projects",
        "skill",
        "skills",
        "experience",
        "internship",
        "internships",
        "python",
        "java",
        "sql",
        "machine learning",
        "data",
        "analytics",
        "career",
        "goal",
        "team",
        "problem",
        "learning"
    ]

    text_lower = text.lower()

    keyword_count = sum(
        1
        for keyword in relevant_keywords
        if keyword in text_lower
    )

    if keyword_count >= 5:
        relevance = 9

    elif keyword_count >= 4:
        relevance = 8

    elif keyword_count >= 2:
        relevance = 7

    elif keyword_count >= 1:
        relevance = 6

    else:
        relevance = 5

    # ======================================================
    # OVERALL SCORE
    # ======================================================

    overall = (
        grammar_score
        + clarity
        + vocabulary
        + relevance
        + sentence_structure
    ) / 5

    overall = round(overall, 1)

    # ======================================================
    # COMMUNICATION LEVEL
    # ======================================================

    if overall >= 8.5:
        level = "Excellent"

    elif overall >= 7:
        level = "Good"

    elif overall >= 5.5:
        level = "Average"

    else:
        level = "Needs Improvement"

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {
        "Grammar": grammar_score,
        "Clarity": clarity,
        "Vocabulary": vocabulary,
        "Relevance": relevance,
        "Sentence Structure": sentence_structure,
        "Overall Score": overall,
        "Level": level
    }