import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys
import os
import plotly.express as px


# ==========================================================
# PROJECT PATH
# ==========================================================

BASE_DIR = Path(__file__).parent

sys.path.append(
    os.path.abspath(BASE_DIR)
)


# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================

from src.resume_extractor import extract_resume_text
from src.resume_analyzer import analyze_resume
from src.resume_feature_mapper import map_resume_to_features
from src.communication_assessment import analyze_communication
from src.coding_assessment import analyze_coding
from src.skill_gap_analysis import analyze_skill_gap
from src.ai_recommendations import generate_recommendations
from src.career_guidance import generate_career_guidance


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* ================================
       MAIN APPLICATION
    ================================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #141E30,
            #243B55
        );

        color: white;
    }


    /* ================================
       SIDEBAR
    ================================= */

    section[data-testid="stSidebar"] {

        background: rgba(
            255,
            255,
            255,
            0.08
        );

        backdrop-filter: blur(18px);
    }


    /* ================================
       GLASS CARD
    ================================= */

    .glass {

        background: rgba(
            255,
            255,
            255,
            0.08
        );

        border-radius: 20px;

        padding: 25px;

        border: 1px solid rgba(
            255,
            255,
            255,
            0.15
        );

        box-shadow:
            0 8px 30px
            rgba(0, 0, 0, 0.30);

        margin-bottom: 20px;
    }


    /* ================================
       TITLE
    ================================= */

    .big-title {

        font-size: 42px;

        font-weight: 700;

        text-align: center;

        color: white;

        margin-bottom: 10px;
    }


    .subtitle {

        text-align: center;

        font-size: 18px;

        color: #dddddd;

        margin-bottom: 30px;
    }


    /* ================================
       KPI CARDS
    ================================= */

    .metric-card {

        background:
            linear-gradient(
                135deg,
                #00c6ff,
                #0072ff
            );

        padding: 22px;

        border-radius: 20px;

        text-align: center;

        color: white;

        box-shadow:
            0 10px 30px
            rgba(0, 0, 0, 0.30);

        min-height: 120px;
    }


    .metric-card h4 {

        margin: 0;

        font-size: 16px;
    }


    .metric-card h2 {

        margin-top: 12px;

        font-size: 28px;
    }


    /* ================================
       BUTTONS
    ================================= */

    .stButton > button {

        width: 100%;

        background:
            linear-gradient(
                90deg,
                #00c6ff,
                #0072ff
            );

        color: white;

        font-size: 17px;

        border: none;

        padding: 12px;

        border-radius: 14px;

        font-weight: 600;
    }


    .stButton > button:hover {

        transform: scale(1.02);

        background:
            linear-gradient(
                90deg,
                #0072ff,
                #00c6ff
            );
    }


    /* ================================
       METRIC CONTAINER
    ================================= */

    div[data-testid="metric-container"] {

        background:
            rgba(
                255,
                255,
                255,
                0.08
            );

        border-radius: 15px;

        padding: 12px;
    }


    /* ================================
       HEADINGS
    ================================= */

    h1,
    h2,
    h3,
    h4 {

        color: white;
    }


    /* ================================
       DIVIDER
    ================================= */

    hr {

        border: 1px solid
        rgba(
            255,
            255,
            255,
            0.15
        );
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_DIR / "best_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    feature_names = joblib.load(
        MODEL_DIR / "feature_names.pkl"
    )

    return model, scaler, feature_names


model, scaler, feature_names = load_model()


# ==========================================================
# SESSION STATE
# ==========================================================

default_state = {

    "extracted_text": "",

    "resume_analysis": None,

    "resume_features": None,

    "ai_recommendations": None,

    "communication_result": None,

    "coding_result": None,

    "skill_gap_result": None,

    "career_result": None,

    "placement_probability": 0,

    "prediction": None,

    "not_probability": 0

}


for key, value in default_state.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px;
        ">

        <h1>🎓</h1>

        <h2>Placement AI</h2>

        <p>
        Student Placement Prediction System
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "📌 Navigation",

        [
            "🏠 Home",
            "🎓 Placement Prediction",
            "📄 Resume Analysis",
            "🗣️ Communication Assessment",
            "💻 Coding Assessment",
            "🧩 Skill Gap Analysis",
            "🤖 AI Recommendations",
            "🚀 Career Guidance"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        ### About

        **Student Placement Prediction**

        🤖 XGBoost Classifier

        📄 AI Resume Analysis

        🗣️ Communication Assessment

        💻 Coding Assessment

        🧩 Skill Gap Analysis

        🚀 Career Guidance

        ---

        **Technology**

        Python  
        Streamlit  
        XGBoost  
        Scikit-Learn  
        Plotly
        """
    )


# ==========================================================
# COMMON HEADER
# ==========================================================

st.markdown(
    """
    <div class="big-title">
        🎓 Student Placement Prediction
    </div>

    <div class="subtitle">
        AI-Powered Student Career & Placement Analysis
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# PAGE 1 — HOME
# ==========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="glass">

        <h2>👋 Welcome to Placement AI</h2>

        <p>
        This system predicts student placement chances
        and provides AI-powered career analysis.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # KPI
    # ------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="metric-card">

            <h4>Features</h4>

            <h2>22</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="metric-card">

            <h4>Algorithm</h4>

            <h2>XGBoost</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="metric-card">

            <h4>Prediction</h4>

            <h2>Realtime</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
            <div class="metric-card">

            <h4>AI Analysis</h4>

            <h2>Enabled</h2>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ------------------------------------------------------
    # SYSTEM MODULES
    # ------------------------------------------------------

    st.subheader("🚀 System Modules")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.info(
            """
            ### 🎓 Placement Prediction

            Predict placement probability
            using the trained XGBoost model.
            """
        )

    with m2:

        st.info(
            """
            ### 📄 Resume Analysis

            Extract and analyse resume
            information using AI.
            """
        )

    with m3:

        st.info(
            """
            ### 🗣️ Communication

            Evaluate grammar,
            clarity and vocabulary.
            """
        )

    m4, m5, m6 = st.columns(3)

    with m4:

        st.info(
            """
            ### 💻 Coding Assessment

            Evaluate programming,
            algorithms and data structures.
            """
        )

    with m5:

        st.info(
            """
            ### 🧩 Skill Gap

            Identify strong skills,
            missing skills and weak areas.
            """
        )

    with m6:

        st.info(
            """
            ### 🚀 Career Guidance

            Generate personalised
            career recommendations.
            """
        )


# ==========================================================
# PAGE 2 — PLACEMENT PREDICTION
# ==========================================================

elif page == "🎓 Placement Prediction":

    st.header("🎓 Placement Prediction")

    st.write(
        "Enter student academic and technical details "
        "to predict placement probability."
    )

    st.markdown("---")

    # ------------------------------------------------------
    # STUDENT INPUT
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("🎓 Academic Details")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        college_tier = st.selectbox(
            "College Tier",
            [1, 2, 3]
        )

        specialization = st.selectbox(
            "Specialization",
            [
                "CSE",
                "ECE",
                "EEE",
                "Mechanical",
                "Civil",
                "IT"
            ]
        )

        cgpa = st.slider(
            "CGPA",
            0.0,
            10.0,
            8.0,
            0.1
        )

        ssc = st.slider(
            "SSC Marks",
            0,
            100,
            85
        )

        hsc = st.slider(
            "HSC Marks",
            0,
            100,
            85
        )

        dsa = st.number_input(
            "DSA Problems Solved",
            min_value=0,
            value=200
        )

        internships = st.number_input(
            "Internships",
            min_value=0,
            value=1
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            value=3
        )

        projects = st.number_input(
            "Projects Count",
            min_value=0,
            value=3
        )

    with right:

        st.subheader("💻 Skills & Performance")

        communication = st.slider(
            "Communication Skills",
            1,
            10,
            7
        )

        aptitude = st.slider(
            "Aptitude Test Score",
            0,
            100,
            80
        )

        leetcode = st.number_input(
            "LeetCode Rating",
            min_value=0,
            value=1500
        )

        github = st.number_input(
            "GitHub Contributions",
            min_value=0,
            value=100
        )

        hackathons = st.number_input(
            "Hackathons Participated",
            min_value=0,
            value=2
        )

        aiml = st.slider(
            "AI/ML Skill Level",
            1,
            10,
            7
        )

        system_design = st.slider(
            "System Design Knowledge",
            1,
            10,
            6
        )

        resume = st.slider(
            "Resume Score",
            0,
            100,
            80
        )

        mock = st.slider(
            "Mock Interview Score",
            0,
            100,
            75
        )

        softskills = st.slider(
            "Soft Skills Rating",
            1,
            10,
            7
        )

        study = st.slider(
            "Study Hours Per Day",
            1,
            12,
            5
        )

        coding = st.slider(
            "Coding Skill Score",
            0,
            100,
            80
        )

    st.markdown("---")

    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------

    if st.button(
        "🚀 Predict Placement",
        key="placement_prediction_button"
    ):

        student = {

            "Gender": gender,

            "College_Tier": college_tier,

            "Specialization": specialization,

            "CGPA": cgpa,

            "SSC_Marks": ssc,

            "HSC_Marks": hsc,

            "DSA_Problems_Solved": dsa,

            "Internships": internships,

            "Certifications": certifications,

            "Projects_Count": projects,

            "Communication_Skills": communication,

            "Aptitude_Test_Score": aptitude,

            "LeetCode_Rating": leetcode,

            "GitHub_Contributions": github,

            "Hackathons_Participated": hackathons,

            "AI_ML_Skill_Level": aiml,

            "System_Design_Knowledge": system_design,

            "Resume_Score": resume,

            "Mock_Interview_Score": mock,

            "SoftSkillsRating": softskills,

            "Study_Hours_Per_Day": study,

            "Coding_Skill_Score": coding
        }

        input_df = pd.DataFrame(
            [student]
        )

        input_df = pd.get_dummies(
            input_df
        )

        input_df = input_df.reindex(
            columns=feature_names,
            fill_value=0
        )

        input_scaled = scaler.transform(
            input_df
        )

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0]

        placed_probability = (
            probability[1] * 100
        )

        not_probability = (
            probability[0] * 100
        )

        # SAVE RESULTS

        st.session_state[
            "prediction"
        ] = prediction

        st.session_state[
            "placement_probability"
        ] = placed_probability

        st.session_state[
            "not_probability"
        ] = not_probability

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Placement Prediction Result"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Placement Probability",
                f"{placed_probability:.2f}%"
            )

        with r2:

            if prediction == 1:

                st.metric(
                    "Prediction",
                    "PLACED ✅"
                )

            else:

                st.metric(
                    "Prediction",
                    "NOT PLACED ❌"
                )

        with r3:

            if placed_probability >= 80:

                confidence = "Very High"

            elif placed_probability >= 60:

                confidence = "High"

            elif placed_probability >= 40:

                confidence = "Medium"

            else:

                confidence = "Low"

            st.metric(
                "Confidence",
                confidence
            )

        st.markdown("---")

        if prediction == 1:

            st.success(
                "🎉 Congratulations! "
                "The student is likely to be PLACED."
            )

        else:

            st.warning(
                "⚠️ The student is currently "
                "predicted as NOT PLACED."
            )

        # --------------------------------------------------
        # PROGRESS
        # --------------------------------------------------

        st.subheader(
            "📈 Placement Probability"
        )

        st.progress(
            float(
                placed_probability / 100
            )
        )

        p1, p2 = st.columns(2)

        with p1:

            st.write(
                f"**Placed:** "
                f"{placed_probability:.2f}%"
            )

        with p2:

            st.write(
                f"**Not Placed:** "
                f"{not_probability:.2f}%"
            )

        # --------------------------------------------------
        # CHART
        # --------------------------------------------------

        chart_df = pd.DataFrame({

            "Status": [
                "Placed",
                "Not Placed"
            ],

            "Probability": [
                placed_probability,
                not_probability
            ]
        })

        fig = px.bar(
            chart_df,
            x="Status",
            y="Probability",
            text="Probability",
            title="Placement Probability Analysis"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        st.markdown("---")

        st.subheader(
            "👨‍🎓 Student Summary"
        )

        s1, s2 = st.columns(2)

        with s1:

            st.write(
                f"**Gender:** {gender}"
            )

            st.write(
                f"**College Tier:** {college_tier}"
            )

            st.write(
                f"**Specialization:** {specialization}"
            )

            st.write(
                f"**CGPA:** {cgpa}"
            )

            st.write(
                f"**SSC:** {ssc}%"
            )

            st.write(
                f"**HSC:** {hsc}%"
            )

            st.write(
                f"**Projects:** {projects}"
            )

            st.write(
                f"**Internships:** {internships}"
            )

        with s2:

            st.write(
                f"**DSA Problems:** {dsa}"
            )

            st.write(
                f"**Coding Score:** {coding}"
            )

            st.write(
                f"**Resume Score:** {resume}"
            )

            st.write(
                f"**Mock Interview:** {mock}"
            )

            st.write(
                f"**Communication:** {communication}"
            )

            st.write(
                f"**AI/ML:** {aiml}"
            )

            st.write(
                f"**GitHub:** {github}"
            )

            st.write(
                f"**LeetCode:** {leetcode}"
            )

        # --------------------------------------------------
        # REPORT
        # --------------------------------------------------

        report = pd.DataFrame({

            "Field": [

                "Prediction",

                "Placement Probability",

                "CGPA",

                "Coding Score",

                "Resume Score",

                "Mock Interview",

                "Projects",

                "Internships",

                "GitHub",

                "LeetCode"
            ],

            "Value": [

                "PLACED"
                if prediction == 1
                else "NOT PLACED",

                f"{placed_probability:.2f}%",

                cgpa,

                coding,

                resume,

                mock,

                projects,

                internships,

                github,

                leetcode
            ]
        })

        csv_data = report.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Report",
            csv_data,
            file_name="placement_prediction_report.csv",
            mime="text/csv"
        )


# ==========================================================
# PAGE 3 — RESUME ANALYSIS
# ==========================================================

elif page == "📄 Resume Analysis":

    st.header("📄 AI Resume Analysis")

    st.write(
        "Upload your PDF or DOCX resume to extract "
        "and analyse your profile."
    )

    st.markdown("---")

    resume_file = st.file_uploader(
        "📤 Upload Resume",
        type=["pdf", "docx"],
        key="resume_upload"
    )

    if resume_file is not None:

        st.success(
            f"✅ Resume uploaded: {resume_file.name}"
        )

        if st.button(
            "📄 Extract & Analyse Resume",
            key="resume_analysis_button"
        ):

            try:

                extracted_text = extract_resume_text(
                    resume_file
                )

                if (
                    extracted_text
                    and
                    extracted_text.strip()
                ):

                    st.session_state[
                        "extracted_text"
                    ] = extracted_text

                    analysis = analyze_resume(
                        extracted_text
                    )

                    st.session_state[
                        "resume_analysis"
                    ] = analysis

                    # MAP FEATURES

                    try:

                        mapped_features = (
                            map_resume_to_features(
                                analysis
                            )
                        )

                        st.session_state[
                            "resume_features"
                        ] = mapped_features

                    except Exception:

                        st.session_state[
                            "resume_features"
                        ] = None

                    # AI RECOMMENDATIONS

                    if (
                        analysis
                        and
                        analysis.get(
                            "success",
                            True
                        )
                    ):

                        try:

                            ai_result = (
                                generate_recommendations(
                                    resume_analysis=analysis
                                )
                            )

                            st.session_state[
                                "ai_recommendations"
                            ] = ai_result

                        except Exception:

                            pass

                    st.success(
                        "✅ Resume analysis completed!"
                    )

                else:

                    st.warning(
                        "⚠️ No readable text found "
                        "in the resume."
                    )

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

    # ------------------------------------------------------
    # DISPLAY EXTRACTED TEXT
    # ------------------------------------------------------

    if st.session_state[
        "extracted_text"
    ]:

        st.markdown("---")

        st.subheader(
            "📋 Extracted Resume Text"
        )

        st.text_area(
            "Resume Content",
            st.session_state[
                "extracted_text"
            ],
            height=350
        )

    # ------------------------------------------------------
    # DISPLAY ANALYSIS
    # ------------------------------------------------------

    if st.session_state[
        "resume_analysis"
    ] is not None:

        analysis = st.session_state[
            "resume_analysis"
        ]

        st.markdown("---")

        st.subheader(
            "📊 Resume Analysis Results"
        )

        overall_score = analysis.get(
            "overall_score",
            analysis.get(
                "resume_score",
                None
            )
        )

        academic_data = analysis.get(
            "academic",
            {}
        )

        coding_data = analysis.get(
            "coding",
            {}
        )

        communication_data = analysis.get(
            "communication",
            {}
        )

        academic_score = (
            academic_data.get("score")
            if isinstance(
                academic_data,
                dict
            )
            else analysis.get(
                "academic_score"
            )
        )

        coding_score = (
            coding_data.get("score")
            if isinstance(
                coding_data,
                dict
            )
            else analysis.get(
                "coding_score"
            )
        )

        communication_score = (
            communication_data.get("score")
            if isinstance(
                communication_data,
                dict
            )
            else analysis.get(
                "communication_score"
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Overall",
                f"{float(overall_score):.0f}/100"
                if overall_score is not None
                else "Analysed"
            )

        with c2:

            st.metric(
                "Academic",
                f"{float(academic_score):.0f}/100"
                if academic_score is not None
                else "Analysed"
            )

        with c3:

            st.metric(
                "Coding",
                f"{float(coding_score):.0f}/100"
                if coding_score is not None
                else "Analysed"
            )

        with c4:

            st.metric(
                "Communication",
                f"{float(communication_score):.0f}/100"
                if communication_score is not None
                else "Analysed"
            )

        st.markdown("---")

        st.subheader(
            "🔎 Detailed Analysis"
        )

        for key, value in analysis.items():

            if key == "success":

                continue

            title = (
                key
                .replace("_", " ")
                .title()
            )

            if isinstance(
                value,
                list
            ):

                st.markdown(
                    f"### {title}"
                )

                for item in value:

                    st.write(
                        f"• {item}"
                    )

            elif isinstance(
                value,
                dict
            ):

                st.markdown(
                    f"### {title}"
                )

                for sub_key, sub_value in value.items():

                    sub_title = (
                        sub_key
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )

                    if isinstance(
                        sub_value,
                        list
                    ):

                        st.write(
                            f"**{sub_title}:**"
                        )

                        for item in sub_value:

                            st.write(
                                f"• {item}"
                            )

                    else:

                        st.write(
                            f"**{sub_title}:** "
                            f"{sub_value}"
                        )

            else:

                st.write(
                    f"**{title}:** {value}"
                )

        # --------------------------------------------------
        # MAPPED FEATURES
        # --------------------------------------------------

        if st.session_state[
            "resume_features"
        ]:

            st.markdown("---")

            st.subheader(
                "🎯 Placement Features Detected"
            )

            feature_data = (
                st.session_state[
                    "resume_features"
                ]
            )

            for feature, value in feature_data.items():

                st.write(
                    f"**{feature}:** {value}"
                )


# ==========================================================
# PAGE 4 — COMMUNICATION
# ==========================================================

elif page == "🗣️ Communication Assessment":

    st.header(
        "🗣️ Communication Assessment"
    )

    st.write(
        "Answer the interview question below "
        "to evaluate your communication skills."
    )

    st.markdown("---")

    st.info(
        """
        🎤 **Tell me about yourself, your education,
        your technical skills, your projects,
        and your career goals.**
        """
    )

    answer = st.text_area(
        "✍️ Type your answer here",
        height=220,
        placeholder=(
            "Example: My name is Harsha. "
            "I am pursuing M.Sc Data Analytics. "
            "I have knowledge of Python, SQL "
            "and Machine Learning..."
        ),
        key="communication_answer"
    )

    if st.button(
        "🗣️ Evaluate Communication",
        key="evaluate_communication"
    ):

        if not answer.strip():

            st.warning(
                "⚠️ Please enter your answer."
            )

        else:

            try:

                result = analyze_communication(
                    answer
                )

                st.session_state[
                    "communication_result"
                ] = result

                st.success(
                    "✅ Communication assessment completed!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    result = st.session_state[
        "communication_result"
    ]

    if result is not None:

        st.markdown("---")

        st.subheader(
            "📊 Communication Results"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:

            st.metric(
                "Grammar",
                f"{result['Grammar']}/10"
            )

        with c2:

            st.metric(
                "Clarity",
                f"{result['Clarity']}/10"
            )

        with c3:

            st.metric(
                "Vocabulary",
                f"{result['Vocabulary']}/10"
            )

        with c4:

            st.metric(
                "Relevance",
                f"{result['Relevance']}/10"
            )

        with c5:

            st.metric(
                "Sentence Structure",
                f"{result['Sentence Structure']}/10"
            )

        st.markdown("---")

        overall = result[
            "Overall Score"
        ]

        level = result[
            "Level"
        ]

        st.metric(
            "🗣️ Overall Communication Score",
            f"{overall}/10"
        )

        if level == "Excellent":

            st.success(
                "🌟 Communication Level: Excellent"
            )

        elif level == "Good":

            st.success(
                "✅ Communication Level: Good"
            )

        elif level == "Average":

            st.warning(
                "⚠️ Communication Level: Average"
            )

        else:

            st.error(
                "📚 Communication Level: Needs Improvement"
            )

        st.subheader(
            "📌 Assessment Summary"
        )

        if overall >= 8.5:

            st.write(
                "Your communication skills are excellent. "
                "You demonstrate good clarity, vocabulary, "
                "sentence structure and relevance."
            )

        elif overall >= 7:

            st.write(
                "Your communication skills are good. "
                "You can communicate your ideas effectively, "
                "with some areas available for improvement."
            )

        elif overall >= 5.5:

            st.write(
                "Your communication skills are average. "
                "Work on vocabulary, clarity and "
                "sentence structure."
            )

        else:

            st.write(
                "Your communication skills need improvement. "
                "Practice speaking and writing regularly."
            )


# ==========================================================
# PAGE 5 — CODING ASSESSMENT
# ==========================================================

elif page == "💻 Coding Assessment":

    st.header(
        "💻 Coding Assessment"
    )

    st.write(
        "Answer the following programming questions "
        "to evaluate your coding knowledge."
    )

    st.markdown("---")

    questions = [

        (
            "Question 1",
            """
            What is the time complexity of Binary Search?
            Explain your answer.
            """
        ),

        (
            "Question 2",
            """
            What is the difference between an Array
            and a Linked List?
            """
        ),

        (
            "Question 3",
            """
            What is the difference between a Stack
            and a Queue?
            """
        ),

        (
            "Question 4",
            """
            What is an algorithm?
            Give an example.
            """
        ),

        (
            "Question 5",
            """
            Which programming language do you know?
            Explain one programming concept.
            """
        )
    ]

    coding_answers = []

    for index, (
        title,
        question
    ) in enumerate(questions):

        st.subheader(title)

        st.write(question)

        answer = st.text_area(
            f"Your Answer - {title}",
            height=120,
            key=f"coding_question_{index}"
        )

        coding_answers.append(answer)

    st.markdown("---")

    if st.button(
        "💻 Evaluate Coding",
        key="evaluate_coding"
    ):

        if not any(
            answer.strip()
            for answer in coding_answers
        ):

            st.warning(
                "⚠️ Please answer at least one question."
            )

        else:

            try:

                result = analyze_coding(
                    coding_answers
                )

                st.session_state[
                    "coding_result"
                ] = result

                st.success(
                    "✅ Coding assessment completed!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

    # ------------------------------------------------------
    # CODING RESULT
    # ------------------------------------------------------

    result = st.session_state[
        "coding_result"
    ]

    if result is not None:

        st.markdown("---")

        st.subheader(
            "📊 Coding Assessment Results"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:

            st.metric(
                "Concept",
                f"{result['Concept Score']}/10"
            )

        with c2:

            st.metric(
                "Problem Solving",
                f"{result['Problem Solving']}/10"
            )

        with c3:

            st.metric(
                "Algorithm",
                f"{result['Algorithm Score']}/10"
            )

        with c4:

            st.metric(
                "Data Structures",
                f"{result['Data Structure Score']}/10"
            )

        with c5:

            st.metric(
                "Programming",
                f"{result['Programming Score']}/10"
            )

        st.markdown("---")

        overall = result[
            "Overall Score"
        ]

        level = result[
            "Level"
        ]

        st.metric(
            "💻 Overall Coding Score",
            f"{overall}/10"
        )

        if level == "Excellent":

            st.success(
                "🌟 Coding Level: Excellent"
            )

        elif level == "Good":

            st.success(
                "✅ Coding Level: Good"
            )

        elif level == "Average":

            st.warning(
                "⚠️ Coding Level: Average"
            )

        else:

            st.error(
                "📚 Coding Level: Needs Improvement"
            )

        st.subheader(
            "📌 Coding Assessment Summary"
        )

        if overall >= 8.5:

            st.write(
                "Excellent coding knowledge. "
                "You demonstrate strong understanding "
                "of programming concepts, algorithms "
                "and data structures."
            )

        elif overall >= 7:

            st.write(
                "Good coding knowledge. "
                "You have a good understanding of "
                "programming and problem-solving."
            )

        elif overall >= 5.5:

            st.write(
                "Average coding knowledge. "
                "Focus more on algorithms, data "
                "structures and problem-solving."
            )

        else:

            st.write(
                "Coding knowledge needs improvement. "
                "Practice programming fundamentals, "
                "algorithms and data structures."
            )


# ==========================================================
# PAGE 6 — SKILL GAP
# ==========================================================

elif page == "🧩 Skill Gap Analysis":

    st.header(
        "🧩 Skill Gap Analysis"
    )

    st.write(
        "Identify your strong skills, missing skills "
        "and areas requiring improvement."
    )

    st.markdown("---")

    # ------------------------------------------------------
    # GET RESUME DATA
    # ------------------------------------------------------

    detected_skills = []

    resume_score = 0

    if st.session_state[
        "resume_analysis"
    ] is not None:

        analysis = st.session_state[
            "resume_analysis"
        ]

        if isinstance(
            analysis,
            dict
        ):

            resume_score = analysis.get(
                "overall_score",
                analysis.get(
                    "resume_score",
                    0
                )
            )

            detected_skills = analysis.get(
                "Detected Skills",
                analysis.get(
                    "detected_skills",
                    []
                )
            )

    # ------------------------------------------------------
    # GET COMMUNICATION
    # ------------------------------------------------------

    communication_score = 0

    if st.session_state[
        "communication_result"
    ] is not None:

        communication_score = (
            st.session_state[
                "communication_result"
            ].get(
                "Overall Score",
                0
            )
        )

    # ------------------------------------------------------
    # GET CODING
    # ------------------------------------------------------

    coding_score = 0

    if st.session_state[
        "coding_result"
    ] is not None:

        coding_score = (
            st.session_state[
                "coding_result"
            ].get(
                "Overall Score",
                0
            )
        )

    # ------------------------------------------------------
    # STUDENT VALUES
    # ------------------------------------------------------

    cgpa_value = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    dsa_value = st.number_input(
        "DSA Problems Solved",
        min_value=0,
        value=200
    )

    internship_value = st.number_input(
        "Internships",
        min_value=0,
        value=1
    )

    project_value = st.number_input(
        "Projects",
        min_value=0,
        value=3
    )

    if st.button(
        "🧩 Analyze Skill Gap",
        key="skill_gap_button"
    ):

        try:

            result = analyze_skill_gap(

                detected_skills=detected_skills,

                communication_score=communication_score,

                coding_score=coding_score,

                resume_score=resume_score,

                cgpa=cgpa_value,

                dsa_problems=dsa_value,

                internships=internship_value,

                projects=project_value
            )

            st.session_state[
                "skill_gap_result"
            ] = result

            st.success(
                "✅ Skill gap analysis completed!"
            )

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )

    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    result = st.session_state[
        "skill_gap_result"
    ]

    if result is not None:

        st.markdown("---")

        st.subheader(
            "📊 Skill Coverage"
        )

        coverage = result[
            "Skill Coverage"
        ]

        readiness = result[
            "Readiness Level"
        ]

        st.metric(
            "Overall Skill Coverage",
            f"{coverage}%"
        )

        if readiness == "Excellent":

            st.success(
                "🌟 Skill Readiness: Excellent"
            )

        elif readiness == "Good":

            st.success(
                "✅ Skill Readiness: Good"
            )

        elif readiness == "Average":

            st.warning(
                "⚠️ Skill Readiness: Average"
            )

        else:

            st.error(
                "📚 Skill Readiness: Needs Improvement"
            )

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "💪 Strong Skills"
            )

            strong_skills = result[
                "Strong Skills"
            ]

            if strong_skills:

                for skill in strong_skills:

                    st.success(
                        f"✅ {skill.title()}"
                    )

            else:

                st.info(
                    "No strong skills detected yet."
                )

        with c2:

            st.subheader(
                "❌ Missing Skills"
            )

            missing_skills = result[
                "Missing Skills"
            ]

            if missing_skills:

                for skill in missing_skills:

                    st.warning(
                        f"⚠️ {skill.title()}"
                    )

            else:

                st.success(
                    "🎉 No major missing skills!"
                )

        st.markdown("---")

        st.subheader(
            "📌 Areas That Need Improvement"
        )

        weak_areas = result[
            "Weak Areas"
        ]

        if weak_areas:

            for area in weak_areas:

                st.error(
                    f"🔴 {area}"
                )

        else:

            st.success(
                "🎉 No major weak areas detected!"
            )

        st.subheader(
            "🚀 Improvement Priorities"
        )

        priorities = result[
            "Improvement Priorities"
        ]

        for index, priority in enumerate(
            priorities,
            start=1
        ):

            st.write(
                f"**{index}.** {priority}"
            )


# ==========================================================
# PAGE 7 — AI RECOMMENDATIONS
# ==========================================================

elif page == "🤖 AI Recommendations":

    st.header(
        "🤖 AI Career Recommendations"
    )

    st.write(
        "Personalised recommendations based "
        "on your resume analysis."
    )

    st.markdown("---")

    if st.session_state[
        "resume_analysis"
    ] is not None:

        if st.session_state[
            "ai_recommendations"
        ] is None:

            try:

                result = generate_recommendations(

                    resume_analysis=
                    st.session_state[
                        "resume_analysis"
                    ]
                )

                st.session_state[
                    "ai_recommendations"
                ] = result

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

        ai_result = st.session_state[
            "ai_recommendations"
        ]

        if ai_result is not None:

            priority = ai_result.get(
                "priority",
                "Low"
            )

            if priority == "High":

                st.error(
                    f"🚨 Recommendation Priority: "
                    f"{priority}"
                )

            elif priority == "Medium":

                st.warning(
                    f"⚠️ Recommendation Priority: "
                    f"{priority}"
                )

            else:

                st.success(
                    f"✅ Recommendation Priority: "
                    f"{priority}"
                )

            st.markdown("---")

            st.subheader(
                "🎯 Personalized Recommendations"
            )

            recommendations = ai_result.get(
                "recommendations",
                []
            )

            if recommendations:

                for index, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    st.write(
                        f"**{index}.** {recommendation}"
                    )

            else:

                st.success(
                    "🌟 No major recommendations."
                )

            st.markdown("---")

            st.subheader(
                "💪 Your Strengths"
            )

            strengths = ai_result.get(
                "strengths",
                []
            )

            if strengths:

                for strength in strengths:

                    st.success(
                        f"✅ {strength}"
                    )

            else:

                st.info(
                    "Continue developing your skills."
                )

            st.markdown("---")

            st.subheader(
                "📚 Skills to Improve"
            )

            missing_skills = ai_result.get(
                "missing_skills",
                []
            )

            if missing_skills:

                for skill in missing_skills:

                    st.warning(
                        f"⚠️ {skill}"
                    )

            else:

                st.success(
                    "🎉 No major missing skills."
                )

            st.markdown("---")

            st.subheader(
                "🚀 Career Guidance"
            )

            career_guidance = ai_result.get(
                "career_guidance",
                []
            )

            for guidance in career_guidance:

                st.info(
                    f"💡 {guidance}"
                )

    else:

        st.info(
            "📄 Please go to Resume Analysis "
            "and upload your resume first."
        )


# ==========================================================
# PAGE 8 — CAREER GUIDANCE
# ==========================================================

elif page == "🚀 Career Guidance":

    st.header(
        "🚀 Career Guidance Dashboard"
    )

    st.write(
        "Get an overall assessment of your "
        "career readiness."
    )

    st.markdown("---")

    if st.session_state[
        "resume_analysis"
    ] is not None:

        # --------------------------------------------------
        # RESUME SCORE
        # --------------------------------------------------

        resume_data = st.session_state[
            "resume_analysis"
        ]

        resume_score = resume_data.get(
            "overall_score",
            resume_data.get(
                "resume_score",
                0
            )
        )

        # --------------------------------------------------
        # COMMUNICATION SCORE
        # --------------------------------------------------

        communication_score = 0

        if st.session_state[
            "communication_result"
        ] is not None:

            communication_score = (
                st.session_state[
                    "communication_result"
                ].get(
                    "Overall Score",
                    0
                )
            )

        # --------------------------------------------------
        # CODING SCORE
        # --------------------------------------------------

        coding_score = 0

        if st.session_state[
            "coding_result"
        ] is not None:

            coding_score = (
                st.session_state[
                    "coding_result"
                ].get(
                    "Overall Score",
                    0
                )
            )

        # --------------------------------------------------
        # SKILL COVERAGE
        # --------------------------------------------------

        skill_coverage = 0

        if st.session_state[
            "skill_gap_result"
        ] is not None:

            skill_coverage = (
                st.session_state[
                    "skill_gap_result"
                ].get(
                    "Skill Coverage",
                    0
                )
            )

        # --------------------------------------------------
        # PLACEMENT PROBABILITY
        # --------------------------------------------------

        placement_probability = (
            st.session_state[
                "placement_probability"
            ]
        )

        # --------------------------------------------------
        # CONVERT 10 TO 100
        # --------------------------------------------------

        if communication_score <= 10:

            communication_score *= 10

        if coding_score <= 10:

            coding_score *= 10

        # --------------------------------------------------
        # GENERATE CAREER GUIDANCE
        # --------------------------------------------------

        try:

            career_result = generate_career_guidance(

                resume_score=resume_score,

                communication_score=
                communication_score,

                coding_score=coding_score,

                skill_coverage=skill_coverage,

                placement_probability=
                placement_probability
            )

            st.session_state[
                "career_result"
            ] = career_result

        except Exception as e:

            st.error(
                f"❌ Error: {str(e)}"
            )

            career_result = None

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        if career_result is not None:

            st.subheader(
                "📊 Overall Career Readiness"
            )

            st.metric(
                "Career Readiness Score",
                f"{career_result['overall_score']}/100"
            )

            readiness = career_result[
                "readiness_level"
            ]

            if readiness == "Excellent":

                st.success(
                    "🌟 Career Readiness: Excellent"
                )

            elif readiness == "Good":

                st.success(
                    "✅ Career Readiness: Good"
                )

            elif readiness == "Average":

                st.warning(
                    "⚠️ Career Readiness: Average"
                )

            else:

                st.error(
                    "📚 Career Readiness: Needs Improvement"
                )

            st.markdown("---")

            # ------------------------------------------------
            # BREAKDOWN
            # ------------------------------------------------

            st.subheader(
                "📈 Career Readiness Breakdown"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                st.metric(
                    "Resume",
                    f"{resume_score:.0f}/100"
                )

            with c2:

                st.metric(
                    "Communication",
                    f"{communication_score:.0f}/100"
                )

            with c3:

                st.metric(
                    "Coding",
                    f"{coding_score:.0f}/100"
                )

            with c4:

                st.metric(
                    "Skills",
                    f"{skill_coverage:.0f}/100"
                )

            with c5:

                st.metric(
                    "Placement",
                    f"{placement_probability:.0f}%"
                )

            st.markdown("---")

            # ------------------------------------------------
            # STRENGTHS
            # ------------------------------------------------

            st.subheader(
                "💪 Key Strengths"
            )

            strengths = career_result[
                "strengths"
            ]

            if strengths:

                for strength in strengths:

                    st.success(
                        f"✅ {strength}"
                    )

            else:

                st.info(
                    "Continue developing your skills."
                )

            # ------------------------------------------------
            # IMPROVEMENT
            # ------------------------------------------------

            st.subheader(
                "🎯 Areas for Improvement"
            )

            areas = career_result[
                "areas_for_improvement"
            ]

            if areas:

                for area in areas:

                    st.warning(
                        f"⚠️ {area}"
                    )

            else:

                st.success(
                    "🎉 No major improvement areas."
                )

            # ------------------------------------------------
            # CAREER ADVICE
            # ------------------------------------------------

            st.subheader(
                "💡 Recommended Career Path"
            )

            for advice in career_result[
                "career_advice"
            ]:

                st.info(
                    f"🚀 {advice}"
                )

    else:

        st.info(
            "📄 Please upload and analyse your resume "
            "first."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        color:#dddddd;
    ">

    <h4>🎓 Student Placement Prediction System</h4>

    <p>
    AI-Powered Student Career & Placement Analysis
    </p>

    <p>
    Algorithm: XGBoost Classifier
    </p>

    <p>
    Built using Python | Streamlit | XGBoost | Plotly
    </p>

    </div>
    """,
    unsafe_allow_html=True
)