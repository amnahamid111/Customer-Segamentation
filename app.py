import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_files():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

kmeans, scaler, training_columns = load_files()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main{
    background:#f5f7fb;
}

.title{
    font-size:38px;
    font-weight:bold;
    color:#003366;
}

.subtitle{
    font-size:18px;
    color:gray;
}

div.stButton > button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:12px;
    background:#0066cc;
    color:white;
}

div.stButton > button:hover{
    background:#004c99;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.image("https://img.icons8.com/color/96/customer-insight.png", width=90)

st.sidebar.title("AI Dashboard")

st.sidebar.info("""
### Customer Segmentation

Machine Learning Algorithm

✔ K-Means Clustering

Dataset

✔ 8068 Customers

Purpose

✔ Customer Group Analysis
""")

# ---------------- HEADER ----------------

st.markdown("<div class='title'>👥 AI Customer Segmentation Dashboard</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Predict which customer segment a new customer belongs to using Machine Learning.</div>", unsafe_allow_html=True)

st.divider()

# ---------------- CUSTOMER DETAILS ----------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        18,
        100,
        30
    )

    work_experience = st.number_input(
        "Work Experience",
        0,
        50,
        2
    )

    family_size = st.number_input(
        "Family Size",
        1,
        10,
        3
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married = st.selectbox(
        "Ever Married",
        ["Yes","No"]
    )

    graduated = st.selectbox(
        "Graduated",
        ["Yes","No"]
    )

with col3:

    profession = st.selectbox(
        "Profession",
        [
            "Doctor",
            "Engineer",
            "Entertainment",
            "Executive",
            "Healthcare",
            "Homemaker",
            "Lawyer",
            "Marketing"
        ]
    )

    spending = st.selectbox(
        "Spending Score",
        [
            "High",
            "Average",
            "Low"
        ]
    )

    var1 = st.selectbox(
        "Customer Category",
        [
            "Cat_1",
            "Cat_2",
            "Cat_3",
            "Cat_4",
            "Cat_5",
            "Cat_6",
            "Cat_7"
        ]
    )

st.divider()

predict = st.button("🔍 Predict Customer Segment")

if predict:

    # -----------------------------
    # Create Input DataFrame
    # -----------------------------
    input_df = pd.DataFrame({
        "Age": [age],
        "Work_Experience": [work_experience],
        "Family_Size": [family_size],
        "Gender": [gender],
        "Ever_Married": [married],
        "Graduated": [graduated],
        "Profession": [profession],
        "Spending_Score": [spending],
        "Var_1": [var1]
    })

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    input_df = pd.get_dummies(input_df)

    # -----------------------------
    # Match Training Columns
    # -----------------------------
    input_df = input_df.reindex(
        columns=training_columns,
        fill_value=0
    )

    # -----------------------------
    # Convert Boolean to Integer
    # -----------------------------
    bool_columns = input_df.select_dtypes(include=["bool"]).columns
    input_df[bool_columns] = input_df[bool_columns].astype(int)

    st.write("Training Columns:")
    st.write(training_columns)

    st.write("Input Columns:")
    st.write(input_df.columns.tolist())
    st.write("Training Column Count:", len(training_columns))
    st.write("Input Column Count:", len(input_df.columns))
    # -----------------------------
    # Scale the Data
    # -----------------------------
    scaled_input = scaler.transform(input_df)

    # -----------------------------
    # Predict Cluster
    # -----------------------------
    prediction = kmeans.predict(scaled_input)[0]

    st.divider()

    st.subheader("Prediction Result")

    st.success(f"Customer belongs to Cluster {prediction}")

    # Store prediction for Part 3
    predicted_cluster = prediction


    st.divider()

    st.subheader("📊 Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", age)
        st.metric("Family Size", family_size)

    with col2:
        st.metric("Work Experience", work_experience)
        st.metric("Gender", gender)

    with col3:
        st.metric("Profession", profession)
        st.metric("Spending Score", spending)

    st.divider()

    # -----------------------------
    # Cluster Explanation
    # -----------------------------

    if prediction == 0:

        st.success("### 🟢 Cluster 0 - Stable Customers")

        st.write("""
### Characteristics

- 👨 Middle-aged customers
- 💍 Mostly married
- 🎓 Mostly graduates
- 💰 Mostly low spending customers

### Business Recommendation

- Offer personalised discounts
- Introduce loyalty programmes
- Recommend family products
        """)

    elif prediction == 1:

        st.info("### 🔵 Cluster 1 - Premium Customers")

        st.write("""
### Characteristics

- 👴 Older customers
- 💎 High spending behaviour
- 💍 Mostly married

### Business Recommendation

- Premium membership
- VIP offers
- Exclusive rewards
- Luxury product recommendations
        """)

    else:

        st.warning("### 🟠 Cluster 2 - Young Customers")

        st.write("""
### Characteristics

- 👦 Young customers
- 👨‍🎓 Mostly unmarried
- 👨‍👩‍👧 Larger families
- 💰 Low spending behaviour

### Business Recommendation

- Student discounts
- Budget-friendly products
- Promotional campaigns
        """)

    st.divider()

    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Work Experience",
            "Family Size",
            "Gender",
            "Married",
            "Graduated",
            "Profession",
            "Spending Score",
            "Category",
            "Predicted Cluster"
        ],
        "Value": [
            age,
            work_experience,
            family_size,
            gender,
            married,
            graduated,
            profession,
            spending,
            var1,
            prediction
        ]
    })

    st.dataframe(summary, use_container_width=True)

    csv = summary.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )

st.divider()

st.subheader("📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", "8,068")

with col2:
    st.metric("Clusters", "3")

with col3:
    st.metric("Algorithm", "K-Means")

st.caption("Developed using Python • Streamlit • Scikit-Learn • K-Means Clustering")

import matplotlib.pyplot as plt

st.divider()

st.subheader("📊 Customer Distribution")

cluster_sizes = [4395, 1341, 2332]

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(
    ["Cluster 0", "Cluster 1", "Cluster 2"],
    cluster_sizes
)

ax.set_ylabel("Customers")
ax.set_title("Customer Segments")

st.pyplot(fig)

st.divider()

with st.expander("📖 About This Project"):

    st.write("""
### AI Customer Segmentation

This project uses the K-Means Clustering algorithm to group customers into similar segments.

### Machine Learning Workflow

✔ Load Dataset

✔ Data Cleaning

✔ Handle Missing Values

✔ One-Hot Encoding

✔ Feature Scaling

✔ Elbow Method

✔ K-Means Clustering

✔ Customer Prediction

### Purpose

Help businesses understand different customer groups and create better marketing strategies.
""")