import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL FILES
# -------------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

kmeans, scaler, training_columns = load_files()

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.title{
    font-size:40px;
    font-weight:bold;
    color:#003366;
}

.subtitle{
    font-size:18px;
    color:#666666;
}

div.stButton > button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:10px;
    background-color:#0066CC;
    color:white;
}

div.stButton > button:hover{
    background-color:#004999;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/customer-insight.png",
    width=90
)

st.sidebar.title("🤖 AI Dashboard")

st.sidebar.success("""
### Customer Segmentation

✔ Machine Learning

✔ K-Means Clustering

✔ 8068 Customers

✔ Business Analytics
""")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    "<div class='title'>👥 AI Customer Segmentation Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict customer segments using Machine Learning.</div>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# DASHBOARD OVERVIEW
# -------------------------------------------------

st.subheader("📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👥 Total Customers", "8,068")

with col2:
    st.metric("📊 Clusters", "3")

with col3:
    st.metric("🤖 Algorithm", "K-Means")

st.divider()

# -------------------------------------------------
# CUSTOMER INFORMATION
# -------------------------------------------------

st.subheader("📝 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    work_experience = st.number_input(
        "Work Experience",
        min_value=0,
        max_value=50,
        value=2
    )

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=10,
        value=3
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

    # -------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -------------------------------------------------

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

    # -------------------------------------------------
    # ONE-HOT ENCODING
    # -------------------------------------------------

    input_df = pd.get_dummies(input_df)

    # -------------------------------------------------
    # MATCH TRAINING COLUMNS
    # -------------------------------------------------

    training_cols = [col for col in training_columns if col != "Cluster"]

    input_df = input_df.reindex(
        columns=training_cols,
        fill_value=0
    )

    # -------------------------------------------------
    # CONVERT BOOLEAN TO INTEGER
    # -------------------------------------------------

    bool_columns = input_df.select_dtypes(include=["bool"]).columns

    input_df[bool_columns] = input_df[bool_columns].astype(int)

    # -------------------------------------------------
    # SCALE INPUT
    # -------------------------------------------------

    scaled_input = scaler.transform(input_df)

    # -------------------------------------------------
    # PREDICT CUSTOMER CLUSTER
    # -------------------------------------------------

    prediction = kmeans.predict(scaled_input)[0]

    # -------------------------------------------------
    # CLUSTER INFORMATION
    # -------------------------------------------------

    cluster_info = {

        0: {
            "name": "🟢 Stable Customer",
            "description": "These customers are generally middle-aged, loyal and usually spend less. They respond well to discounts and loyalty programmes."
        },

        1: {
            "name": "🔵 Premium Customer",
            "description": "These customers are valuable and tend to spend more. They are ideal for premium offers and exclusive memberships."
        },

        2: {
            "name": "🟠 Young Customer",
            "description": "These customers are younger and usually look for affordable products and promotional offers."
        }

    }

    st.divider()

    # -------------------------------------------------
    # AI PREDICTION
    # -------------------------------------------------

    st.subheader("🤖 AI Prediction")

    st.success(cluster_info[prediction]["name"])

    st.info(cluster_info[prediction]["description"])

    # -------------------------------------------------
    # BUSINESS RECOMMENDATION
    # -------------------------------------------------

    st.subheader("💡 Business Recommendation")

    if prediction == 0:

        st.write("""
✅ Offer loyalty programmes

✅ Family discount packages

✅ Cross-selling opportunities

✅ Seasonal promotions
""")

    elif prediction == 1:

        st.write("""
✅ VIP membership

✅ Premium product recommendations

✅ Exclusive rewards

✅ Luxury marketing campaigns
""")

    else:

        st.write("""
✅ Student discounts

✅ Budget-friendly offers

✅ Mobile app promotions

✅ Coupon campaigns
""")

    st.divider()

    # -------------------------------------------------
    # CUSTOMER PROFILE
    # -------------------------------------------------

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

    # -------------------------------------------------
    # CUSTOMER SUMMARY
    # -------------------------------------------------

    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Work Experience",
            "Family Size",
            "Gender",
            "Married",
            "Graduated",
            "Profession",
            "Spending Score",
            "Customer Category",
            "Predicted Cluster"
        ],

        "Value":[
            age,
            work_experience,
            family_size,
            gender,
            married,
            graduated,
            profession,
            spending,
            var1,
            cluster_info[prediction]["name"]
        ]

    })

    st.dataframe(summary, use_container_width=True)

    csv = summary.to_csv(index=False)

    st.download_button(

        "📥 Download Prediction Report",

        csv,

        file_name="Customer_Prediction_Report.csv",

        mime="text/csv"

    )

    st.divider()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.caption(
    "Developed using Python • Streamlit • Scikit-Learn • K-Means Clustering"
)

st.divider()

# -------------------------------------------------
# CUSTOMER DISTRIBUTION
# -------------------------------------------------

st.subheader("📊 Customer Distribution")

cluster_sizes = [4395, 1341, 2332]

fig, ax = plt.subplots(figsize=(7,4))

bars = ax.bar(
    ["Stable", "Premium", "Young"],
    cluster_sizes
)

ax.set_ylabel("Number of Customers")
ax.set_title("Customer Segments")

# Show numbers on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

st.pyplot(fig)

st.divider()

# -------------------------------------------------
# ABOUT PROJECT
# -------------------------------------------------

with st.expander("📖 About This Project"):

    st.markdown("""
## 👥 AI Customer Segmentation

This project uses **Machine Learning (K-Means Clustering)** to group customers with similar characteristics.

### 📂 Dataset

- Total Customers: **8,068**
- Features: **22**
- Algorithm: **K-Means Clustering**

---

## ⚙ Machine Learning Workflow

✅ Load Dataset

✅ Data Cleaning

✅ Handle Missing Values

✅ One-Hot Encoding

✅ Feature Scaling

✅ Find Optimal Clusters (Elbow Method)

✅ Train K-Means Model

✅ Predict Customer Segment

---

## 🎯 Business Benefits

Businesses can use customer segmentation to:

- Understand different customer groups
- Improve marketing campaigns
- Create personalised offers
- Increase customer satisfaction
- Improve business decisions

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Matplotlib

---

## 👨‍💻 Developed By

AI Customer Segmentation Project
""")

st.divider()

# -------------------------------------------------
# FINAL FOOTER
# -------------------------------------------------

st.markdown(
    """
    <center>
        <p style="color:gray;">
        © 2026 AI Customer Segmentation Dashboard
        </p>
    </center>
    """,
    unsafe_allow_html=True
)