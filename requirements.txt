import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AUTHOR INFORMATION ---
AUTHOR_NAME = "Ruth Musevele"
AUTHOR_EMAIL = "ruthiemwema04@gmail.com"
AUTHOR_PHONE = "0724752429"
AUTHOR_GITHUB = "#"
AUTHOR_LINKEDIN = "#"
APP_VERSION = "1.0.0"
APP_NAME = "Breast Cancer Prediction System"
APP_DESCRIPTION = "Machine Learning-based diagnosis prediction using Logistic Regression"

# --- FEATURE DESCRIPTIONS ---
FEATURE_DESCRIPTIONS = {
    "radius_mean": "Mean of distances from center to points on perimeter",
    "texture_mean": "Standard deviation of gray-scale values",
    "perimeter_mean": "Mean perimeter of tumor",
    "area_mean": "Mean area of tumor",
    "smoothness_mean": "Mean local variation in radius lengths",
    "compactness_mean": "Mean of (perimeter² / area - 1.0)",
    "concavity_mean": "Mean severity of concave portions of the contour",
    "concave_points_mean": "Mean number of concave portions of the contour",
    "symmetry_mean": "Mean symmetry of tumor",
    "fractal_dimension_mean": "Mean fractal dimension (coastline approximation)",
    "radius_se": "Standard error of radius",
    "texture_se": "Standard error of texture",
    "perimeter_se": "Standard error of perimeter",
    "area_se": "Standard error of area",
    "smoothness_se": "Standard error of smoothness",
    "compactness_se": "Standard error of compactness",
    "concavity_se": "Standard error of concavity",
    "concave_points_se": "Standard error of concave points",
    "symmetry_se": "Standard error of symmetry",
    "fractal_dimension_se": "Standard error of fractal dimension",
    "radius_worst": "Worst (largest) radius value",
    "texture_worst": "Worst texture value",
    "perimeter_worst": "Worst perimeter value",
    "area_worst": "Worst area value",
    "smoothness_worst": "Worst smoothness value",
    "compactness_worst": "Worst compactness value",
    "concavity_worst": "Worst concavity value",
    "concave_points_worst": "Worst concave points value",
    "symmetry_worst": "Worst symmetry value",
    "fractal_dimension_worst": "Worst fractal dimension value"
}

# --- LOAD DATA AND TRAIN MODEL ---
@st.cache_resource
def train_fresh_model():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, scaler, feature_names, X_train, X_test, y_train, y_test, X_test_scaled, predictions, accuracy

model, scaler, feature_names, X_train, X_test, y_train, y_test, X_test_scaled, predictions, accuracy = train_fresh_model()

# --- SIDEBAR WITH AUTHOR INFO ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("👤 About the Developer")
    st.markdown(f"**{AUTHOR_NAME}**")
    st.caption(f"Version {APP_VERSION}")
    
    st.markdown("---")
    
    st.markdown("### 📌 Contact")
    st.caption(f"📧 {AUTHOR_EMAIL}")
    st.caption(f"📱 {AUTHOR_PHONE}")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    st.caption(f"**Algorithm:** Logistic Regression")
    st.caption(f"**Accuracy:** {accuracy:.2%}")
    st.caption(f"**Features:** {len(feature_names)}")
    st.caption(f"**Training samples:** {len(X_train)}")
    st.caption(f"**Test samples:** {len(X_test)}")
    
    st.markdown("---")
    
    st.markdown("### 🏷️ Built With")
    st.caption("• Python")
    st.caption("• Streamlit")
    st.caption("• Scikit-learn")
    st.caption("• Pandas")
    st.caption("• NumPy")
    
    st.markdown("---")
    st.caption(f"© {datetime.now().year} {AUTHOR_NAME}")

# --- PAGE HEADER ---
st.title(f"🩺 {APP_NAME}")
st.write(APP_DESCRIPTION)
st.info(
    "⚠️ This is an educational machine-learning project and "
    "is not a medical diagnostic tool."
)

# --- FEATURE INFORMATION SECTION ---
st.header("📋 About the Features")

with st.expander("🔍 Click here to see all 30 features the model uses", expanded=False):
    st.write("""
    The model uses **30 features** derived from digitized images of fine needle aspirates (FNA) of breast masses.
    These features describe characteristics of the cell nuclei present in the image.
    """)
    
    feature_info = pd.DataFrame({
        "Feature Name": list(FEATURE_DESCRIPTIONS.keys()),
        "Description": list(FEATURE_DESCRIPTIONS.values())
    })
    
    feature_info["Type"] = feature_info["Feature Name"].apply(
        lambda x: "Mean" if "_mean" in x else ("SE" if "_se" in x else "Worst")
    )
    
    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )
    
    st.info("💡 **Tip:** Understanding these features helps interpret what the model is using to make predictions.")

st.markdown("---")

# --- MODEL TESTING SECTION ---
st.header("🧪 Test Model Accuracy")

if st.button("🧪 Run Full Model Test", type="primary"):
    try:
        accuracy = accuracy_score(y_test, predictions)
        
        st.subheader("📊 Test Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        with col2:
            correct = (predictions == y_test).sum()
            st.metric("✅ Correct", f"{correct}")
        with col3:
            incorrect = (predictions != y_test).sum()
            st.metric("❌ Incorrect", f"{incorrect}")
        
        cm = confusion_matrix(y_test, predictions)
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Benign", "Predicted Malignant"],
            index=["Actual Benign", "Actual Malignant"]
        )
        
        st.write("**Confusion Matrix:**")
        st.dataframe(cm_df)
        
        report = classification_report(y_test, predictions, 
                                      target_names=["Benign", "Malignant"])
        st.text("📋 Classification Report:")
        st.text(report)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

st.markdown("---")

# --- QUICK PREDICTIONS SECTION ---
st.header("🎯 Make Predictions (No Manual Entry!)")

st.write("Choose one of the methods below to get predictions without typing anything:")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Browse Samples", "🎲 Random Sample", "📤 Upload CSV", "📊 Batch Predict All"])

# --- TAB 1: Browse Samples ---
with tab1:
    st.subheader("📋 Browse and Predict Test Samples")
    
    st.write("""
    **How it works:** 
    - Browse through the test samples in the table below
    - Each sample has 30 feature values
    - Select any sample to see if the model predicts Benign or Malignant
    """)
    
    sample_df_display = pd.DataFrame(X_test, columns=feature_names)
    sample_df_display["Actual Diagnosis"] = ["Malignant" if y == 1 else "Benign" for y in y_test]
    sample_df_display["Predicted Diagnosis"] = ["Malignant" if p == 1 else "Benign" for p in predictions]
    sample_df_display["Correct"] = ["✅" if y_test[i] == predictions[i] else "❌" for i in range(len(y_test))]
    sample_df_display.index = [f"Sample {i+1}" for i in range(len(sample_df_display))]
    
    preview_cols = list(feature_names[:5]) + ["Actual Diagnosis", "Predicted Diagnosis", "Correct"]
    st.write("**Preview of test data (showing first 5 features):**")
    st.dataframe(
        sample_df_display[preview_cols],
        use_container_width=True
    )
    
    st.write("---")
    
    selected_sample = st.selectbox(
        "🔍 Select a sample to predict:",
        options=sample_df_display.index,
        format_func=lambda x: f"{x} - Actual: {sample_df_display.loc[x, 'Actual Diagnosis']} | Predicted: {sample_df_display.loc[x, 'Predicted Diagnosis']} | {sample_df_display.loc[x, 'Correct']}"
    )
    
    if selected_sample:
        idx = int(selected_sample.split()[1]) - 1
        sample_data = X_test[idx].reshape(1, -1)
        actual_label = y_test[idx]
        prediction = predictions[idx]
        probabilities = model.predict_proba(scaler.transform(sample_data))[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Actual Diagnosis", "Malignant" if actual_label == 1 else "Benign")
        with col2:
            st.metric("Predicted Diagnosis", "Malignant" if prediction == 1 else "Benign")
        with col3:
            status = "✅ Correct" if prediction == actual_label else "❌ Incorrect"
            st.metric("Status", status)
        
        st.info(f"**Model Confidence:** {max(probabilities):.2%}")
        
        with st.expander("📊 View All 30 Feature Values for This Sample", expanded=False):
            feature_df = pd.DataFrame({
                "Feature Name": feature_names,
                "Description": [FEATURE_DESCRIPTIONS.get(f, "") for f in feature_names],
                "Value": sample_data[0]
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

# --- TAB 2: Random Sample ---
with tab2:
    st.subheader("🎲 Generate Random Sample")
    
    st.write("""
    **How it works:**
    - Click the button to randomly select a sample from the test data
    - The model will predict Benign or Malignant
    - You'll see if the prediction was correct
    """)
    
    if st.button("🎲 Generate Random Test Sample", type="primary"):
        idx = random.randint(0, len(X_test) - 1)
        sample_data = X_test[idx].reshape(1, -1)
        actual_label = y_test[idx]
        
        sample_scaled = scaler.transform(sample_data)
        prediction = model.predict(sample_scaled)[0]
        probabilities = model.predict_proba(sample_scaled)[0]
        
        st.success(f"✅ Sample #{idx+1} loaded randomly!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if actual_label == 1:
                st.error("🔴 Actual: Malignant")
            else:
                st.success("🟢 Actual: Benign")
        with col2:
            if prediction == 1:
                st.error("🔴 Predicted: Malignant")
            else:
                st.success("🟢 Predicted: Benign")
        with col3:
            if prediction == actual_label:
                st.success(f"✅ Correct! ({max(probabilities):.2%} confidence)")
            else:
                st.error(f"❌ Incorrect ({max(probabilities):.2%} confidence)")
        
        with st.expander("📊 View Feature Values", expanded=False):
            feature_df = pd.DataFrame({
                "Feature Name": feature_names,
                "Description": [FEATURE_DESCRIPTIONS.get(f, "") for f in feature_names],
                "Value": sample_data[0]
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

# --- TAB 3: Upload CSV ---
with tab3:
    st.subheader("📤 Upload CSV File for Batch Prediction")
    
    st.write("""
    **How it works:**
    - Upload a CSV file with patient data
    - Each row should have 30 feature values (in the same order as the features above)
    - The model will predict Benign or Malignant for each row
    - Download the results as a CSV file
    """)
    
    with st.expander("📋 Required Feature Order in CSV", expanded=False):
        st.write("Your CSV should have these 30 columns in this exact order:")
        feature_order_df = pd.DataFrame({
            "Column Order": list(range(1, 31)),
            "Feature Name": feature_names,
            "Description": [FEATURE_DESCRIPTIONS.get(f, "") for f in feature_names]
        })
        st.dataframe(feature_order_df, use_container_width=True, hide_index=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            
            st.write(f"📊 Loaded {len(uploaded_data)} samples")
            st.write("Preview of uploaded data:")
            st.dataframe(uploaded_data.head())
            
            if len(uploaded_data.columns) == 30:
                if st.button("🔬 Predict All Samples in CSV", type="primary"):
                    X_upload = uploaded_data.values
                    X_upload_scaled = scaler.transform(X_upload)
                    upload_predictions = model.predict(X_upload_scaled)
                    upload_probs = model.predict_proba(X_upload_scaled)
                    
                    results_df = uploaded_data.copy()
                    results_df["Predicted Diagnosis"] = ["Malignant" if p == 1 else "Benign" for p in upload_predictions]
                    results_df["Confidence"] = [max(prob) for prob in upload_probs]
                    
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"⚠️ Your CSV has {len(uploaded_data.columns)} columns, but 30 are required. Please check the feature order above.")
                
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# --- TAB 4: Batch Predict All ---
with tab4:
    st.subheader("📊 Batch Predict All Test Samples")
    
    st.write(f"""
    **How it works:**
    - Predict all **{len(X_test)} test samples** at once
    - See overall accuracy, correct/incorrect count
    - Download full results with all feature values
    """)
    
    if st.button("📊 Predict All Samples", type="primary"):
        correct = (predictions == y_test).sum()
        incorrect = (predictions != y_test).sum()
        accuracy = accuracy_score(y_test, predictions)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(y_test))
        with col2:
            st.metric("✅ Correct", correct)
        with col3:
            st.metric("❌ Incorrect", incorrect)
        with col4:
            st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        
        st.subheader("📋 Detailed Results")
        
        results_df = pd.DataFrame(X_test, columns=feature_names)
        results_df["Actual"] = ["Malignant" if y == 1 else "Benign" for y in y_test]
        results_df["Predicted"] = ["Malignant" if p == 1 else "Benign" for p in predictions]
        results_df["Correct"] = ["✅" if y_test[i] == predictions[i] else "❌" for i in range(len(y_test))]
        results_df["Confidence"] = [max(model.predict_proba(scaler.transform(X_test[i].reshape(1, -1)))[0]) for i in range(len(X_test))]
        
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Predictions as CSV",
            data=csv,
            file_name="all_predictions.csv",
            mime="text/csv"
        )

st.markdown("---")

# --- SECTION 2: MANUAL INPUT ---
with st.expander("📝 Manual Entry (Advanced - Only if you have specific values to test)"):
    st.caption("Use this section only if you want to manually enter values instead of using the automated methods above.")
    
    st.subheader("Mean Values")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius_mean = st.number_input("Radius (mean)", min_value=0.0, value=14.0, step=0.1)
        texture_mean = st.number_input("Texture (mean)", min_value=0.0, value=19.0, step=0.1)
        perimeter_mean = st.number_input("Perimeter (mean)", min_value=0.0, value=91.0, step=0.1)
        area_mean = st.number_input("Area (mean)", min_value=0.0, value=650.0, step=1.0)
    
    with col2:
        smoothness_mean = st.number_input("Smoothness (mean)", min_value=0.0, value=0.1, step=0.001, format="%.3f")
        compactness_mean = st.number_input("Compactness (mean)", min_value=0.0, value=0.1, step=0.001, format="%.3f")
        concavity_mean = st.number_input("Concavity (mean)", min_value=0.0, value=0.1, step=0.001, format="%.3f")
        concave_points_mean = st.number_input("Concave Points (mean)", min_value=0.0, value=0.05, step=0.001, format="%.3f")
    
    with col3:
        symmetry_mean = st.number_input("Symmetry (mean)", min_value=0.0, value=0.18, step=0.001, format="%.3f")
        fractal_dimension_mean = st.number_input("Fractal Dimension (mean)", min_value=0.0, value=0.06, step=0.001, format="%.3f")
    
    st.subheader("Standard Error Values")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        radius_se = st.number_input("Radius (se)", min_value=0.0, value=0.5, step=0.01)
        texture_se = st.number_input("Texture (se)", min_value=0.0, value=1.0, step=0.01)
        perimeter_se = st.number_input("Perimeter (se)", min_value=0.0, value=3.0, step=0.1)
        area_se = st.number_input("Area (se)", min_value=0.0, value=30.0, step=1.0)
    
    with col5:
        smoothness_se = st.number_input("Smoothness (se)", min_value=0.0, value=0.005, step=0.001, format="%.3f")
        compactness_se = st.number_input("Compactness (se)", min_value=0.0, value=0.02, step=0.001, format="%.3f")
        concavity_se = st.number_input("Concavity (se)", min_value=0.0, value=0.03, step=0.001, format="%.3f")
        concave_points_se = st.number_input("Concave Points (se)", min_value=0.0, value=0.01, step=0.001, format="%.3f")
    
    with col6:
        symmetry_se = st.number_input("Symmetry (se)", min_value=0.0, value=0.02, step=0.001, format="%.3f")
        fractal_dimension_se = st.number_input("Fractal Dimension (se)", min_value=0.0, value=0.002, step=0.001, format="%.3f")
    
    st.subheader("Worst Values")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        radius_worst = st.number_input("Radius (worst)", min_value=0.0, value=16.0, step=0.1)
        texture_worst = st.number_input("Texture (worst)", min_value=0.0, value=25.0, step=0.1)
        perimeter_worst = st.number_input("Perimeter (worst)", min_value=0.0, value=105.0, step=0.1)
        area_worst = st.number_input("Area (worst)", min_value=0.0, value=880.0, step=1.0)
    
    with col8:
        smoothness_worst = st.number_input("Smoothness (worst)", min_value=0.0, value=0.12, step=0.001, format="%.3f")
        compactness_worst = st.number_input("Compactness (worst)", min_value=0.0, value=0.25, step=0.001, format="%.3f")
        concavity_worst = st.number_input("Concavity (worst)", min_value=0.0, value=0.3, step=0.001, format="%.3f")
        concave_points_worst = st.number_input("Concave Points (worst)", min_value=0.0, value=0.1, step=0.001, format="%.3f")
    
    with col9:
        symmetry_worst = st.number_input("Symmetry (worst)", min_value=0.0, value=0.25, step=0.001, format="%.3f")
        fractal_dimension_worst = st.number_input("Fractal Dimension (worst)", min_value=0.0, value=0.08, step=0.001, format="%.3f")
    
    if st.button("🔬 Predict Manual Entry", type="secondary"):
        try:
            input_data = np.array([[
                radius_mean, texture_mean, perimeter_mean, area_mean,
                smoothness_mean, compactness_mean, concavity_mean,
                concave_points_mean, symmetry_mean, fractal_dimension_mean,
                radius_se, texture_se, perimeter_se, area_se,
                smoothness_se, compactness_se, concavity_se,
                concave_points_se, symmetry_se, fractal_dimension_se,
                radius_worst, texture_worst, perimeter_worst, area_worst,
                smoothness_worst, compactness_worst, concavity_worst,
                concave_points_worst, symmetry_worst, fractal_dimension_worst
            ]])
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.header("📊 Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ **MALIGNANT** (Probability: {probabilities[1]:.2%})")
            else:
                st.success(f"✅ **BENIGN** (Probability: {probabilities[0]:.2%})")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# --- FOOTER WITH AUTHOR INFO ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")

with col2:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px 0;">
            <p style="font-size: 14px; color: #666;">
                Built with ❤️ by <strong>{AUTHOR_NAME}</strong>
            </p>
            <p style="font-size: 12px; color: #999;">
                📧 {AUTHOR_EMAIL} &bull; 📱 {AUTHOR_PHONE}
            </p>
            <p style="font-size: 12px; color: #999;">
                {APP_NAME} v{APP_VERSION} &bull; {datetime.now().year}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.write("")