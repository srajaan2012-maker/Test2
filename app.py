import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer Churn Prediction Dashboard")
st.markdown("Predict which customers are likely to churn and take proactive action!")

# Load model with comprehensive error handling
@st.cache_resource
def load_model():
    try:
        with open('churn_prediction_model.pkl', 'rb') as f:
            assets = pickle.load(f)
        
        # Verify all required components
        required_keys = ['model', 'scaler', 'feature_names']
        for key in required_keys:
            if key not in assets:
                st.error(f"❌ Model missing required key: {key}")
                return None
        
        st.success("✅ Original model loaded successfully!")
        return assets
        
    except Exception as e:
        st.error(f"❌ Error loading original model: {e}")
        st.info("Please ensure the model file is properly uploaded and formatted.")
        return None

def main():
    assets = load_model()
    if assets is None:
        st.stop()
    
    model = assets['model']
    scaler = assets['scaler']
    feature_names = assets['feature_names']
    
    # Display model info
    st.sidebar.header("📊 Model Information")
    if 'model_performance' in assets:
        perf = assets['model_performance']
        st.sidebar.metric("Accuracy", f"{perf.get('accuracy', 0.865)*100:.1f}%")
        st.sidebar.metric("Recall", f"{perf.get('recall', 0.464)*100:.1f}%")
        st.sidebar.metric("AUC Score", f"{perf.get('auc', 0.850)*100:.1f}%")
    else:
        st.sidebar.metric("Accuracy", "86.5%")
        st.sidebar.metric("Recall", "46.4%")
        st.sidebar.metric("AUC Score", "85.0%")
    
    st.sidebar.header("🎯 Top Risk Factors")
    st.sidebar.write("1. **Age** - Older customers")
    st.sidebar.write("2. **Geography** - German customers")  
    st.sidebar.write("3. **Activity** - Inactive members")
    st.sidebar.write("4. **Balance** - High balance accounts")
    st.sidebar.write("5. **Gender** - Female customers")
    
    # Main input form
    st.header("📋 Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider('Age', 18, 80, 40)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        geography = st.selectbox('Country', ['France', 'Germany', 'Spain'])
        
        st.subheader("Financial Information")
        credit_score = st.slider('Credit Score', 350, 850, 650)
        balance = st.number_input('Account Balance ($)', 0.0, 500000.0, 50000.0)
        estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 50000.0)
    
    with col2:
        st.subheader("Banking Relationship")
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        num_products = st.slider('Number of Products', 1, 4, 2)
        has_credit_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
        is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
    
    # Convert inputs
    gender_encoded = 1 if gender == 'Female' else 0
    has_credit_card_encoded = 1 if has_credit_card == 'Yes' else 0
    is_active_member_encoded = 1 if is_active_member == 'Yes' else 0
    
    # Create feature vector
    feature_vector = []
    for feature in feature_names:
        if feature.startswith('Geo_'):
            geo_feature = f"Geo_{geography}"
            feature_vector.append(1 if feature == geo_feature else 0)
        else:
            feature_mapping = {
                'CreditScore': credit_score,
                'Gender': gender_encoded,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': has_credit_card_encoded,
                'IsActiveMember': is_active_member_encoded,
                'EstimatedSalary': estimated_salary
            }
            feature_vector.append(feature_mapping[feature])
    
    # Prediction section
    st.header("🎯 Churn Prediction Results")
    
    if st.button('🔍 Predict Churn Risk', type='primary', use_container_width=True):
        # Prepare and scale features
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make prediction
        with st.spinner('Analyzing customer data...'):
            prediction = model.predict(feature_array_scaled)[0]
            probability = model.predict_proba(feature_array_scaled)[0][1]
        
        # Display results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"🚨 HIGH CHURN RISK")
                st.metric("Churn Probability", f"{probability:.1%}")
                st.progress(probability)
                
                st.warning("""
                **🚨 Immediate Actions Recommended:**
                - Proactive retention outreach
                - Special offers or discounts  
                - Personal account review
                - Customer success follow-up
                - Loyalty program enrollment
                """)
            else:
                st.success(f"✅ LOW CHURN RISK")
                st.metric("Churn Probability", f"{probability:.1%}")
                st.progress(probability)
                
                st.info("""
                **✅ Maintenance Actions:**
                - Continue current engagement
                - Monitor for changes in behavior
                - Cross-sell additional products
                - Regular satisfaction check-ins
                """)
        
        with result_col2:
            st.subheader("Risk Analysis")
            
            # Risk level
            if probability < 0.3:
                risk_level = "Low"
                risk_color = "green"
            elif probability < 0.7:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "High" 
                risk_color = "red"
            
            st.metric("Risk Level", risk_level)
            
            # Key factors
            st.subheader("Key Influencing Factors")
            if age > 50:
                st.write("🔴 **Age**: Older customer (higher risk)")
            else:
                st.write("🟢 **Age**: Younger customer (lower risk)")
                
            if geography == "Germany":
                st.write("🔴 **Geography**: German market (higher risk)")
            else:
                st.write("🟢 **Geography**: Non-German market (lower risk)")
                
            if is_active_member == "No":
                st.write("🔴 **Activity**: Inactive member (higher risk)")
            else:
                st.write("🟢 **Activity**: Active member (lower risk)")
    
    # Customer profile summary
    st.header("📊 Customer Profile Summary")
    
    profile_col1, profile_col2, profile_col3 = st.columns(3)
    
    with profile_col1:
        st.subheader("Demographics")
        st.write(f"**Age:** {age} years")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Country:** {geography}")
    
    with profile_col2:
        st.subheader("Financial")
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Balance:** ${balance:,.0f}")
        st.write(f"**Salary:** ${estimated_salary:,.0f}")
    
    with profile_col3:
        st.subheader("Relationship")
        st.write(f"**Tenure:** {tenure} years")
        st.write(f"**Products:** {num_products}")
        st.write(f"**Active Member:** {is_active_member}")
        st.write(f"**Credit Card:** {has_credit_card}")

if __name__ == '__main__':
    main()