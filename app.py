"""
Streamlit Web Interface for Farmsky Credit Scoring System
Provides interactive UI for training, prediction, and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

from config import *
from data_generator import SyntheticDataGenerator
from feature_engineering import FeatureEngineer
from scoring_engine import CreditScoringEngine
from model_trainer import CreditModelTrainer
from predictor import CreditPredictor


# Page configuration
st.set_page_config(
    page_title="Farmsky Credit Scoring System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #388E3C;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F1F8F4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #FF9800;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #F44336;
    }
    
    /* Navigation button styling */
    .stButton > button {
        background-color: white;
        color: #333;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        text-align: left;
        height: auto;
        white-space: normal;
        line-height: 1.4;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-color: #4CAF50;
        color: #1B5E20;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton > button:active,
    .stButton > button:focus {
        background-color: #C8E6C9;
        border-color: #2E7D32;
        color: #1B5E20;
        box-shadow: 0 2px 6px rgba(76, 175, 80, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Arial Black', sans-serif;
    }
    
    /* Navigation icon and text alignment */
    [data-testid="stSidebar"] .row-widget.stButton {
        margin-bottom: 0.5rem;
    }
    
    /* Better spacing for navigation section */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Sidebar navigation with logo
    with st.sidebar:
        # Logo in sidebar
        logo_path = "assets/logo.png"  
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            # Fallback: Use emoji as logo
            st.markdown('<div style="text-align: center; font-size: 4rem; margin: 1rem 0;">🌾</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
    
        
        # Initialize session state for page navigation
        if 'page' not in st.session_state:
            st.session_state.page = "🏠Home"
        
        # Navigation menu with justified format
        nav_options = [
            ("🏠", "Home", "Dashboard overview"),
            ("📊", "Data Generation", "Manage training data"),
            ("🤖", "Train Model", "Build ML models"),
            ("🎯", "Multiple Evaluations", "Batch evaluation"),
            ("👤", "Single Evaluations", "Individual assessment"),
            ("📈", "Analytics Dashboard", "Insights & reports")
        ]
        
        for icon, page_name, description in nav_options:
            # Create a button with icon and description
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f'<div style="font-size: 1.5rem; text-align: center; padding-top: 0.3rem;">{icon}</div>', unsafe_allow_html=True)
            with col2:
                full_name = f"{icon}{page_name}"
                if st.button(f"**{page_name}**\n\n{description}", key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.page = full_name
        
        st.markdown("---")
        
        # System info in sidebar
        st.markdown("### ℹ️ Quick Info")
        data_exists = os.path.exists(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}")
        model_exists = os.path.exists(f"{MODELS_DIR}/{TRAINED_MODEL_FILE}")
        
        if data_exists:
            st.success("✓ Data Ready", icon="✅")
        else:
            st.warning("⚠ No Data", icon="⚠️")
        
        if model_exists:
            st.success("✓ Model Ready", icon="✅")
        else:
            st.warning("⚠ No Model", icon="⚠️")
    
    page = st.session_state.page
    
    # Main content area header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="color: #2E7D32; margin: 0; font-size: 2.5rem;">🌾 Farmsky Credit Scoring System</h1>
        <p style="color: #666; font-size: 1.1rem; margin-top: 0.5rem;">AI-Powered Agricultural Lending Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "🏠Home":
        show_home()
    elif page == "📊Data Generation":
        show_data_generation()
    elif page == "🤖Train Model":
        show_training()
    elif page == "🎯Multiple Evaluations":
        show_predictions()
    elif page == "📈Analytics Dashboard":
        show_analytics()
    elif page == "👤Single Evaluations":
        show_single_application()


def generate_detailed_explanation(farmer_data: dict, result: dict) -> dict:
    """Generate detailed, actionable explanation for credit decision"""
    
    explanation = {
        'income_analysis': [],
        'agro_risk_factors': [],
        'market_risk_factors': [],
        'financial_behavior': [],
        'social_capital': [],
        'recommendations': []
    }
    
    # Calculate expected income
    expected_yield = farmer_data.get('historical_yield', 0)
    farm_size = farmer_data.get('farm_size', 0)
    projected_price = farmer_data.get('projected_harvest_price', 0)
    loan_amount = farmer_data.get('loan_amount_kes', 0)
    
    expected_income = expected_yield * farm_size * projected_price * 1000  # Convert to KES
    
    # 1. INCOME VS LOAN ANALYSIS
    if expected_income > 0:
        income_ratio = expected_income / loan_amount if loan_amount > 0 else 0
        
        if income_ratio < 1:
            explanation['income_analysis'].append(
                f"⚠️ **Income Shortfall**: Expected harvest income (KES {expected_income:,.0f}) is BELOW loan amount (KES {loan_amount:,.0f})"
            )
            explanation['income_analysis'].append(
                f"   • Yield: {expected_yield:.2f} tons/acre × Farm size: {farm_size:.2f} acres × Price: KES {projected_price:,.0f}/ton"
            )
            explanation['income_analysis'].append(
                f"   • Income covers only {income_ratio:.0%} of loan - HIGH RISK"
            )
            explanation['recommendations'].append("Reduce loan amount or increase farm productivity")
        elif income_ratio < 1.5:
            explanation['income_analysis'].append(
                f"⚠️ **Tight Margins**: Expected income (KES {expected_income:,.0f}) only {income_ratio:.1f}x the loan amount (KES {loan_amount:,.0f})"
            )
            explanation['recommendations'].append("Consider longer repayment period to reduce monthly burden")
        else:
            explanation['income_analysis'].append(
                f"✓ **Good Income**: Expected income (KES {expected_income:,.0f}) is {income_ratio:.1f}x the loan amount"
            )
    
    # 2. AGRO-ECOLOGICAL RISK FACTORS
    drought_index = farmer_data.get('drought_index', 0)
    rainfall_12m = farmer_data.get('rainfall_12m_mm', 0)
    irrigation = farmer_data.get('irrigation_access', False)
    climate_suitability = farmer_data.get('climate_suitability', 0)
    soil_suitability = farmer_data.get('soil_suitability', 0)
    
    if drought_index > 0.6:
        explanation['agro_risk_factors'].append(
            f"⚠️ **High Drought Risk**: Drought index is {drought_index:.0%} (HIGH)"
        )
        if not irrigation:
            explanation['agro_risk_factors'].append(
                "   • NO irrigation access - crops fully dependent on rainfall"
            )
            explanation['recommendations'].append("Critical: Install irrigation system or wait for better rainfall season")
        else:
            explanation['agro_risk_factors'].append(
                "   • Has irrigation - partially mitigates drought risk"
            )
    
    if rainfall_12m < 800:
        explanation['agro_risk_factors'].append(
            f"⚠️ **Low Rainfall**: Only {rainfall_12m:.0f}mm in last 12 months (Below optimal)"
        )
        explanation['agro_risk_factors'].append(
            "   • Planting off-cycle with low rainfall increases crop failure risk"
        )
        if not irrigation:
            explanation['recommendations'].append("Delay planting until rainfall improves OR install irrigation")
    
    if climate_suitability < 0.6:
        explanation['agro_risk_factors'].append(
            f"⚠️ **Poor Climate Match**: Climate suitability only {climate_suitability:.0%} for selected crop"
        )
        explanation['recommendations'].append("Consider drought-resistant crop varieties or different crop type")
    
    if soil_suitability < 0.6:
        explanation['agro_risk_factors'].append(
            f"⚠️ **Poor Soil Quality**: Soil suitability only {soil_suitability:.0%}"
        )
        explanation['recommendations'].append("Invest in soil improvement or select crops suited to current soil type")
    
    # 3. MARKET RISK FACTORS
    current_price = farmer_data.get('current_price_kes', 0)
    price_6m_avg = farmer_data.get('price_6m_avg', 0)
    price_volatility = farmer_data.get('price_volatility', 0)
    price_trend = farmer_data.get('price_trend', 'Stable')
    
    if price_trend == 'Falling':
        explanation['market_risk_factors'].append(
            f"⚠️ **Falling Prices**: Market prices are declining (Current: KES {current_price:,.0f}, 6-month avg: KES {price_6m_avg:,.0f})"
        )
        price_drop = ((current_price - price_6m_avg) / price_6m_avg * 100) if price_6m_avg > 0 else 0
        explanation['market_risk_factors'].append(
            f"   • Prices down {abs(price_drop):.1f}% - reduces expected income"
        )
        explanation['recommendations'].append("Join cooperative for collective bargaining power")
    
    if price_volatility > current_price * 0.15:
        explanation['market_risk_factors'].append(
            f"⚠️ **High Price Volatility**: Price fluctuations of KES {price_volatility:,.0f} ({price_volatility/current_price:.0%} of current price)"
        )
        explanation['market_risk_factors'].append(
            "   • Unstable market makes income forecasting unreliable"
        )
    
    # 4. FINANCIAL BEHAVIOR ANALYSIS
    deposits_6m = farmer_data.get('total_deposits_6m', 0)
    withdrawals_6m = farmer_data.get('total_withdrawals_6m', 0)
    avg_transactions = farmer_data.get('avg_monthly_transactions', 0)
    transaction_consistency = farmer_data.get('transaction_consistency', 0)
    account_age = farmer_data.get('account_age_months', 0)
    
    if transaction_consistency < 0.5:
        explanation['financial_behavior'].append(
            f"⚠️ **Irregular M-Pesa Activity**: Transaction consistency only {transaction_consistency:.0%}"
        )
        explanation['financial_behavior'].append(
            "   • Pattern shows sporadic large payments (1-2 times) rather than regular income"
        )
        explanation['financial_behavior'].append(
            "   • Suggests unstable income stream - may struggle with regular loan repayments"
        )
        explanation['recommendations'].append("Build more consistent income before taking loan")
    
    if avg_transactions < 5:
        explanation['financial_behavior'].append(
            f"⚠️ **Low Transaction Activity**: Only {avg_transactions:.0f} transactions per month"
        )
        explanation['financial_behavior'].append(
            "   • Limited financial activity suggests minimal business engagement"
        )
    
    savings_rate = farmer_data.get('savings_rate', 0)
    if savings_rate < 0.1:
        explanation['financial_behavior'].append(
            f"⚠️ **No Savings Buffer**: Savings rate only {savings_rate:.0%}"
        )
        explanation['financial_behavior'].append(
            f"   • Deposits: KES {deposits_6m:,.0f}, Withdrawals: KES {withdrawals_6m:,.0f}"
        )
        explanation['financial_behavior'].append(
            "   • No financial cushion for emergencies or crop failures"
        )
        explanation['recommendations'].append("Build emergency savings (at least 10% of deposits) before borrowing")
    
    # 5. SOCIAL CAPITAL ANALYSIS
    coop_member = farmer_data.get('cooperative_member', False)
    group_farming = farmer_data.get('group_farming', False)
    extension_access = farmer_data.get('extension_access', False)
    training_sessions = farmer_data.get('training_sessions', 0)
    
    social_issues = []
    if not coop_member:
        social_issues.append("Not a SACCO/cooperative member")
    if not group_farming:
        social_issues.append("Does not participate in group farming")
    if not extension_access:
        social_issues.append("No agricultural extension services access")
    if training_sessions < 3:
        social_issues.append(f"Limited training (only {training_sessions} sessions)")
    
    if social_issues:
        explanation['social_capital'].append(
            "⚠️ **Weak Support Network**: Farmer operates largely in isolation"
        )
        for issue in social_issues:
            explanation['social_capital'].append(f"   • {issue}")
        explanation['social_capital'].append(
            "   • Lacks collective bargaining power, shared resources, and technical support"
        )
        explanation['recommendations'].append("Join agricultural cooperative for market access, training, and peer support")
    
    return explanation


def show_home():
    """Home page"""
    
    # Welcome Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <p style="color: #f0f0f0; font-size: 1.1rem; margin-top: 0.5rem;">
            Empowering financial institutions with AI-driven credit assessment for agricultural lending
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("### 🎯 Key Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 180px; border-left: 5px solid #FF9800;">
            <h4 style="color: #FF9800; margin-top: 0;">🤖 AI-Powered</h4>
            <p style="color: #555; font-size: 0.9rem;">Advanced machine learning models analyze 60+ data points to predict credit risk with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 180px; border-left: 5px solid #4CAF50;">
            <h4 style="color: #4CAF50; margin-top: 0;">🌾 Agro-Focused</h4>
            <p style="color: #555; font-size: 0.9rem;">Specialized scoring for agricultural risks including climate, soil, and crop-specific factors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 180px; border-left: 5px solid #2196F3;">
            <h4 style="color: #2196F3; margin-top: 0;">📊 Data-Driven</h4>
            <p style="color: #555; font-size: 0.9rem;">Leverage M-Pesa transaction history, market trends, and social capital for comprehensive assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Credit Score Composition Section
    st.markdown("### 📊 Credit Score Composition")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Create pie chart data
        score_data = pd.DataFrame({
            'Component': ['Repayment Score', 'Agro-Ecological Score', 'Market Score'],
            'Weight': [45, 30, 25],
            'Description': [
                'Income, M-Pesa Transactions, Savings',
                'Climate, Soil, Yield, Irrigation',
                'Prices, Trends, Volatility'
            ]
        })
        
        # Create beautiful donut chart
        fig = go.Figure(data=[go.Pie(
            labels=score_data['Component'],
            values=score_data['Weight'],
            hole=0.5,  # Bigger hole for donut
            marker=dict(
                colors=['#FF9800', '#4CAF50', '#2196F3'],  # Orange, Green, Blue
                line=dict(color='white', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='white', family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Weight: %{value}%<br>%{text}<extra></extra>',
            text=score_data['Description']
        )])
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(t=0, b=0, l=0, r=0),
            annotations=[
                dict(
                    text='100%<br>Total',
                    x=0.5, y=0.5,
                    font=dict(size=28, color='#1976D2', family='Arial Black'),
                    showarrow=False
                )
            ]
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
    # Score Components in horizontal flow below the chart
    st.markdown("#### 📋 Score Components Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card" style="border-left: 4px solid #FF9800;">', unsafe_allow_html=True)
        st.markdown("**💰 Repayment Score (45%)**")
        st.markdown("• Expected income coverage")
        st.markdown("• M-Pesa transaction history")
        st.markdown("• Savings behavior pattern")
        st.markdown("• Social capital & network")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card" style="border-left: 4px solid #4CAF50;">', unsafe_allow_html=True)
        st.markdown("**🌾 Agro-Ecological (30%)**")
        st.markdown("• Climate suitability match")
        st.markdown("• Soil quality assessment")
        st.markdown("• Historical yield performance")
        st.markdown("• Irrigation access status")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card" style="border-left: 4px solid #2196F3;">', unsafe_allow_html=True)
        st.markdown("**� Market Risk (25%)**")
        st.markdown("• Current price trends")
        st.markdown("• Price volatility index")
        st.markdown("• Market access rating")
        st.markdown("• Harvest period pricing")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Quick Start Guide")
    st.markdown("""
    1. **Generate Data**: Create synthetic farmer data for training
    2. **Train Model**: Train machine learning models on the data
    3. **Multiple Evaluations**: Evaluate new loan applications
    4. **Analytics**: View insights and model performance
    """)
    
    # Check system status
    st.markdown("### System Status")
    
    data_exists = os.path.exists(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}")
    model_exists = os.path.exists(f"{MODELS_DIR}/{TRAINED_MODEL_FILE}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if data_exists:
            st.success("✓ Training data available")
        else:
            st.warning("⚠ No training data found. Please generate data first.")
    
    with col2:
        if model_exists:
            st.success("✓ Trained model available")
        else:
            st.warning("⚠ No trained model found. Please train a model first.")


def show_data_generation():
    """Data generation page"""
    
    st.markdown("## Training Data Setup")
    
    # Create tabs for different data input methods
    tab1, tab2 = st.tabs(["📊 Generate Synthetic Data", "📁 Upload Excel Data"])
    
    # Tab 1: Generate Synthetic Data
    with tab1:
        st.markdown("""
        Generate realistic farmer data for model training. The data includes:
        - Farmer demographics and farm characteristics
        - Financial behavior (M-Pesa transactions)
        - Agro-ecological conditions
        - Market indicators
        - Loan applications
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size = st.number_input(
                "Number of farmer records",
                min_value=100,
                max_value=5000,
                value=SYNTHETIC_SAMPLE_SIZE,
                step=100
            )
        
        with col2:
            target_default_rate = st.slider(
                "Target default rate (%)",
                min_value=10,
                max_value=40,
                value=int(BASE_DEFAULT_RATE * 100),
                step=5
            )
        
        if st.button("Generate Data", type="primary", key="generate_btn"):
            with st.spinner("Generating synthetic data..."):
                generator = SyntheticDataGenerator(sample_size=sample_size)
                df = generator.generate_complete_dataset()
                
                # Save data
                os.makedirs(DATA_DIR, exist_ok=True)
                output_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
                df.to_csv(output_path, index=False)
                
                st.success(f"✓ Generated {len(df)} farmer records")
                st.success(f"✓ Data saved to {output_path}")
                
                # Show summary
                st.markdown("### Data Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Default Rate", f"{df['default_label'].mean():.1%}")
                with col3:
                    st.metric("Avg Farm Size", f"{df['farm_size'].mean():.2f} acres")
                with col4:
                    st.metric("Avg Loan Amount", f"KES {df['loan_amount_kes'].mean():,.0f}")
                
                # Show sample data
                st.markdown("### Sample Data")
                st.dataframe(df.head(10))
                
                # Show distributions
                st.markdown("### Data Distributions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(df, x='age', title='Age Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig = px.pie(df, names='education_level', title='Education Levels')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(df, x='farm_size', title='Farm Size Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Handle crop_type column name variations
                    crop_col = 'crop_type' if 'crop_type' in df.columns else ('crop_type_x' if 'crop_type_x' in df.columns else None)
                    if crop_col:
                        fig = px.pie(df, names=crop_col, title='Crop Types')
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Upload Excel Data
    with tab2:
        st.markdown("""
        Upload your own farmer data in Excel format (.xlsx or .xls).
        
        **Requirements:**
        - Excel file with farmer information
        - Must include essential columns (farmer_id, age, farm_size, crop_type, loan_amount_kes, etc.)
        - The system will validate and process your data
        """)
        
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing farmer data"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Reading Excel file..."):
                    # Read Excel file
                    df_upload = pd.read_excel(uploaded_file)
                    
                    st.success(f"✓ Successfully read {len(df_upload)} records from Excel file")
                    
                    # Show file information
                    st.markdown("### Uploaded File Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(df_upload))
                    with col2:
                        st.metric("Total Columns", len(df_upload.columns))
                    with col3:
                        st.metric("File Name", uploaded_file.name)
                    
                    # Show column names
                    st.markdown("### Detected Columns")
                    st.write(", ".join(df_upload.columns.tolist()))
                    
                    # Show preview
                    st.markdown("### Data Preview")
                    st.dataframe(df_upload.head(10))
                    
                    # Validate essential columns
                    required_columns = ['farmer_id', 'age', 'farm_size', 'crop_type', 'loan_amount_kes']
                    missing_columns = [col for col in required_columns if col not in df_upload.columns]
                    
                    if missing_columns:
                        st.warning(f"⚠ Missing recommended columns: {', '.join(missing_columns)}")
                        st.info("The system will work best with all required columns, but you can proceed if you have most of the data.")
                    else:
                        st.success("✓ All essential columns detected!")
                    
                    # Option to save uploaded data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        save_option = st.radio(
                            "Save as:",
                            ["Replace existing training data", "Save as new file"],
                            help="Choose whether to replace the current training data or save as a separate file"
                        )
                    
                    with col2:
                        if save_option == "Save as new file":
                            custom_filename = st.text_input(
                                "Custom filename (without extension)",
                                value="uploaded_farmers",
                                help="Enter a name for your uploaded data file"
                            )
                    
                    if st.button("Save Uploaded Data", type="primary", key="upload_save_btn"):
                        with st.spinner("Saving data..."):
                            os.makedirs(DATA_DIR, exist_ok=True)
                            
                            if save_option == "Replace existing training data":
                                output_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
                            else:
                                output_path = f"{DATA_DIR}/{custom_filename}.csv"
                            
                            # Save as CSV
                            df_upload.to_csv(output_path, index=False)
                            
                            st.success(f"✓ Data saved successfully to {output_path}")
                            st.info(f"💡 You can now use this data for model training!")
                            
                            # Show basic statistics if default_label exists
                            if 'default_label' in df_upload.columns:
                                st.markdown("### Default Statistics")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Default Rate", f"{df_upload['default_label'].mean():.1%}")
                                with col2:
                                    st.metric("Non-Default Rate", f"{(1 - df_upload['default_label'].mean()):.1%}")
                
            except Exception as e:
                st.error(f"❌ Error reading Excel file: {str(e)}")
                st.info("Please ensure your file is a valid Excel format (.xlsx or .xls) and contains the necessary data.")
    
    # Show existing data if available
    if os.path.exists(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"):
        st.markdown("---")
        st.markdown("### Existing Data")
        
        df = pd.read_csv(f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(df))
        with col2:
            if 'default_label' in df.columns:
                st.metric("Default Rate", f"{df['default_label'].mean():.1%}")
            else:
                st.metric("Default Label", "Not Available")
        with col3:
            st.metric("Features", len(df.columns))
        with col4:
            st.metric("File Size", f"{os.path.getsize(f'{DATA_DIR}/{SYNTHETIC_DATA_FILE}') / 1024:.1f} KB")


def show_training():
    """Model training page"""
    
    st.markdown("## Train Credit Scoring Model")
    
    # Check if data exists
    data_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
    if not os.path.exists(data_path):
        st.error("⚠ No training data found. Please generate data first.")
        return
    
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=40,
            value=int(TEST_SIZE * 100),
            step=5
        ) / 100
        
        use_smote = st.checkbox("Use SMOTE for class balancing", value=True)
    
    with col2:
        models_to_train = st.multiselect(
            "Select models to train",
            ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
            default=['Random Forest', 'XGBoost', 'LightGBM']
        )
    
    if st.button("Start Training", type="primary"):
        if not models_to_train:
            st.error("Please select at least one model to train")
            return
        
        with st.spinner("Loading and preprocessing data..."):
            # Load data
            df = pd.read_csv(data_path)
            
            # Feature engineering
            engineer = FeatureEngineer()
            df = engineer.prepare_for_modeling(df)
            
            feature_cols = engineer.get_all_model_features()
            
            st.info(f"✓ Loaded {len(df)} records with {len(feature_cols)} features")
        
        with st.spinner("Training models... This may take a few minutes."):
            # Train models
            trainer = CreditModelTrainer()
            
            X_train, X_test, y_train, y_test = trainer.prepare_data(
                df, feature_cols, test_size=test_size, use_smote=use_smote
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = trainer.train_all_models(
                X_train, X_test, y_train, y_test,
                models_to_train=models_to_train
            )
            
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            # Save model
            trainer.save_model()
            
            st.success(f"✓ Best model: {trainer.best_model_name}")
            st.success(f"✓ Model saved to {MODELS_DIR}/")
        
        # Show results
        st.markdown("### Training Results")
        
        results_display = results[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].copy()
        results_display.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        st.dataframe(results_display.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']))
        
        # Visualize model comparison
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=results['model_name'],
                y=results[metric]
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("### Top 15 Important Features")
        
        feature_imp = trainer.get_feature_importance(top_n=15)
        
        fig = px.bar(
            feature_imp,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)


def show_predictions():
    """Batch predictions page"""
    
    st.markdown("## Make Batch Predictions")
    
    # Check if model exists
    model_path = f"{MODELS_DIR}/{TRAINED_MODEL_FILE}"
    if not os.path.exists(model_path):
        st.error("⚠ No trained model found. Please train a model first.")
        return
    
    st.markdown("### Load Application Data")
    
    # Option to use test data or upload file
    data_source = st.radio(
        "Select data source",
        ["Use existing test data", "Upload CSV file"]
    )
    
    df = None
    
    if data_source == "Use existing test data":
        data_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
        if os.path.exists(data_path):
            df_full = pd.read_csv(data_path)
            
            sample_size = st.slider(
                "Number of applications to evaluate",
                min_value=10,
                max_value=min(500, len(df_full)),
                value=min(100, len(df_full)),
                step=10
            )
            
            df = df_full.sample(n=sample_size, random_state=RANDOM_SEED)
            st.info(f"✓ Loaded {len(df)} applications from test data")
        else:
            st.error("No test data found")
            return
    
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.info(f"✓ Loaded {len(df)} applications from uploaded file")
    
    if df is not None and st.button("Run Predictions", type="primary"):
        with st.spinner("Making predictions..."):
            predictor = CreditPredictor()
            predictions = predictor.predict_with_scores(df)
            
            st.success("✓ Predictions complete!")
        
        # Summary
        st.markdown("### Prediction Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Applications", len(predictions))
        with col2:
            approve_pct = (predictions['final_decision'] == 'Approve').mean() * 100
            st.metric("Approved", f"{approve_pct:.1f}%")
        with col3:
            decline_pct = (predictions['final_decision'] == 'Decline').mean() * 100
            st.metric("Declined", f"{decline_pct:.1f}%")
        with col4:
            review_pct = (predictions['final_decision'] == 'Manual Review').mean() * 100
            st.metric("Manual Review", f"{review_pct:.1f}%")
        
        # Decision distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                predictions,
                names='final_decision',
                title='Final Decision Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                predictions,
                x='final_credit_score',
                nbins=20,
                title='Credit Score Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show predictions
        st.markdown("### Detailed Predictions")
        
        display_cols = [
            'farmer_id', 'name', 'final_credit_score',
            'agro_score', 'market_score', 'repayment_score',
            'default_probability', 'final_decision'
        ]
        
        available_cols = [col for col in display_cols if col in predictions.columns]
        st.dataframe(predictions[available_cols])
        
        # Download button
        csv = predictions.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # DETAILED EXPLANATION FOR SELECTED APPLICATION
        st.markdown("---")
        st.markdown("### 🔍 View Detailed Explanation")
        st.info("Select a farmer to see detailed breakdown of their credit decision")
        
        # Create dropdown with farmer names/IDs
        farmer_options = predictions['farmer_id'].tolist() if 'farmer_id' in predictions.columns else list(range(len(predictions)))
        if 'name' in predictions.columns:
            farmer_display = [f"{predictions.iloc[i]['farmer_id']} - {predictions.iloc[i]['name']}" for i in range(len(predictions))]
        else:
            farmer_display = [str(f) for f in farmer_options]
        
        selected_farmer_idx = st.selectbox(
            "Select farmer for detailed explanation:",
            range(len(predictions)),
            format_func=lambda x: farmer_display[x]
        )
        
        if st.button("Show Detailed Explanation", type="primary", key="batch_explain_btn"):
            selected_data = predictions.iloc[selected_farmer_idx].to_dict()
            
            # Generate explanation
            with st.spinner("Generating detailed explanation..."):
                detailed_explanation = generate_detailed_explanation(selected_data, {})
            
            st.markdown("---")
            st.markdown(f"## 📋 Detailed Explanation for {farmer_display[selected_farmer_idx]}")
            
            # Show decision and scores
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                decision = selected_data.get('final_decision', 'N/A')
                if decision == 'Approve':
                    st.success(f"**Decision:** {decision}")
                elif decision == 'Decline':
                    st.error(f"**Decision:** {decision}")
                else:
                    st.warning(f"**Decision:** {decision}")
            with col2:
                st.metric("Credit Score", f"{selected_data.get('final_credit_score', 0):.1f}/100")
            with col3:
                st.metric("Default Prob", f"{selected_data.get('default_probability', 0):.1%}")
            with col4:
                st.metric("Loan Amount", f"KES {selected_data.get('loan_amount_kes', 0):,.0f}")
            
            # Income Analysis
            if detailed_explanation['income_analysis']:
                st.markdown("### 💰 Income vs Loan Analysis")
                for item in detailed_explanation['income_analysis']:
                    if '✓' in item:
                        st.success(item)
                    else:
                        st.error(item)
            
            # Agro-Ecological Risks
            if detailed_explanation['agro_risk_factors']:
                st.markdown("### 🌾 Agro-Ecological Risk Factors")
                for item in detailed_explanation['agro_risk_factors']:
                    if '✓' in item:
                        st.success(item)
                    else:
                        st.warning(item)
            
            # Market Risks
            if detailed_explanation['market_risk_factors']:
                st.markdown("### 📈 Market Risk Factors")
                for item in detailed_explanation['market_risk_factors']:
                    if '✓' in item:
                        st.success(item)
                    else:
                        st.warning(item)
            
            # Financial Behavior
            if detailed_explanation['financial_behavior']:
                st.markdown("### 💳 M-Pesa Financial Behavior")
                for item in detailed_explanation['financial_behavior']:
                    if '✓' in item:
                        st.success(item)
                    else:
                        st.warning(item)
            
            # Social Capital
            if detailed_explanation['social_capital']:
                st.markdown("### 🤝 Social Capital & Support Network")
                for item in detailed_explanation['social_capital']:
                    if '✓' in item:
                        st.success(item)
                    else:
                        st.warning(item)
            
            # Recommendations
            if detailed_explanation['recommendations']:
                st.markdown("### 💡 Actionable Recommendations")
                for i, rec in enumerate(detailed_explanation['recommendations'], 1):
                    st.info(f"{i}. {rec}")
        
        # If actual labels are available, show evaluation
        if 'default_label' in predictions.columns:
            st.markdown("### Model Evaluation")
            
            metrics = predictor.evaluate_predictions(predictions)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            with col5:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")


def show_analytics():
    """Analytics dashboard"""
    
    st.markdown("## Analytics Dashboard")
    
    # Load data if available
    data_path = f"{DATA_DIR}/{SYNTHETIC_DATA_FILE}"
    if not os.path.exists(data_path):
        st.error("⚠ No data found. Please generate data first.")
        return
    
    df = pd.read_csv(data_path)
    
    # Calculate scores if not already done
    if 'final_credit_score' not in df.columns:
        with st.spinner("Calculating credit scores..."):
            scorer = CreditScoringEngine()
            df = scorer.score_applications(df)
    
    # Overview metrics
    st.markdown("### Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Farmers", len(df))
    with col2:
        st.metric("Avg Credit Score", f"{df['final_credit_score'].mean():.1f}")
    with col3:
        st.metric("Total Loan Amount", f"KES {df['loan_amount_kes'].sum():,.0f}")
    with col4:
        default_rate = df['default_label'].mean() * 100 if 'default_label' in df.columns else 0
        st.metric("Default Rate", f"{default_rate:.1f}%")
    
    # Score distributions
    st.markdown("### Score Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df,
            y=['agro_score', 'market_score', 'repayment_score'],
            title='Score Component Distributions'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='final_credit_score',
            color='credit_decision',
            title='Credit Score by Decision',
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis
    st.markdown("### Geographic Analysis")
    
    county_stats = df.groupby('county').agg({
        'farmer_id': 'count',
        'final_credit_score': 'mean',
        'loan_amount_kes': 'sum'
    }).reset_index()
    
    county_stats.columns = ['County', 'Farmers', 'Avg Score', 'Total Loans']
    
    fig = px.bar(
        county_stats.sort_values('Farmers', ascending=False).head(10),
        x='County',
        y='Farmers',
        title='Top 10 Counties by Number of Farmers'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Crop analysis
    st.markdown("### Crop Analysis")
    
    # Handle crop_type column name variations
    crop_col = 'crop_type' if 'crop_type' in df.columns else ('crop_type_x' if 'crop_type_x' in df.columns else None)
    
    if crop_col:
        crop_stats = df.groupby(crop_col).agg({
            'farmer_id': 'count',
            'final_credit_score': 'mean',
            'historical_yield': 'mean'
        }).reset_index()
        
        crop_stats.columns = ['Crop', 'Farmers', 'Avg Score', 'Avg Yield']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(crop_stats, values='Farmers', names='Crop', title='Farmers by Crop Type')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(crop_stats, x='Crop', y='Avg Score', title='Average Credit Score by Crop')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Crop type information not available in the dataset.")


def show_single_application():
    """Single Evaluations evaluation"""
    
    st.markdown("## Evaluate Single Evaluations")
    
    # Check if model exists
    model_path = f"{MODELS_DIR}/{TRAINED_MODEL_FILE}"
    if not os.path.exists(model_path):
        st.error("⚠ No trained model found. Please train a model first.")
        return
    
    st.markdown("### Enter Farmer Details")
    
    # Create form
    with st.form("application_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demographics**")
            name = st.text_input("Farmer Name", "John Doe")
            age = st.number_input("Age", 18, 80, 42)
            gender = st.selectbox("Gender", ["M", "F"])
            education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
            dependents = st.number_input("Dependents", 0, 15, 3)
            
            st.markdown("**Farm Details**")
            farm_size = st.number_input("Farm Size (acres)", 0.1, 50.0, 2.5)
            crop_type = st.selectbox("Crop Type", ["Maize", "Beans", "Potatoes", "Vegetables", "Other"])
            historical_yield = st.number_input("Historical Yield (tons/acre)", 0.1, 10.0, 1.5)
            irrigation = st.checkbox("Has Irrigation Access")
            years_farming = st.number_input("Years Farming", 1, 50, 10)
        
        with col2:
            st.markdown("**Financial Behavior**")
            account_age = st.number_input("M-Pesa Account Age (months)", 1, 120, 24)
            avg_transactions = st.number_input("Avg Monthly Transactions", 0, 100, 20)
            avg_trans_value = st.number_input("Avg Transaction Value (KES)", 0, 50000, 2000)
            deposits_6m = st.number_input("Total Deposits 6m (KES)", 0, 500000, 50000)
            withdrawals_6m = st.number_input("Total Withdrawals 6m (KES)", 0, 500000, 45000)
            
            st.markdown("**Community**")
            coop_member = st.checkbox("Cooperative Member")
            training_sessions = st.number_input("Training Sessions Attended", 0, 50, 5)
            group_farming = st.checkbox("Participates in Group Farming")
            
            st.markdown("**Loan Details**")
            loan_amount = st.number_input("Loan Amount (KES)", 1000, 500000, 25000)
            repayment_months = st.selectbox("Repayment Period (months)", [3, 4, 6, 9, 12])
        
        submitted = st.form_submit_button("Evaluate Application", type="primary")
    
    if submitted:
        # Create farmer record
        farmer_data = {
            'farmer_id': 'APP00001',
            'name': name,
            'age': age,
            'gender': gender,
            'education_level': education,
            'dependents': dependents,
            'farm_size': farm_size,
            'crop_type': crop_type,
            'historical_yield': historical_yield,
            'irrigation_access': irrigation,
            'years_farming': years_farming,
            'account_age_months': account_age,
            'avg_monthly_transactions': avg_transactions,
            'avg_transaction_value': avg_trans_value,
            'total_deposits_6m': deposits_6m,
            'total_withdrawals_6m': withdrawals_6m,
            'cooperative_member': coop_member,
            'training_sessions': training_sessions,
            'group_farming': group_farming,
            'extension_access': True,
            'loan_amount_kes': loan_amount,
            'repayment_months': repayment_months,
            # Add defaults for required fields
            'county': 'Nakuru',
            'subcounty': 'Nakuru North',
            'latitude': 0.0,
            'longitude': 36.0,
            'id_number': '12345678',
            'phone_number': '+254712345678',
            'agro_zone': 'UM2',
            'land_ownership': 'Owned',
            'rainfall_12m_mm': 1200,
            'rainfall_variability': 0.2,
            'drought_index': 0.3,
            'soil_type': 'Loam',
            'soil_suitability': 0.8,
            'temperature_min': 15,
            'temperature_max': 28,
            'climate_suitability': 0.85,
            'current_price_kes': 45,
            'price_6m_avg': 43,
            'price_volatility': 5,
            'expected_harvest_month': 'October',
            'projected_harvest_price': 46,
            'price_trend': 'Stable',
            'cooperative_name': 'Test Coop' if coop_member else None,
            'years_in_coop': 3 if coop_member else None,
            'application_id': 'APP00001',
            'loan_purpose': 'Seeds',
            'planting_date': datetime.now().date(),
            'harvest_date': (datetime.now() + timedelta(days=120)).date(),
            'application_date': datetime.now().date(),
            'savings_rate': deposits_6m / (deposits_6m + withdrawals_6m) if (deposits_6m + withdrawals_6m) > 0 else 0,
            'peak_transaction_months': 'Oct,Nov,Dec',
            'transaction_consistency': 0.75
        }
        
        with st.spinner("Evaluating application..."):
            predictor = CreditPredictor()
            result = predictor.predict_single(farmer_data)
            detailed_explanation = generate_detailed_explanation(farmer_data, result)
        
        # Display results
        st.markdown("---")
        st.markdown("## Evaluation Results")
        
        # Decision banner
        decision = result['decisions']['final']
        if decision == 'Approve':
            st.markdown(f'<div class="success-box"><h2 style="color: #2E7D32;">✓ {decision}</h2></div>', unsafe_allow_html=True)
        elif decision == 'Decline':
            st.markdown(f'<div class="error-box"><h2 style="color: #C62828;">✗ {decision}</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box"><h2 style="color: #F57C00;">⚠ {decision}</h2></div>', unsafe_allow_html=True)
        
        # Overall Scores with weights
        st.markdown("### 📊 Overall Credit Scores")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Score", f"{result['credit_scores']['final_score']:.1f}/100")
        with col2:
            st.metric("Repayment Score (45%)", f"{result['credit_scores']['repayment_score']:.1f}/100")
        with col3:
            st.metric("Agro Score (30%)", f"{result['credit_scores']['agro_score']:.1f}/100")
        with col4:
            st.metric("Market Score (25%)", f"{result['credit_scores']['market_score']:.1f}/100")
        
        st.markdown("---")
        
        # Detailed Component Scores Breakdown
        st.markdown("### 📋 Detailed Component Scores & Coverage")
        
        # Repayment Components
        st.markdown("#### 💰 Repayment Score Components (Weight: 45%)")
        rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)
        
        with rep_col1:
            st.metric(
                "Income Coverage (40%)", 
                f"{result['repayment_components']['income_coverage_score']:.1f}",
                help="Expected income vs loan amount"
            )
        with rep_col2:
            st.metric(
                "Transaction Health (35%)", 
                f"{result['repayment_components']['transaction_health_score']:.1f}",
                help="M-Pesa transaction patterns"
            )
        with rep_col3:
            st.metric(
                "Savings Pattern (15%)", 
                f"{result['repayment_components']['savings_pattern_score']:.1f}",
                help="Savings behavior score"
            )
        with rep_col4:
            st.metric(
                "Social Capital (10%)", 
                f"{result['repayment_components']['social_capital_score']:.1f}",
                help="Cooperative membership & training"
            )
        
        # Agro Components
        st.markdown("#### 🌾 Agro-Ecological Score Components (Weight: 30%)")
        agro_col1, agro_col2, agro_col3, agro_col4 = st.columns(4)
        
        with agro_col1:
            st.metric(
                "Climate Match (35%)", 
                f"{result['agro_components']['climate_match']:.1f}",
                help="Climate suitability for crop"
            )
        with agro_col2:
            st.metric(
                "Soil Match (30%)", 
                f"{result['agro_components']['soil_match']:.1f}",
                help="Soil quality match"
            )
        with agro_col3:
            st.metric(
                "Yield Performance (25%)", 
                f"{result['agro_components']['yield_performance']:.1f}",
                help="Historical yield vs zone average"
            )
        with agro_col4:
            st.metric(
                "Irrigation (10%)", 
                f"{result['agro_components']['irrigation_score']:.1f}",
                help="Irrigation access"
            )
        
        # Market Components
        st.markdown("#### 📈 Market Score Components (Weight: 25%)")
        mkt_col1, mkt_col2, mkt_col3, mkt_col4 = st.columns(4)
        
        with mkt_col1:
            st.metric(
                "Price Trend (40%)", 
                f"{result['market_components']['price_trend_score']:.1f}",
                help="Current crop price trends"
            )
        with mkt_col2:
            st.metric(
                "Harvest Price (35%)", 
                f"{result['market_components']['harvest_price_score']:.1f}",
                help="Expected harvest period pricing"
            )
        with mkt_col3:
            st.metric(
                "Price Volatility (15%)", 
                f"{result['market_components']['volatility_score']:.1f}",
                help="Price stability index"
            )
        with mkt_col4:
            st.metric(
                "Market Access (10%)", 
                f"{result['market_components']['market_access_score']:.1f}",
                help="Market access & cooperative"
            )
        
        # Risk assessment
        st.markdown("---")
        st.markdown("### 🎲 ML Risk Assessment")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.metric("Default Probability", f"{result['ml_prediction']['default_probability']:.1%}")
        with risk_col2:
            st.metric("ML Prediction", result['ml_prediction']['prediction'])
        with risk_col3:
            st.metric("ML Decision", result['ml_prediction']['decision'])
        
        # DETAILED EXPLAINABILITY SECTION
        st.markdown("---")
        st.markdown("## 📋 Detailed Decision Explanation")
        
        # Income Analysis
        if detailed_explanation['income_analysis']:
            st.markdown("### 💰 Income vs Loan Analysis")
            for item in detailed_explanation['income_analysis']:
                if '✓' in item:
                    st.success(item)
                else:
                    st.error(item)
        
        # Agro-Ecological Risks
        if detailed_explanation['agro_risk_factors']:
            st.markdown("### 🌾 Agro-Ecological Risk Factors")
            for item in detailed_explanation['agro_risk_factors']:
                if '✓' in item:
                    st.success(item)
                else:
                    st.warning(item)
        
        # Market Risks
        if detailed_explanation['market_risk_factors']:
            st.markdown("### 📈 Market Risk Factors")
            for item in detailed_explanation['market_risk_factors']:
                if '✓' in item:
                    st.success(item)
                else:
                    st.warning(item)
        
        # Financial Behavior
        if detailed_explanation['financial_behavior']:
            st.markdown("### 💳 M-Pesa Financial Behavior")
            for item in detailed_explanation['financial_behavior']:
                if '✓' in item:
                    st.success(item)
                else:
                    st.warning(item)
        
        # Social Capital
        if detailed_explanation['social_capital']:
            st.markdown("### 🤝 Social Capital & Support Network")
            for item in detailed_explanation['social_capital']:
                if '✓' in item:
                    st.success(item)
                else:
                    st.warning(item)
        
        # Recommendations
        if detailed_explanation['recommendations']:
            st.markdown("### 💡 Actionable Recommendations")
            for i, rec in enumerate(detailed_explanation['recommendations'], 1):
                st.info(f"{i}. {rec}")


if __name__ == "__main__":
    main()
