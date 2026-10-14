"""
RailWay AI - Single-Page Streamlit Chatbot
Domain-specific Question Answering Assistant for Indian Railways Reservation & Ticketing.
Classifies user queries with a custom-trained PyTorch Bidirectional LSTM model
and retrieves authoritative, curated domain knowledge.
"""

import os
import json
import streamlit as st
import pandas as pd
from inference import RailwayAssistant

# Page Configuration
st.set_page_config(
    page_title="RailWay AI - Railway Domain Assistant",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Responsiveness
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .hero-container {
        background: linear-gradient(135deg, #0d3b66 0%, #001e3d 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        color: #f4d06f;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #e0e6ed;
        margin-top: 0.4rem;
        line-height: 1.4;
    }
    .hero-badges {
        margin-top: 0.8rem;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge-chip {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    
    /* Category Badges */
    .cat-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .cat-WAITLIST { background-color: #ffe8cc; color: #d9480f; border: 1px solid #ffd8a8; }
    .cat-RAC { background-color: #e6fcf5; color: #0ca678; border: 1px solid #c3fae8; }
    .cat-PNR_STATUS { background-color: #e7f5ff; color: #1971c2; border: 1px solid #d0ebff; }
    .cat-CANCELLATION { background-color: #fff5f5; color: #e03131; border: 1px solid #ffe3e3; }
    .cat-REFUND { background-color: #ebfbee; color: #2f9e44; border: 1px solid #d3f9d8; }
    .cat-BOOKING_ISSUES { background-color: #f3f0ff; color: #6741d9; border: 1px solid #e5dbff; }
    .cat-OUT_OF_DOMAIN { background-color: #f8f9fa; color: #495057; border: 1px solid #dee2e6; }
    
    /* Disclaimer Box */
    .disclaimer-card {
        background-color: #fff9db;
        border-left: 4px solid #f59f00;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #744210;
        margin: 0.8rem 0;
        line-height: 1.45;
    }
    
    /* Sidebar Styles */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model and Assistant Instance
@st.cache_resource(show_spinner="Loading trained PyTorch LSTM weights & vocabulary...")
def get_assistant():
    return RailwayAssistant(artifacts_dir="model_artifacts", kb_path="knowledge_base.json")

try:
    assistant = get_assistant()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🚆 RailWay AI")
    st.caption("Railway Domain Question-Answering Model")
    
    st.divider()
    
    # Model Architecture Card
    st.markdown("#### 🧠 Model Architecture")
    if model_loaded:
        metrics_file = os.path.join("model_artifacts", "evaluation_metrics.json")
        test_acc_str = "91.1%"
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    m = json.load(f)
                    test_acc_str = f"{m.get('test_accuracy', 0.9111)*100:.1f}%"
            except Exception:
                pass
                
        st.markdown(f"""
        - **Classifier**: 2-Layer Bidirectional LSTM
        - **Weights**: Randomly Initialized (Trained from Scratch)
        - **Test Accuracy**: **`{test_acc_str}`**
        - **Vocabulary**: `{assistant.config.get('vocab_size', 749)}` unique tokens
        - **Embedding Dim**: `64` | **Hidden Dim**: `64`
        - **External APIs**: None (100% Offline PyTorch)
        """)
    else:
        st.error("Model artifacts not loaded.")
        
    st.divider()
    
    # Supported Categories
    st.markdown("#### 📂 Supported Categories")
    st.markdown("""
    1. ⏳ **WAITLIST** (GNWL, RLWL, PQWL)
    2. 💺 **RAC** (Berth allocation, rights)
    3. 🎫 **PNR_STATUS** (Charting timings, codes)
    4. ❌ **CANCELLATION** (Deductions, slabs, TDR)
    5. 💸 **REFUND** (Auto-refund, timelines)
    6. ⚠️ **BOOKING_ISSUES** (Tatkal hours, failed payments)
    """)
    
    st.divider()
    
    # Interactive Controls
    st.markdown("#### ⚙️ Settings")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Queries classified with confidence below this threshold are flagged as Out of Domain."
    )
    show_debug = st.checkbox("Show Classification Details & Probabilities", value=True)
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    # Legal & Academic Disclaimer
    st.markdown("""
    <div class="disclaimer-card">
        <b>Academic Project Disclaimer:</b><br>
        RailWay AI explains official Indian Railways reservation rules. It does <b>NOT</b> connect to live CRIS/IRCTC booking databases and does not claim to provide real-time PNR confirmation probabilities.
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Official Links:")
    st.markdown("- [IRCTC Official Booking](https://www.irctc.co.in)\n- [Indian Railways PNR Enquiry](https://www.indianrail.gov.in/enquiry/PNR/PnrEnquiry.html)")

# Main Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚆 RailWay AI: Railway Domain Intelligence Assistant</div>
    <div class="hero-subtitle">
        An academic domain-specific Question Answering assistant powered by a pure PyTorch 
        Bidirectional LSTM text classifier trained strictly from randomly initialized weights.
    </div>
    <div class="hero-badges">
        <span class="badge-chip">⚡ PyTorch BiLSTM</span>
        <span class="badge-chip">🎯 91.1% Test Accuracy</span>
        <span class="badge-chip">🔒 Zero External AI APIs</span>
        <span class="badge-chip">📚 Curated IRCTC Knowledge Base</span>
        <span class="badge-chip">🛡️ Out-of-Domain Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sample Query Chips for Fast Faculty Testing
st.markdown("##### 💡 Quick Test Prompts (Click to test):")
sample_cols = st.columns(3)
sample_prompts = [
    ("💺 RAC Rights", "What does RAC mean and do I get a full berth?"),
    ("⏳ Waitlist Rules", "Can I travel in sleeper class with a waitlisted e-ticket?"),
    ("🎫 PNR Status", "Check status for PNR 8472910392"),
    ("❌ Cancellation", "What are the cancellation charges for 3rd AC 24 hours before train?"),
    ("💸 Refund / Failed", "Money debited from bank account but ticket was not generated"),
    ("🚫 Out of Domain", "How to bake a chocolate cake?")
]

prompt_to_submit = None
for idx, (label, text) in enumerate(sample_prompts):
    col = sample_cols[idx % 3]
    if col.button(f"{label}", key=f"sample_{idx}", use_container_width=True):
        prompt_to_submit = text

# Initialize Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste! I am **RailWay AI**, your intelligent Indian Railways domain assistant.\n\n"
                "I can help clarify questions on **Waitlist rules (GNWL/RLWL/PQWL)**, **RAC rights**, "
                "**PNR charting timelines**, **Cancellation fee slabs**, **Refund processes**, and **IRCTC booking issues**.\n\n"
                "*How may I assist with your railway journey today?*"
            ),
            "category": "WELCOME",
            "confidence": 1.0,
            "is_out_of_domain": False
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🧑‍💻" if role == "user" else "🚆"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and msg.get("category") != "WELCOME":
            cat = msg.get("category", "GENERAL")
            conf = msg.get("confidence", 0.0)
            ood = msg.get("is_out_of_domain", False)
            
            badge_class = f"cat-{cat}"
            badge_label = f"Category: {cat}" if not ood else "Category: OUT OF DOMAIN"
            
            st.markdown(
                f'<span class="cat-badge {badge_class}">{badge_label} • Confidence: {conf*100:.1f}%</span>',
                unsafe_allow_html=True
            )
            st.markdown(msg["content"])
            
            # Show Probability Distribution
            if show_debug and "all_probabilities" in msg:
                with st.expander("🔍 Model Prediction Breakdown & Class Probabilities"):
                    probs_df = pd.DataFrame(
                        list(msg["all_probabilities"].items()),
                        columns=["Category", "Softmax Probability"]
                    ).sort_values(by="Softmax Probability", ascending=False)
                    st.dataframe(
                        probs_df.style.format({"Softmax Probability": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.markdown(msg["content"])

# Handle User Input
user_query = st.chat_input("Ask a question about railway reservation, waitlist, RAC, cancellation, or refunds...")

# If sample prompt button was clicked, override user_query
if prompt_to_submit:
    user_query = prompt_to_submit

if user_query:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)
        
    # 2. Generate Assistant Response
    with st.chat_message("assistant", avatar="🚆"):
        if not model_loaded:
            st.error(f"Cannot perform inference: {load_error}")
        else:
            with st.spinner("Analyzing question with PyTorch LSTM Classifier..."):
                result = assistant.predict(user_query, confidence_threshold=conf_thresh)
                
            cat = result.get("category", "OUT_OF_DOMAIN")
            conf = result.get("confidence", 0.0)
            ood = result.get("is_out_of_domain", False)
            response_text = result.get("response_text", "")
            
            badge_class = f"cat-{cat}"
            badge_label = f"Category: {cat}" if not ood else "Category: OUT OF DOMAIN"
            
            st.markdown(
                f'<span class="cat-badge {badge_class}">{badge_label} • Confidence: {conf*100:.1f}%</span>',
                unsafe_allow_html=True
            )
            st.markdown(response_text)
            
            if show_debug and "all_probabilities" in result:
                with st.expander("🔍 Model Prediction Breakdown & Class Probabilities"):
                    probs_df = pd.DataFrame(
                        list(result["all_probabilities"].items()),
                        columns=["Category", "Softmax Probability"]
                    ).sort_values(by="Softmax Probability", ascending=False)
                    st.dataframe(
                        probs_df.style.format({"Softmax Probability": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
            # Save to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "category": cat,
                "confidence": conf,
                "is_out_of_domain": ood,
                "all_probabilities": result.get("all_probabilities", {})
            })
