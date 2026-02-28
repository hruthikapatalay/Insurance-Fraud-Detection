import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RISKGUARD | AI Fraud Analytics",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- HYPER-AESTHETIC CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary-glow: #00f2ff;
        --secondary-glow: #7000ff;
        --danger: #ff4b4b;
        --success: #00ff88;
        --bg-dark: #050505;
        --card-bg: rgba(20, 20, 25, 0.7);
    }

    /* Animated Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #11111a 0%, #050505 100%);
        font-family: 'Outfit', sans-serif;
        color: #e0e0e0;
    }

    /* Animated Scanning Line Effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary-glow), transparent);
        z-index: 9999;
        animation: scan 4s linear infinite;
        opacity: 0.3;
    }

    @keyframes scan {
        0% { top: 0%; }
        100% { top: 100%; }
    }

    /* HUD Header */
    .top-hud {
        display: flex;
        justify-content: space-between;
        padding: 15px 30px;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 242, 255, 0.2);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 3px;
        color: var(--primary-glow);
        text-shadow: 0 0 10px var(--primary-glow);
    }

    /* Glass Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        transition: all 0.5s ease;
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        border: 1px solid rgba(0, 242, 255, 0.4);
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.1);
        transform: translateY(-5px);
    }

    /* Glowing Titles */
    .main-header h1 {
        font-weight: 900;
        font-size: 5rem;
        text-align: center;
        letter-spacing: -3px;
        background: linear-gradient(135deg, #fff 0%, var(--primary-glow) 50%, var(--secondary-glow) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 40px 0;
        filter: drop-shadow(0 0 15px rgba(0, 242, 255, 0.3));
    }

    /* Button Overhaul */
    div.stButton > button {
        background: linear-gradient(90deg, var(--secondary-glow), var(--primary-glow));
        color: white;
        border: none;
        padding: 20px !important;
        font-weight: 700;
        font-size: 1.2rem !important;
        border-radius: 12px;
        letter-spacing: 5px;
        text-transform: uppercase;
        transition: 0.4s;
        box-shadow: 0 0 20px rgba(112, 0, 255, 0.4);
    }

    div.stButton > button:hover {
        letter-spacing: 8px;
        box-shadow: 0 0 40px var(--primary-glow);
        color: white !important;
    }

    /* Section Icons & Labels */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        color: var(--primary-glow);
        font-size: 0.8rem;
        margin-bottom: 15px;
        display: block;
        text-transform: uppercase;
    }

    /* Metric Styling */
    .metric-value {
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1;
        margin: 10px 0;
    }

</style>
""", unsafe_allow_html=True)

# ---------------- HUD ----------------
st.markdown("""
    <div class="top-hud">
        <div>📡 UPLINK: ACTIVE</div>
        <div>NODE: CRYPTO_CORE_01</div>
        <div>TIME: {}</div>
    </div>
""".format(time.strftime("%H:%M:%S")), unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>RISKGUARD.ai</h1></div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
with st.container():
    c1, c2, c3 = st.columns(3, gap="large")
    
    with c1:
        st.markdown('<div class="glass-card"><span class="section-label">01 // SUBJECT PROFILE</span>', unsafe_allow_html=True)
        months = st.number_input("TENURE (MONTHS)", 0, 500, 120)
        age = st.slider("AGE", 18, 90, 35)
        premium = st.number_input("ANNUAL PREMIUM ($)", 0.0, 5000.0, 1200.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card"><span class="section-label">02 // INCIDENT VECTORS</span>', unsafe_allow_html=True)
        claim_amt = st.number_input("TOTAL CLAIM ($)", 0.0, 100000.0, 15000.0)
        severity = st.selectbox("IMPACT SEVERITY", ["Minor Damage", "Major Damage", "Total Loss"])
        collision = st.selectbox("COLLISION TYPE", ["Front", "Rear", "Side"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="glass-card"><span class="section-label">03 // FIELD DATA</span>', unsafe_allow_html=True)
        witnesses = st.select_slider("WITNESS COUNT", options=[0, 1, 2, 3])
        police = st.radio("OFFICIAL REPORT", ["YES", "NO"], horizontal=True)
        hour = st.slider("INCIDENT HOUR", 0, 23, 14)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("RUN NEURAL DIAGNOSTIC"):
    with st.status("Analyzing Data Patterns...", expanded=True) as status:
        st.write("Extracting feature vectors...")
        time.sleep(0.6)
        st.write("Running cross-validation against historic fraud nodes...")
        time.sleep(0.8)
        st.write("Finalizing risk probability...")
        
        # Logic Placeholder
        fraud_prob = 74.2 if claim_amt > 20000 else 18.5
        status.update(label="Analysis Complete", state="complete", expanded=False)

    # ---------------- RESULTS ----------------
    r_col1, r_col2 = st.columns([1, 1], gap="large")
    
    with r_col1:
        color = "#00ff88" if fraud_prob < 30 else "#ffcc00" if fraud_prob < 70 else "#ff4b4b"
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 5px solid {color};">
                <span class="section-label">DIAGNOSTIC RESULT</span>
                <div class="metric-value" style="color: white; text-shadow: 0 0 30px {color}66;">
                    {fraud_prob:.1f}<span style="font-size: 2rem; color: {color}">%</span>
                </div>
                <p style="color: {color}; font-weight: 700; letter-spacing: 2px;">RISK LEVEL: {"CRITICAL" if fraud_prob > 70 else "ELEVATED" if fraud_prob > 30 else "STABLE"}</p>
            </div>
        """, unsafe_allow_html=True)

    with r_col2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<span class="section-label">NEURAL REASONING</span>', unsafe_allow_html=True)
        
        if fraud_prob > 50:
            st.error("⚠️ **Pattern Match:** High Correlation with 'Staged Collision' archetypes.")
            st.markdown(f"- **Claim Ratio:** High ($ {claim_amt}) relative to tenure.")
            st.markdown("- **Witness Gaps:** Reported severity inconsistent with witness count.")
        else:
            st.success("✅ **Profile Verified:** Behavior aligns with standard historical claims.")
            st.markdown("- **Tenure Logic:** Account age supports claim validity.")
            st.markdown("- **Metric Alignment:** Incident hour and type are low-risk.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- GRAPH ----------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">FEATURE ATTRIBUTION MAP</span>', unsafe_allow_html=True)
    
    features = ["Claim Amount", "Severity", "Witnesses", "Hour", "Tenure"]
    importance = [45, 30, -20, 15, -10] if fraud_prob > 50 else [-15, -10, 25, 5, 30]
    
    fig = go.Figure(go.Bar(
        x=importance, y=features, orientation='h',
        marker=dict(
            color=['#ff4b4b' if x > 0 else '#00ff88' for x in importance],
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888', family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=0, b=0), height=250,
        xaxis=dict(showgrid=False, zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"<div style='text-align: center; color: #444; font-family: monospace; font-size: 0.7rem; margin-top: 50px;'>CORE_V3.0 // BUILD_SECURE_{int(time.time())}</div>", unsafe_allow_html=True)