import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="AI Bet Predictor", layout="centered")

st.title("⚽ AI Match Analysis")

@st.cache_resource
def load_models():
    model_names = ['res', 'u15', 'o15', 'u25', 'o25', 'u35', 'o35', 'gg', 'ng']
    loaded_models = {}
    for name in model_names:
        try:
            loaded_models[name] = joblib.load(f'model_{name}.pkl')
        except:
            loaded_models[name] = None
    return loaded_models

models = load_models()

# --- ΕΙΣΟΔΟΣ ---
col1, col2, col3 = st.columns(3)
with col1: o1 = st.number_input("Άσος (1)", value=2.10, format="%.2f")
with col2: ox = st.number_input("Ισοπαλία (X)", value=3.20, format="%.2f")
with col3: o2 = st.number_input("Διπλό (2)", value=3.40, format="%.2f")

if st.button("📊 ΑΝΑΛΥΣΗ ΑΓΩΝΑ"):
    input_data = np.array([[o1, ox, o2]])
    
    # --- ΠΙΘΑΝΟΤΗΤΕΣ 1-X-2 ---
    st.subheader("🎯 Πιθανότητες 1-X-2")
    if models['res']:
        probs = models['res'].predict_proba(input_data)[0]
        labels = models['res'].classes_
        
        # Λεξικό για να μαζέψουμε τα ποσοστά
        final_results = {"1": 0.0, "X": 0.0, "2": 0.0}
        
        for lbl, pr in zip(labels, probs):
            s_lbl = str(lbl).strip().upper()
            if '1' in s_lbl: final_results["1"] += pr
            elif '2' in s_lbl: final_results["2"] += pr
            else: final_results["X"] += pr # Όλα τα υπόλοιπα (X, R, κλπ) πάνε στο Χ
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Άσος (1)", f"{final_results['1']*100:.1f}%")
        c2.metric("Ισοπαλία (X)", f"{final_results['X']*100:.1f}%")
        c3.metric("Διπλό (2)", f"{final_results['2']*100:.1f}%")

    # --- GOALS & GG ---
    st.subheader("📈 Αγορές Goals & GG")
    data = []
    for title, key in [('Under 1.5', 'u15'), ('Over 1.5', 'o15'), ('Under 2.5', 'u25'), ('Over 2.5', 'o25'), ('Under 3.5', 'u35'), ('Over 3.5', 'o35'), ('Goal/Goal', 'gg'), ('No Goal', 'ng')]:
        m = models[key]
        if m and len(m.classes_) > 1:
            # Παίρνουμε την πιθανότητα για την κλάση που σημαίνει "έγινε το γεγονός"
            p = m.predict_proba(input_data)[0][1] * 100
            if p > 0: data.append({"Αγορά": title, "Πιθανότητα": f"{p:.1f}%"})
    
    if data:
        st.table(pd.DataFrame(data))
    else:
        st.warning("Δεν βρέθηκαν δεδομένα για τα Goals.")
