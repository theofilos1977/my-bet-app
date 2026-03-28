import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Ρυθμίσεις σελίδας
st.set_page_config(page_title="AI Match Predictor", layout="centered")

st.title("⚽ AI Match Analysis")
st.write("Εισάγετε τις αποδόσεις του ΟΠΑΠ.")

# 2. Λίστα μοντέλων
model_names = ['res', 'u15', 'o15', 'u25', 'o25', 'u35', 'o35', 'gg', 'ng']

@st.cache_resource
def load_models():
    loaded_models = {}
    for name in model_names:
        filename = f'model_{name}.pkl'
        loaded_models[name] = joblib.load(filename)
    return loaded_models

try:
    models = load_models()

    # 3. Είσοδος δεδομένων
    st.subheader("1. Εισαγωγή Αποδόσεων")
    col1, col2, col3 = st.columns(3)
    with col1:
        o1 = st.number_input("Άσος (1)", value=2.10, step=0.01, format="%.2f")
    with col2:
        ox = st.number_input("Ισοπαλία (X)", value=3.20, step=0.01, format="%.2f")
    with col3:
        o2 = st.number_input("Διπλό (2)", value=3.40, step=0.01, format="%.2f")

    if st.button("📊 ΥΠΟΛΟΓΙΣΜΟΣ ΠΙΘΑΝΟΤΗΤΩΝ"):
        input_data = np.array([[o1, ox, o2]])

        # 4. Πρόβλεψη 1-X-2
        st.divider()
        st.subheader("🎯 Πιθανότητες Σημείου (1-X-2)")
        prob_1x2 = models['res'].predict_proba(input_data)[0]
        classes_1x2 = models['res'].classes_
        
        cols = st.columns(len(classes_1x2))
        for idx, label in enumerate(classes_1x2):
            val = prob_1x2[idx] * 100
            # Διόρθωση ονόματος X
            lbl = str(label).strip().upper()
            display_label = "X" if lbl in ['X', 'R'] else label
            cols[idx].metric(f"Σημείο {display_label}", f"{val:.1f}%")

        # 5. Πρόβλεψη Goals & GG/NG
        st.subheader("📈 Αγορές Goals & GG")
        
        def get_p(m_key):
            p_array = models[m_key].predict_proba(input_data)[0]
            if len(p_array) > 1:
                return f"{p_array[1]*100:.1f}%"
            else:
                return "0.0%" if models[m_key].classes_[0] == 0 else "100.0%"

        results = {
            "Αγορά": ["Under 1.5", "Over 1.5", "Under 2.5", "Over 2.5", "Under 3.5", "Over 3.5", "Goal/Goal", "No Goal"],
            "Πιθανότητα %": [
                get_p('u15'), get_p('o15'), get_p('u25'), get_p('o25'),
                get_p('u35'), get_p('o35'), get_p('gg'), get_p('ng')
            ]
        }
        st.table(pd.DataFrame(results))

except FileNotFoundError:
    st.error("⚠️ Σφάλμα: Δεν βρέθηκαν τα αρχεία .pkl.")
except Exception as e:
    st.error(f"⚠️ Κάτι πήγε στραβά: {e}")
