import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Ρυθμίσεις σελίδας για να φαίνεται ωραία στο κινητό
st.set_page_config(page_title="AI Match Predictor", layout="centered")

st.title("? AI Match Analysis")
st.write("Εισάγετε τις αποδόσεις του ΟΠΑΠ για πρόβλεψη 11 σημείων.")

# Λίστα με τα μοντέλα που πρέπει να φορτωθούν
model_names = ['res', 'u15', 'o15', 'u25', 'o25', 'u35', 'o35', 'gg', 'ng']

# Συνάρτηση για φόρτωση των μοντέλων (με cache για ταχύτητα)
@st.cache_resource
def load_models():
    loaded_models = {}
    for name in model_names:
        filename = f'model_{name}.pkl'
        loaded_models[name] = joblib.load(filename)
    return loaded_models

try:
    models = load_models()

    # --- ΕΙΣΟΔΟΣ ΔΕΔΟΜΕΝΩΝ ---
    st.subheader("1. Εισαγωγή Αποδόσεων")
    col1, col2, col3 = st.columns(3)
    with col1:
        o1 = st.number_input("Άσος (1)", value=2.10, step=0.01, format="%.2f")
    with col2:
        ox = st.number_input("Ισοπαλία (X)", value=3.20, step=0.01, format="%.2f")
    with col3:
        o2 = st.number_input("Διπλό (2)", value=3.40, step=0.01, format="%.2f")

    if st.button("?? ΥΠΟΛΟΓΙΣΜΟΣ ΠΙΘΑΝΟΤΗΤΩΝ"):
        # Προετοιμασία των δεδομένων για τα μοντέλα
        input_data = np.array([[o1, ox, o2]])

        # --- 1. ΠΡΟΒΛΕΨΗ 1-X-2 ---
        st.divider()
        st.subheader("?? Πιθανότητες Σημείου (1-X-2)")
        prob_1x2 = models['res'].predict_proba(input_data)[0]
        classes_1x2 = models['res'].classes_
        
        # Εμφάνιση 1-X-2 σε στήλες
        c1, c2, c3 = st.columns(3)
        # Προσοχή: Η σειρά εξαρτάται από το πώς τα αποθήκευσε το μοντέλο (συνήθως '1', '2', 'X' αλφαβητικά)
        for idx, label in enumerate(classes_1x2):
            val = prob_1x2[idx] * 100
            if idx == 0: c1.metric(f"Σημείο {label}", f"{val:.1f}%")
            if idx == 1: c2.metric(f"Σημείο {label}", f"{val:.1f}%")
            if idx == 2: c3.metric(f"Σημείο {label}", f"{val:.1f}%")

        # --- 2. ΠΡΟΒΛΕΨΗ GOALS & GG/NG ---
        st.subheader("?? Αγορές Goals & GG")
        
        def get_p(m_key):
            # Παίρνουμε την πιθανότητα για την κλάση 1 (δηλαδή να συμβεί το γεγονός)
            return models[m_key].predict_proba(input_data)[0][1] * 100

        results = {
            "Αγορά": ["Under 1.5", "Over 1.5", "Under 2.5", "Over 2.5", "Under 3.5", "Over 3.5", "Goal/Goal", "No Goal"],
            "Πιθανότητα %": [
                f"{get_p('u15'):.1f}%", f"{get_p('o15'):.1f}%",
                f"{get_p('u25'):.1f}%", f"{get_p('o25'):.1f}%",
                f"{get_p('u35'):.1f}%", f"{get_p('o35'):.1f}%",
                f"{get_p('gg'):.1f}%", f"{get_p('ng'):.1f}%"
            ]
        }
        st.table(pd.DataFrame(results))

except FileNotFoundError:
    st.error("?? Σφάλμα: Δεν βρέθηκαν τα αρχεία .pkl. Βεβαιωθείτε ότι τα έχετε ανεβάσει στο GitHub.")
except Exception as e:
    st.error(f"?? Κάτι πήγε στραβά: {e}")