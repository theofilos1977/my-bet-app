# --- 1. ΠΡΟΒΛΕΨΗ 1-X-2 ---
        st.divider()
        st.subheader("🎯 Πιθανότητες Σημείου (1-X-2)")
        prob_1x2 = models['res'].predict_proba(input_data)[0]
        classes_1x2 = models['res'].classes_
        
        cols = st.columns(len(classes_1x2))
        for idx, label in enumerate(classes_1x2):
            val = prob_1x2[idx] * 100
            # Διόρθωση για να φαίνεται σωστά το X αν το μοντέλο το λέει αλλιώς
            display_label = "X" if str(label).strip().lower() in ['x', 'r'] else label
            cols[idx].metric(f"Σημείο {display_label}", f"{val:.1f}%")

        # --- 2. ΠΡΟΒΛΕΨΗ GOALS & GG/NG ---
        st.subheader("📈 Αγορές Goals & GG")
        
        def get_p(m_key):
            p_array = models[m_key].predict_proba(input_data)[0]
            # Αν το μοντέλο έχει δύο κλάσεις (0 και 1), πάρε τη δεύτερη (το 1)
            # Αν έχει μόνο μία, επέστρεψε 0.0% ή 100.0% αναλόγως
            if len(p_array) > 1:
                return f"{p_array[1]*100:.1f}%"
            else:
                return "0.0%" if models[m_key].classes_[0] == 0 else "100.0%"

        results = {
            "Αγορά": ["Under 1.5", "Over 1.5", "Under 2.5", "Over 2.5", "Under 3.5", "Over 3.5", "Goal/Goal", "No Goal"],
            "Πιθανότητα %": [
                get_p('u15'), get_p('o15'),
                get_p('u25'), get_p('o25'),
                get_p('u35'), get_p('o35'),
                get_p('gg'), get_p('ng')
            ]
        }
        st.table(pd.DataFrame(results))
