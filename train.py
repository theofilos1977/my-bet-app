import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

def train_final_model():
    print("Ανάγνωση matches.xlsx...")
    df = pd.read_excel('matches.xlsx', header=None)

    def c_odds(val):
        try:
            if isinstance(val, str): val = val.replace(',', '.').strip()
            return float(val) if val not in ['-', ''] else np.nan
        except: return np.nan

    # Προετοιμασία Features
    X_raw = pd.DataFrame()
    X_raw['O1'] = df[5].apply(c_odds)
    X_raw['OX'] = df[7].apply(c_odds)
    X_raw['O2'] = df[9].apply(c_odds)
    
    # Προετοιμασία Targets
    targets_df = pd.DataFrame()
    targets_df['res'] = df[10].astype(str)
    targets_df['u15'] = df[15].apply(lambda x: 1 if str(x).strip() == 'u' else 0)
    targets_df['o15'] = df[15].apply(lambda x: 1 if str(x).strip() == 'O' else 0)
    targets_df['u25'] = df[16].apply(lambda x: 1 if str(x).strip() == 'u' else 0)
    targets_df['o25'] = df[16].apply(lambda x: 1 if str(x).strip() == 'O' else 0)
    targets_df['u35'] = df[17].apply(lambda x: 1 if str(x).strip() == 'u' else 0)
    targets_df['o35'] = df[17].apply(lambda x: 1 if str(x).strip() == 'O' else 0)
    targets_df['gg']  = df[18].apply(lambda x: 1 if str(x).strip() == 'G' else 0)
    targets_df['ng']  = df[18].apply(lambda x: 1 if str(x).strip() == 'n' else 0)

    # Ένωση και καθαρισμός κενών
    full_df = pd.concat([X_raw, targets_df], axis=1).dropna()
    X = full_df[['O1', 'OX', 'O2']]

    # Εκπαίδευση
    target_names = ['res', 'u15', 'o15', 'u25', 'o25', 'u35', 'o35', 'gg', 'ng']
    for t in target_names:
        # max_depth=10 για να είναι μικρά τα αρχεία (περίπου 1-2MB το καθένα)
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X, full_df[t])
        joblib.dump(model, f'model_{t}.pkl', compress=3)
        print(f"✅ Δημιουργήθηκε το model_{t}.pkl")

if __name__ == "__main__":
    train_final_model()