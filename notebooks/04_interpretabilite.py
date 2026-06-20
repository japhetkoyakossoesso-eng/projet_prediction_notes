import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('../data/models', exist_ok=True)

# Chargement des données et modèles sauvegardés
df = pd.read_csv('../data/etudiants_notes.csv')
model_reg = joblib.load('../models/best_model_regression.joblib')
model_clf = joblib.load('../models/best_model_classification.joblib')
scaler = joblib.load('../models/scaler.joblib')
le = joblib.load('../models/label_encoder.joblib')

print("Données et modèles chargés ")

from sklearn.model_selection import train_test_split

cols_exclure = ['prenom', 'nom', 'note_finale', 'mention']
df_model = df.drop(columns=cols_exclure)
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model
y_reg = df['note_finale']
y_clf = le.transform(df['mention'])

X_train, X_test, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

feature_names = X.columns.tolist()
print(f"Features disponibles : {len(feature_names)}")


importances_reg = model_reg.feature_importances_
df_imp_reg = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_reg
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(9, 7))
plt.barh(df_imp_reg['feature'][::-1], df_imp_reg['importance'][::-1],
         color='steelblue', edgecolor='white')
plt.title('Top 15 features — Importance (Régression Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../data/models/04_feature_importance_reg.png', dpi=150)
plt.show()
print("Feature importance régression sauvegardée ")
print(df_imp_reg.to_string(index=False))