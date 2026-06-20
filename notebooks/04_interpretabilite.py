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


importances_clf = model_clf.feature_importances_
df_imp_clf = pd.DataFrame({
    'feature': feature_names,
    'importance': importances_clf
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(9, 7))
plt.barh(df_imp_clf['feature'][::-1], df_imp_clf['importance'][::-1],
         color='#2ecc71', edgecolor='white')
plt.title('Top 15 features — Importance (Classification Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../data/models/05_feature_importance_clf.png', dpi=150)
plt.show()
print("Feature importance classification sauvegardée ")


y_pred_reg = model_reg.predict(X_test_s)
residus = y_test_r.values - y_pred_reg

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_pred_reg, residus, alpha=0.5, color='steelblue', s=30)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Résidus vs valeurs prédites')
axes[0].set_xlabel('Note prédite')
axes[0].set_ylabel('Résidu (réel - prédit)')

axes[1].hist(residus, bins=30, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Distribution des résidus')
axes[1].set_xlabel('Résidu')

plt.tight_layout()
plt.savefig('../data/models/06_analyse_residus.png', dpi=150)
plt.show()
print(f"\nRésidu moyen : {residus.mean():.3f}")
print(f"Écart-type des résidus : {residus.std():.3f}")


df_test = df.iloc[X_test.index].copy()
df_test['note_predite'] = y_pred_reg.round(2)
df_test['erreur'] = (df_test['note_finale'] - df_test['note_predite']).abs()

print("\n=== 5 meilleures prédictions (erreur minimale) ===")
print(df_test.nsmallest(5, 'erreur')[
    ['prenom', 'nom', 'note_finale', 'note_predite', 'erreur']
].to_string(index=False))

print("\n=== 5 pires prédictions (erreur maximale) ===")
print(df_test.nlargest(5, 'erreur')[
    ['prenom', 'nom', 'note_finale', 'note_predite', 'erreur']
].to_string(index=False))



nouvel_etudiant = pd.DataFrame([{
    'age': 20, 'filiere': 'Informatique', 'annee_etude': 'L3',
    'bac_serie': 'Général', 'mention_bac': 3, 'boursier': 0,
    'taux_presence': 88.0, 'nb_absences': 2, 'heures_travail_sem': 15,
    'participation_cours': 4, 'nb_devoirs_rendus': 11, 'note_partiel_mi': 14.5,
    'utilisation_biblio': 3, 'groupe_etude': 1, 'ressources_ligne': 1,
    'job_etudiant': 0, 'heures_job_sem': 0, 'distance_univ': 5.0,
    'transport': 'Vélo/Marche', 'soutien_famille': 4, 'stress_percu': 3,
    'sante_percue': 8, 'sport_regulier': 1, 'espace_travail': 1,
    'genre': 'M'
}])

nouvel_etudiant_enc = pd.get_dummies(nouvel_etudiant, drop_first=True)
nouvel_etudiant_enc = nouvel_etudiant_enc.reindex(columns=feature_names, fill_value=0)
nouvel_etudiant_s = scaler.transform(nouvel_etudiant_enc)

pred_note = model_reg.predict(nouvel_etudiant_s)[0]
pred_mention = le.inverse_transform(model_clf.predict(nouvel_etudiant_s))[0]

print(f"\n=== PRÉDICTION NOUVEL ÉTUDIANT ===")
print(f"Note prédite    : {pred_note:.2f}/20")
print(f"Mention prédite : {pred_mention}")


print("\n" + "="*50)
print("RÉSUMÉ INTERPRÉTABILITÉ")
print("="*50)
print(f"Feature la plus importante (régression)     : {df_imp_reg.iloc[0]['feature']}")
print(f"Feature la plus importante (classification) : {df_imp_clf.iloc[0]['feature']}")
print(f"Résidu moyen : {residus.mean():.3f} (proche de 0 = bon signe)")
print(f"\nGraphiques sauvegardés dans ../data/models/")
print("="*50)