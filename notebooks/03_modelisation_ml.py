import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, f1_score, confusion_matrix,
                              classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('../data/models', exist_ok=True)
os.makedirs('../models', exist_ok=True)

df = pd.read_csv('../data/etudiants_notes.csv')
print(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

cols_exclure = ['prenom', 'nom', 'note_finale', 'mention']

df_model = df.drop(columns=cols_exclure)
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model
y_reg = df['note_finale']
le = LabelEncoder()
y_clf = le.fit_transform(df['mention'])

X_train, X_test, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train : {X_train_s.shape[0]} étudiants | Test : {X_test_s.shape[0]} étudiants")
print(f"Nombre de features : {X_train_s.shape[1]}")


modeles_reg = {
    'Régression linéaire':  LinearRegression(),
    'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, random_state=42),
}

resultats_reg = []
for nom, modele in modeles_reg.items():
    modele.fit(X_train_s, y_train_r)
    y_pred = modele.predict(X_test_s)
    resultats_reg.append({
        'Modèle':  nom,
        'R² test': round(r2_score(y_test_r, y_pred), 3),
        'RMSE':    round(np.sqrt(mean_squared_error(y_test_r, y_pred)), 3),
        'MAE':     round(mean_absolute_error(y_test_r, y_pred), 3),
    })

df_reg = pd.DataFrame(resultats_reg)
print("\n=== RÉGRESSION ===")
print(df_reg.to_string(index=False))


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['R² test', 'RMSE', 'MAE']
couleurs = ['#3498db', '#e74c3c', '#f39c12']

for i, (metric, couleur) in enumerate(zip(metrics, couleurs)):
    axes[i].bar(df_reg['Modèle'], df_reg[metric], color=couleur,
                edgecolor='white', alpha=0.85)
    axes[i].set_title(f'Comparaison — {metric}')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=15)
    for j, val in enumerate(df_reg[metric]):
        axes[i].text(j, val + 0.01, str(val), ha='center', fontsize=9)

plt.suptitle('Comparaison des modèles de régression', fontsize=13)
plt.tight_layout()
plt.savefig('../data/models/01_comparaison_regression.png', dpi=150)
plt.show()
print("Graphique régression sauvegardé ")


modeles_clf = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='rbf', random_state=42, probability=True),
    'KNN':           KNeighborsClassifier(n_neighbors=7),
}

resultats_clf = []
for nom, modele in modeles_clf.items():
    modele.fit(X_train_s, y_train_c)
    y_pred = modele.predict(X_test_s)
    cv = cross_val_score(modele, X_train_s, y_train_c, cv=5, scoring='accuracy')
    resultats_clf.append({
        'Modèle':     nom,
        'Accuracy':   round(accuracy_score(y_test_c, y_pred), 3),
        'F1 (macro)': round(f1_score(y_test_c, y_pred, average='macro'), 3),
        'CV moyen':   round(cv.mean(), 3),
    })

df_clf = pd.DataFrame(resultats_clf)
print("\n=== CLASSIFICATION ===")
print(df_clf.to_string(index=False))


meilleur_clf = modeles_clf['Random Forest']
y_pred_rf = meilleur_clf.predict(X_test_s)

labels = list(le.classes_)
cm = confusion_matrix(y_test_c, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Matrice de confusion — Random Forest (Classification)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig('../data/models/02_matrice_confusion.png', dpi=150)
plt.show()
print("\nRapport de classification :")
print(classification_report(y_test_c, y_pred_rf, target_names=labels))


meilleur_reg = modeles_reg['Random Forest']
y_pred_reg = meilleur_reg.predict(X_test_s)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, y_pred_reg, alpha=0.5, color='steelblue', s=30)
plt.plot([0, 20], [0, 20], 'r--', linewidth=1.5, label='Prédiction parfaite')
plt.title('Valeurs réelles vs prédites — Random Forest (Régression)')
plt.xlabel('Note réelle /20')
plt.ylabel('Note prédite /20')
plt.legend()
plt.tight_layout()
plt.savefig('../data/models/03_reel_vs_predit.png', dpi=150)
plt.show()
print("Graphique réel vs prédit sauvegardé ✓")


joblib.dump(meilleur_reg, '../models/best_model_regression.joblib')
joblib.dump(meilleur_clf, '../models/best_model_classification.joblib')
joblib.dump(scaler, '../models/scaler.joblib')
joblib.dump(le, '../models/label_encoder.joblib')

print("\nModèles sauvegardés dans ../models/")


print("\n" + "="*50)
print("RÉSUMÉ MODÉLISATION")
print("="*50)
best_r2 = df_reg.loc[df_reg['R² test'].idxmax()]
best_acc = df_clf.loc[df_clf['Accuracy'].idxmax()]
print(f"Meilleur modèle régression    : {best_r2['Modèle']} (R²={best_r2['R² test']})")
print(f"Meilleur modèle classification: {best_acc['Modèle']} (Acc={best_acc['Accuracy']})")
print("="*50)