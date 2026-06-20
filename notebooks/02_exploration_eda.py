import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('data/eda', exist_ok=True)

df = pd.read_csv('../data/etudiants_notes.csv')

print(f"Datast chargé : {df.shape[0]} étudiants, {df.shape[1]} colonnes.")
print("\nAperçu des 5 premiers étudiants :")
print(df.head())
print(f"\nInfos génerales :")
print(df.info())
print(f"\nStatistiques descriptives :")
print(df.describe().round(2))


plt.figure(figsize=(10, 5))
sns.histplot(df['note_finale'], bins=30, kde=True, color='steelblue')
plt.axvline(df['note_finale'].mean(), color='red', linestyle='--',
            label=f"Moyenne : {df['note_finale'].mean():.2f}")
plt.axvline(df['note_finale'].median(), color='orange', linestyle='--',
            label=f"Médiane : {df['note_finale'].median():.2f}")
plt.title('Distribution des notes finales')
plt.xlabel('Note /20')
plt.ylabel('Nombre d\'étudiants')
plt.legend()
plt.tight_layout()
plt.savefig('../data/eda/01_distribution_notes.png', dpi=150)
plt.show()
print("Graphique 1 sauvegardé ")


plt.figure(figsize=(8, 5))
ordre = ['Faible', 'Moyen', 'Bon', 'Excellent']
couleurs = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
counts = df['mention'].value_counts().reindex(ordre)

bars = plt.bar(counts.index, counts.values, color=couleurs, edgecolor='white')
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val}\n({val/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

plt.title('Répartition des étudiants par mention')
plt.ylabel('Nombre d\'étudiants')
plt.ylim(0, counts.max() + 60)
plt.tight_layout()
plt.savefig('../data/eda/02_repartition_mentions.png', dpi=150)
plt.show()
print("Graphique 2 sauvegardé ")


vars_numeriques = [
    'note_finale', 'note_partiel_mi', 'taux_presence', 'nb_absences',
    'heures_travail_sem', 'participation_cours', 'nb_devoirs_rendus',
    'stress_percu', 'sante_percue', 'soutien_famille', 'heures_job_sem',
    'mention_bac', 'utilisation_biblio', 'distance_univ'
]

corr = df[vars_numeriques].corr()

plt.figure(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Matrice de corrélations entre variables numériques')
plt.tight_layout()
plt.savefig('../data/eda/03_matrice_correlations.png', dpi=150)
plt.show()
print("Graphique 3 sauvegardé ")


corr_note = df[vars_numeriques].corr()['note_finale'].drop('note_finale')
corr_note_sorted = corr_note.abs().sort_values(ascending=True)

couleurs = ['#e74c3c' if corr_note[v] < 0 else '#2ecc71'
            for v in corr_note_sorted.index]

plt.figure(figsize=(9, 6))
bars = plt.barh(corr_note_sorted.index, corr_note_sorted.values,
                color=couleurs, edgecolor='white')
plt.axvline(0.3, color='gray', linestyle='--', alpha=0.5, label='Seuil 0.3')
plt.title('Corrélation de chaque variable avec la note finale\n(vert = positif, rouge = négatif)')
plt.xlabel('Corrélation absolue')
plt.legend()
plt.tight_layout()
plt.savefig('../data/eda/04_correlations_note.png', dpi=150)
plt.show()
print("Graphique 4 sauvegardé ")

plt.figure(figsize=(10, 5))
moy = df.groupby('filiere')['note_finale'].agg(['mean', 'std']).sort_values('mean')
plt.barh(moy.index, moy['mean'], xerr=moy['std'],
         color='steelblue', edgecolor='white', alpha=0.85, capsize=4)
plt.axvline(df['note_finale'].mean(), color='red', linestyle='--',
            label=f"Moyenne globale : {df['note_finale'].mean():.2f}")
plt.title('Note moyenne par filière (avec écart-type)')
plt.xlabel('Note /20')
plt.legend()
plt.tight_layout()
plt.savefig('../data/eda/05_note_par_filiere.png', dpi=150)
plt.show()
print("Graphique 5 sauvegardé ")


plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x='taux_presence', y='note_finale',
                hue='mention', palette=['#e74c3c','#f39c12','#2ecc71','#3498db'],
                alpha=0.6, s=40)
sns.regplot(data=df, x='taux_presence', y='note_finale',
            scatter=False, color='black', line_kws={'linewidth': 1.5})
plt.title('Taux de présence vs note finale')
plt.xlabel('Taux de présence (%)')
plt.ylabel('Note finale /20')
plt.tight_layout()
plt.savefig('../data/eda/06_presence_vs_note.png', dpi=150)
plt.show()
print("Graphique 6 sauvegardé ")


plt.figure(figsize=(10, 5))
ordre_annee = ['L1', 'L2', 'L3', 'M1', 'M2']
sns.boxplot(data=df, x='annee_etude', y='note_finale',
            order=ordre_annee, palette='Blues')
plt.axhline(df['note_finale'].mean(), color='red', linestyle='--', alpha=0.7)
plt.title('Distribution des notes par année d\'étude')
plt.xlabel('Année')
plt.ylabel('Note /20')
plt.tight_layout()
plt.savefig('../data/eda/07_note_par_annee.png', dpi=150)
plt.show()
print("Graphique 7 sauvegardé ")



plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x='stress_percu', y='note_finale',
                hue=df['job_etudiant'].map({0: 'Sans job', 1: 'Avec job'}),
                palette={'Sans job': '#2ecc71', 'Avec job': '#e74c3c'},
                alpha=0.5, s=40)
plt.title('Stress perçu vs note finale\n(selon présence d\'un job étudiant)')
plt.xlabel('Stress perçu (0–10)')
plt.ylabel('Note finale /20')
plt.tight_layout()
plt.savefig('../data/eda/08_stress_job_note.png', dpi=150)
plt.show()
print("Graphique 8 sauvegardé ")


print("\n" + "="*50)
print("RÉSUMÉ EDA")
print("="*50)
print(f"Note moyenne       : {df['note_finale'].mean():.2f} / 20")
print(f"Écart-type         : {df['note_finale'].std():.2f}")
print(f"Meilleur prédicteur: note_partiel_mi (corr = {df['note_partiel_mi'].corr(df['note_finale']):.2f})")
print(f"2e prédicteur      : taux_presence   (corr = {df['taux_presence'].corr(df['note_finale']):.2f})")
print(f"Facteur négatif    : nb_absences     (corr = {df['nb_absences'].corr(df['note_finale']):.2f})")
print(f"\nTous les graphiques sauvegardés dans data/eda/")
print("="*50)