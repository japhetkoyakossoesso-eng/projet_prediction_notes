# 🎓 Prédiction des notes d'étudiants — Machine Learning

Projet de Machine Learning complet, de la génération des données jusqu'au rapport d'analyse,
visant à prédire la réussite académique d'étudiants à partir de variables comportementales,
académiques et socio-environnementales.

---

## 📌 Objectif du projet

Ce projet répond à une question concrète : **peut-on anticiper la note finale d'un étudiant
avant la fin du semestre, à partir d'indicateurs mesurables ?**

C'est un cas d'usage classique de l'*Educational Data Mining* (ou *Learning Analytics*),
un domaine qui aide les établissements à repérer tôt les étudiants à risque pour leur
proposer un accompagnement adapté.

Le projet couvre l'intégralité d'un cycle de Data Science :
génération des données → exploration → modélisation → interprétabilité → restitution.

---

## Outils et technologies utilisés

### Python — génération des données et Machine Learning
| Librairie | Usage |
|---|---|
| `numpy` | Génération de données synthétiques avec distributions statistiques réalistes |
| `pandas` | Manipulation et structuration du dataset |
| `matplotlib` / `seaborn` | Visualisations exploratoires (histogrammes, heatmaps, scatterplots) |
| `scikit-learn` | Modèles ML (régression, classification), preprocessing, évaluation |
| `joblib` | Sauvegarde et rechargement des modèles entraînés |

### R / RStudio — visualisation et rapport
| Package | Usage |
|---|---|
| `tidyverse` (`ggplot2`, `dplyr`) | Visualisations avancées et manipulation de données |
| `corrplot` | Matrices de corrélation visuelles |
| `viridis` | Palettes de couleurs accessibles |
| `RMarkdown` / `knitr` | Génération automatique d'un rapport PDF (texte + code + graphiques) |

### Environnement de développement
- **VSCode** — écriture et exécution du code Python
- **RStudio** — écriture et compilation (*Knit*) du rapport R
- **Terminal (zsh)** — gestion des fichiers et exécution des scripts

---

## 📂 Structure du projet

```
projet_prediction_notes/
├── data/
│   ├── etudiants_notes.csv        # Dataset généré (1000 étudiants)
│   ├── eda/                       # Graphiques d'exploration (8 visuels)
│   └── models/                    # Graphiques de modélisation (6 visuels)
│
├── notebooks/
│   ├── 01_generation_dataset.py   # Création du dataset synthétique
│   ├── 02_exploration_eda.py      # Analyse exploratoire (EDA)
│   ├── 03_modelisation_ml.py      # Entraînement et comparaison des modèles
│   └── 04_interpretabilite.py     # Feature importance, résidus, prédiction
│
├── models/
│   ├── best_model_regression.joblib
│   ├── best_model_classification.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
│
├── rapport/
│   └── rapport_analyse.Rmd        # Rapport R (visualisations + conclusions)
│
└── requirements.txt
```

---

## Comment les données ont été créées et gérées

Aucune université ne partage de vraies données d'étudiants pour des raisons évidentes
de confidentialité (RGPD). J'ai donc **généré moi-même un dataset synthétique de 1000
étudiantsd'autres noms sont tirés aléatoirement avec NumPy** — mais pas de façon purement aléatoire : les variables ont été construites avec des **corrélations réalistes** entre elles.

### Les 3 familles de variables

1. **Profil étudiant** : filière, année d'étude, mention au bac, statut boursier...
2. **Comportement académique** : taux de présence, note au partiel mi-semestre,
   devoirs rendus, participation en cours...
3. **Contexte de vie** : job étudiant, stress perçu, soutien familial, distance à
   l'université...

### Gestion des noms et prénoms

Des **listes de prénoms et noms français courants** ont été utilisées, tirées
aléatoirement et indépendamment du genre déclaré, simplement pour **humaniser le
dataset** et le rendre plus lisible.

⚠️ Ces colonnes (`prenom`, `nom`) sont **explicitement exclues du pipeline de
modélisation** : un nom n'a aucune valeur prédictive, et l'inclure introduirait
un biais inutile dans les modèles.

### Calcul de la variable cible (note finale)

La note finale a été calculée par une **formule pondérée** : chaque variable
contribue positivement ou négativement selon un poids réaliste (le partiel
mi-semestre pèse le plus, le stress et les absences pénalisent), à laquelle
s'ajoute un **bruit gaussien** pour simuler l'aléa inévitable de la vraie vie —
un étudiant peut avoir un mauvais jour, peu importe sa préparation.

Une variable `mention` (Faible / Moyen / Bon / Excellent) a ensuite été dérivée
de la note finale, pour permettre à la fois des tâches de **régression** et de
**classification** sur le même dataset.

---

## Méthodologie en 4 étapes

| Étape | Fichier | Description |
|---|---|---|
| **1. Génération** | `01_generation_dataset.py` | Création des 1000 étudiants fictifs avec variables corrélées → export CSV |
| **2. Exploration (EDA)** | `02_exploration_eda.py` | Distributions, corrélations, comparaisons par filière/année |
| **3. Modélisation** | `03_modelisation_ml.py` | Entraînement et comparaison de plusieurs modèles ML |
| **4. Interprétabilité** | `04_interpretabilite.py` | Feature importance, analyse des erreurs, prédiction sur un cas fictif |

### Modèles testés

**Régression** (prédiction de la note exacte /20) :
- Régression linéaire
- Random Forest Regressor
- Gradient Boosting Regressor

**Classification** (prédiction de la mention) :
- Random Forest Classifier
- SVM
- KNN

Chaque modèle a été évalué sur un split **train/test 80/20**, avec validation
croisée pour fiabiliser les résultats (R², RMSE, MAE pour la régression ;
Accuracy, F1-score et matrice de confusion pour la classification).

---

## 📊 Résultats clés

- La **note au partiel mi-semestre** est le meilleur prédicteur de la note finale.
- Le **taux de présence** est positivement corrélé à la performance.
- Le **stress perçu** et un **job étudiant intensif** ont un effet négatif modéré.
- Le **soutien familial** joue un rôle protecteur sur la réussite.
- Meilleur modèle de régression : **Régression linéaire** (R² ≈ 0.84)

---

##  Installation et exécution

### 1. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

### 2. Lancer le pipeline dans l'ordre
```bash
cd notebooks
python3 01_generation_dataset.py
python3 02_exploration_eda.py
python3 03_modelisation_ml.py
python3 04_interpretabilite.py
```

### 3. Générer le rapport R
Ouvrir `rapport/rapport_analyse.Rmd` dans RStudio, puis cliquer sur
**Knit → Knit to PDF**.

Packages R requis :
```r
install.packages(c("tidyverse", "corrplot", "viridis"))
```

---

## 👤 Auteur

**Japhet Koyakosso-Esso**
Étudiant en L2 MIASHS — Informatique et SHS, Université Toulouse Jean Jaurès
