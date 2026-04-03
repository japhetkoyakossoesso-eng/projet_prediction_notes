import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 1000
print("Librairies chargées et données générées.")



prenoms_m = [
    "Mahamat", "Koulbou", "Henoc", "Salim", "Mohamed", "Ramzi", 
    "Jihad", "Ousmane", "Ahmed", "Malik-Adjhacy", "Nicolas",
    "Koussaila", "Jey", "Ismael", "Abdoul", "Youssouf", "Brandon", 
    "Harold", "Boussad", "Abdoul", "Youssef", "Djidou", "Eric", "Vincent",
    "Kouaho", "Francis", "Jonathan", "Simeon", "Cheikh", "Balla", "Georges",
    "Yassine", "Salim", "Augustin", "Beni", "Stone", "Thomas", "Julfab",
    "Louis", "Vianney", "Georges", "Akoete", "Emmanuel", "Anas",
    "Yowan", "Steph", "Hirche Luca", "Marie Paul", "Belicia", "Cédric"
    "Thomas", "Jey", "Francois", "Hugo", "Mathieu", "Hugo", "Railey",
    "Seyed", "Mohammadreza", "Mbayame", "Omar", "Olafemi","Sabridass", 
    "Hugo", "Francois", "Mathieu","Japhet","Khamal", "Nehuan",
    "Omar", "Hafizou", "Lyes", "Mikael", "Abdou", "Yacine", "Mamadou",  
    "Kouaho", "Francis", "Jonathan", "Simeon", "Cheikh", "Balla", "Georges"
]


prenoms_f = [
    "Sara", "Leilath", "Lea", "Rama", "Mageda", "Esther", "Ayedi",
    "Naela", "Manal", "Awa", "Blanche", "Eva", "Ikram", "Souha", "Catherine",
    "Manon", "Vavo", "Mondesire", "Coomi", "Audrey", "Grace", "Caroline",
    "Delicia", "Francoise", "Kenza", "Rawaa", "Miryam", "Daro", "Dourra",
    "Louise", "Aimee Veronica", "Syrine", "Taisir", "Marie", "Louange", "Elodie",
    "Safiatou", "Clarissia", "Magnificath", "Evadnee", "Tova", "Mouniratou",
    "Paola", "Anny", "Christa", "Belicia", "Hadja", "Bintou", "Khady", "Dania",
    "Lena", "Carla", "Wahiba", "Mariamari", "Jessica", "Abigail", "Sophie",
    "Sandrine", "Mouniratou", "Jessica", "Catherine", "Manon", "Dourra"
]


noms = [
    "Abbes", "Adam", "Adjagbe", "Agbagno", "Aguigah", "Ahmat", "Ait mahammed",
    "Alabdullah alomar", "Alasabi", "Ali", "Andriamampianina", "Attias", "Adjo"
    "Awada", "Ayachi", "Ayedi", "Azerguerras", "Badaoui", "Bamba", "Baouya",
    "Barry", "Belatar", "Ben jeddou", "Bensmaine", "Bergoin", "Bertolini",
    "Billel koussaila", "Boudine-Mervillon", "Cabrera belmar", "Camara", "Chan",
    "Chan", "Chanwin", "Cherigui", "Cihyoka", "Coly", "Comparot", "Conde",
    "Crespo", "Dagou", "Dahouede", "Dede", "Dengue matsogni", "Dione", "Diop desbonnet",
    "Djema", "Djokpo", "Dupont", "Dusabe", "Dushime", "Ekoga bideng", "El abi",
    "El alaoui es sousy", "Elcadhi", "El malki", "Essabe mbome oye", "Et touil mendez",
    "Fall", "Fall", "Faviere--Prado", "Fontbonne", "Gakosso", "Ganda-Te-Grembombo",
    "Garcia", "Gaye", "Gbizie", "Ghedhoui", "Giraud", "Goursolle", "Grosdoy",
    "Gueye", "Hachem majdalani", "Haddad", "Hounhuedo", "Houssam", "Iamundo",
    "Intwari", "Irakoze", "Jday", "Jday", "Jida", "Kaba", "Kamrach", "Kape amani",
    "Keita", "Khalil", "Khelouat", "Kouleta", "Koyakosso esso", "Krim", "Kuzmenko",
    "Laclau", "Lakhal", "Lehie-Bi", "Le page", "Loukou Brou", "Lourenco",
    "Makila", "Manantsoa", "Martinez", "Martins", "Mbengue", "Megbenou", "M'jahad",
    "Moenza", "Monasse", "Montanary", "Montean", "Motondo masengi", "Moudjalou mombo",
    "Mouissi amoussou", "Mousali", "Muller", "Nait mohamed", "Obambi mbongo ngala",
    "Osmas", "Ouattara", "Ovono etoubembe", "Paire", "Paucarima", "Pham Huu",
    "Piquer", "Ponnou", "Prevost-Debaisieux", "Rakotonjanahary", "Redombo-Agnorogoule",
    "Richir", "Roumagnac", "Sagnon", "Sahadia", "Sakkaki", "Sall", "Sangwe",
    "Schwartz", "Senin", "Senouci", "Sossoumihen", "Sy", "Tall", "Tchassama",
    "Tchibozo houessou", "Terkmani", "Tobada", "Toubol", "Tourret", "Traye Balezin",
    "Ulukuz", "Ursini", "Wu", "Yassir", "Zebriquane", "Zouaoui"
]


genre   = np.random.choice(['M', 'F'], n, p=[0.48, 0.52])
prenom  = [np.random.choice(prenoms_m) if g == 'M'
           else np.random.choice(prenoms_f) for g in genre]
nom     = np.random.choice(noms, n)

print(f"Exemple : {prenom[0]} {nom[0]} ({genre[0]})")



filiere = np.random.choice(
    ['Informatique', 'Mathematiques', 'Gestion', 'Lettres', 'Sciences', 'Psychologie', 'Droit'],
    n, p=[0.22, 0.18, 0.16, 0.12, 0.18, 0.16, 0.14]
)

annee_etude = np.random.choice(
    ['L1', 'L2', 'L3', 'M1', 'M2'], n,
    p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

bac_serie = np.random.choice(
    ['Général', 'Technologique', 'Professionnel'], n,
    p=[0.65, 0.25, 0.10]
)

# Mention bac : 0=passable, 1=AB, 2=Bien, 3=TB, 4=Félicitations
mention_bac = np.random.choice(
    [0, 1, 2, 3, 4], n,
    p=[0.12, 0.28, 0.33, 0.18, 0.09]
)

boursier = np.random.choice([0, 1], n, p=[0.58, 0.42])

age_base = {'L1': 19, 'L2': 20, 'L3': 21, 'M1': 22, 'M2': 23}
age = np.array([age_base[a] + np.random.randint(0, 3) for a in annee_etude])

print("Profil académique généré pour les étudiants.")




taux_presence = np.clip(np.random.normal(74, 16, n), 10, 100).round(1)

nb_absences = np.round(np.clip(
    (100 - taux_presence) / 10 + np.random.poisson(1, n), 0, 25
)).astype(int)

heures_travail_sem = np.clip(
    np.random.normal(11, 5, n), 0, 35
).round(0).astype(int)

participation_cours = np.random.choice(
    [0, 1, 2, 3, 4, 5], n,
    p=[0.08, 0.15, 0.27, 0.28, 0.15, 0.07]
)

nb_devoirs_rendus = np.clip(
    np.random.normal(9, 2.5, n), 0, 12
).round(0).astype(int)


utilisation_biblio = np.random.choice(
    [0, 1, 2, 3, 4, 5], n,
    p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]
)

note_partiel_mi = np.clip(
    np.random.normal(10.8, 3.8, n), 0, 20
).round(2)

groupe_etude = np.random.choice([0, 1], n, p=[0.52, 0.48])
ressources_ligne = np.random.choice([0, 1], n, p=[0.35, 0.65])

print("Comportement académique généré pour les étudiants.")


job_etudiant = np.random.choice([0, 1], n, p=[0.52, 0.48])
heures_job_sem = np.where(
    job_etudiant == 1,
    np.random.choice([8, 12, 16, 20, 25, 30, 20], n),
    0
)

distance_univ = np.clip(
    np.random.exponential(18, n), 0.5, 120
).round(1)

transport = np.where(
    distance_univ < 5,  'Vélo/Marche',
    np.where(distance_univ < 20, 'Transport en commun',
    np.where(distance_univ < 50, 'Voiture', 'Logement sur place'))
)

soutien_famille = np.random.choice(
    [0, 1, 2, 3, 4, 5], n,
    p=[0.05, 0.10, 0.20, 0.30, 0.25, 0.10]
)

stress_percu = np.clip(
    np.random.normal(5.5, 2.2, n), 0, 10
).round(0).astype(int)

sante_percue = np.clip(
    np.random.normal(6.8, 1.8, n), 0, 10
).round(0).astype(int)

sport_regulier = np.random.choice([0, 1], n, p=[0.45, 0.55])
espace_travail = np.random.choice([0, 1], n, p=[0.20, 0.80])

print("Contexte de vie généré pour les étudiants.")


score = (
    # Ce qui aide vraiment
    note_partiel_mi      * 0.60 +   
    mention_bac          * 1.10 +  
    taux_presence        * 0.06 +   
    heures_travail_sem   * 0.20 +   
    nb_devoirs_rendus    * 0.22 +   
    participation_cours  * 0.45 +   
    soutien_famille      * 0.20 +   
    groupe_etude         * 0.50 +   
    ressources_ligne     * 0.30 +   
    sport_regulier       * 0.25 +   
    sante_percue         * 0.12 +
    espace_travail       * 0.30 +

    nb_absences          * -0.28 +
    stress_percu         * -0.20 +
    heures_job_sem       * -0.04 +

    np.random.normal(0, 1.5, n)
)

s_min, s_max = score.min(), score.max()
note_finale = ((score - s_min) / (s_max - s_min) * 20).round(2)

print(f"Note moyenne : {note_finale.mean():.2f}/20")
print(f"Écart-type   : {note_finale.std():.2f}")
print(f"Min / Max    : {note_finale.min():.2f} / {note_finale.max():.2f}")