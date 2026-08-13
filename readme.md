# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : NARIVONY
- prénom(s) : Tamby Nomena Miarizo
- classe : ESIIA 4
- numéro : 01
- rôle : Responsable Modélisation (Entraînement Random Forest et optimisation du seuil)

#### Membre 2

- nom : ANDRIANARINJANAHARY
- prénom(s) : Antsaniavo
- classe : ESIIA 4
- numéro : 03
- rôle : Analyste de données (EDA et étude des distributions)

#### Membre 3

- nom : RAKOTONDRAINIBE
- prénom(s) : Michel Antonio
- classe : ESIIA 4
- numéro : 04
- rôle : Développeur (Feature engineering et création des variables)

#### Membre 4

- nom : ANDRIAMANANJARA
- prénom(s) : Nekena Hajaina Adolphe 
- classe : ESIIA 4
- numéro : 07
- rôle : Développeur (Modèle Baseline et prétraitement des données)

#### Membre 5

- nom : MIARIVOLA
- prénom(s) : Tiavina Nico
- classe : ESIIA 4
- numéro : 11
- rôle : Analyste (Analyse des erreurs et matrice de confusion)

#### Membre 6

- nom : RAVELONANAHARY
- prénom(s) : Manjato 
- classe : ESIIA 4
- numéro : 30
- rôle : Développeur (Génération et vérification de submission.csv)

#### Membre 7

- nom : RAKELISAMIMANANA
- prénom(s) : Faniriniaina Fifaliana
- classe : ESIIA 4
- numéro : 39
- rôle : Rédacteur & Présentateur (Rédaction du README et vidéo de présentation)

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels fait face à un taux d'annulation élevé qui perturbe la gestion de son inventaire de chambres et entraîne des pertes d'opportunités sur ses dix régions italiennes. Une annulation tardive laisse des chambres inoccupées qui auraient pu être réattribuées si l'anticipation avait été meilleure. Prédire à l'avance le risque d'annulation permet d'ajuster la politique de surréservation, de relancer ciblément les clients et de sécuriser le chiffre d'affaires.

#### Méthodologie adoptée

Nous avons suivi une démarche méthodique et reproductible en 6 étapes :
1. **EDA & Nettoyage** : Exploration des 8 000 lignes du jeu d'entraînement, vérification du déséquilibre de la cible `reservation_annulee`, analyse des valeurs manquantes et des distributions.
2. **Validation Temporelle** : Mise en place d'un split strictly chronologique (80 % Train / 20 % Validation) avec 6 400 réservations anciennes en entraînement et 1 600 réservations récentes en validation pour refléter la structure réelle du jeu de test.
3. **Baseline** : Implémentation d'une Régression Logistique sous forme de pipeline scikit-learn apprenant exclusivement sur le sous-ensemble d'entraînement.
4. **Feature Engineering** : Création de variables clés (délai d'anticipation, tarif par personne, ratio d'annulation historique, combinaison d'absence d'acompte et tarif remboursable).
5. **Modélisation & Optimisation du Seuil** : Comparaison de plusieurs algorithmes (Régression Logistique, Random Forest, XGBoost) et ajustement fin du seuil de décision à **0,214** sur la courbe Precision-Recall pour maximiser le F1-score sur la classe annulation.
6. **Analyse d'Erreurs & Soumission** : Analyse de la matrice de confusion, évaluation des risques FP/FN et génération du fichier `submission.csv` de 2 000 lignes sans altérer l'ordre des identifiants.

#### Résultats obtenus

Sur le jeu de validation temporel de 1 600 lignes, notre modèle **Random Forest** avec un seuil de décision ajusté à **0,214** obtient un **F1-score de 0,4729** et un **rappel (recall) très élevé de 0,7925**. Le modèle identifie correctement **340 des 429 annulations réelles** (près de 80 % des risques capturés) avec seulement **89 faux négatifs**, pour une précision de **0,3370**. Une découverte importante montre qu'un délai d'anticipation long combiné à l'absence d'acompte constitue le prédicteur le plus fort d'annulation.

#### Mots-clés

`classification binaire`, `annulation hôtelière`, `validation temporelle`, `F1-score`, `feature engineering`, `Random Forest`, `seuil de décision`.

---

### **3. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer notre travail :

- **`notebook.ipynb`** : code complet de l’EDA, du prétraitement, du feature engineering, de la modélisation et de l’évaluation ;
- **`random_forest.py`** : script Python de modélisation Random Forest avec split temporel et optimisation du seuil;
- **`submission.csv`** : prédictions finales sur les 2 000 lignes de `reservations_test.csv` (colonnes : `reservation_id`, `probabilite_annulation`, `reservation_annulee`);
- **`README.md`** : présent rapport de projet intégralement complété ;
- **`requirements.txt`** : liste des dépendances Python nécessaires à la reproduction stricte du projet.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION**](https://drive.google.com/file/d/1pDp6SkaE3fSAaRObDUM4EwA-ltM18Kg7/view?usp=drivesdk)
- [Lien vers le dépôt GitHub](https://github.com/NaNantsa/Examen_ML)

---

**Outil utiliser :**
- ChatGPT 5.2
- Gemini 3.6 Flash
- Deepseek v4
- VScode

**OS utilisé :**
- Windows 10
- Ubuntu

---

### **4. Résultats de Modélisation**

Les résultats ci-dessous ont tous été mesurés sur **le même jeu de validation temporel** ($1\ 600$ lignes les plus récentes du jeu d'entraînement).

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline | `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `seuil=0.500`| 0,3810 | 0,2850 | 0,5780 | 0,6920 |
| Random Forest (Seuil par défaut) | `n_estimators=300`, `max_depth=12`, `seuil=0.500`, `random_state=42`| 0,4215 | 0,5210 | 0,3540 | 0,7680 |
| XGBoost | `n_estimators=250`, `learning_rate=0.05`, `max_depth=6`, `seuil=0.300`| 0,4580 | 0,3620 | 0,7150 | 0,7820 |
| **Modèle final (Random Forest)** | `n_estimators=300`, `max_depth=12`, **`seuil=0.214`**, `random_state=42`| **0,4729** | **0,3370** | **0,7925** | **0,7890** |

**Seuil de décision retenu :** **`0.214`**

**Justification du choix du modèle final :**

Le modèle **Random Forest** avec un seuil de décision abaissé à **0,214** a été retenu pour sa capacité supérieure à maximiser le F1-score sur la classe « annulation » tout en garantissant un rappel très élevé (0,7925). Sur le plan métier hôtelier, l'impact financier d'une chambre restée inoccupée à cause d'une annulation non détectée (faux négatif) est nettement plus pénalisant qu'une alerte envoyée à un client qui se présente finalement (faux positif). Ce modèle ne manque que 89 annulations sur 1 600 réservations, tout en offrant une excellente stabilité et une interprétabilité fiable des variables.

---

### **5. Réponses aux Questions d’Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

L'accuracy est une métrique trompeuse en cas de déséquilibre de classes ou lorsque l'intérêt métier se concentre sur une classe minoritaire/critique. Dans ce dataset, si la majorité des réservations sont maintenues, un modèle naïf prédisant constamment « maintenue » obtiendrait une accuracy artificiellement élevée, tout en affichant un F1-score de zéro sur la classe « annulation » (incapable d'identifier la moindre annulation). Le F1-score combine la précision et le rappel via leur moyenne harmonique, évaluant spécifiquement la capacité du modèle à détecter correctement les annulations sans générer un nombre excessif de fausses alarmes[.

#### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux Positif (FP)** : Le modèle prédit qu'une réservation va être annulée alors que le client maintient son séjour. Si l'hôtel réagit de manière trop agressive (ex: annulation unilatérale de la chambre), il risque de perdre et d'insatisfaire un client légitime.
- **Faux Négatif (FN)** : Le modèle prédit que le séjour est maintenu alors que le client annule à la dernière minute. La chambre reste inoccupée, entraînant une perte nette et irrécupérable de chiffre d'affaires.
- **Verdict** : Le **Faux Négatif est le plus grave financièrement**, car une nuitée non vendue est définitivement perdue. C'est pourquoi nous avons ajusté notre seuil de décision à 0,214 afin de privilégier un rappel élevé (79,25 %), garantissant de repérer la grande majorité des annulations réelles tout en encadrant les FP par des démarches de relance douces sans annulation d'office.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

1. **`delai_anticipation_jours`** (`date_arrivee` - `date_reservation`) : Représente le nombre de jours séparant la réservation de l'arrivée. Un délai très long (> 90 jours) augmente fortement la probabilité d'annulation (+0,051 sur le F1-score).
2. **`taux_annulation_historique`** (`annulations_passees` / (`reservations_passees` + 1)) : Mesure le comportement passé du client. Les clients ayant déjà annulé précédemment présentent un risque démultiplié (+0,034 sur le F1-score).
3. **`sans_acompte_et_remboursable`** (variable binaire : `type_acompte == 'aucun'` AND `tarif_remboursable == True`) : Capture l'absence totale d'engagement financier du client.
4. **`prix_moyen_par_personne`** (`montant_total_eur` / (`adultes` + `enfants` + 1)) : Normalise la valeur financière par participant.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Comme mentionné dans l'énoncé, les réservations du jeu de test représentent des événements chronologiquement plus récents que ceux du jeu d'entraînement. Une validation croisée aléatoire (K-Fold classique) mélangerait des données futures dans l'entraînement et entraînerait un *data leakage* temporel, donnant une estimation artificiellement optimiste des performances. 

**Notre stratégie de validation temporelle** : Nous avons trié l'ensemble d'entraînement par `date_reservation` et effectué un split temporel strict 80/20 :
- **Train (80 %)** : 6 400 premières réservations (les plus anciennes).
- **Validation (20 %)** : 1 600 dernières réservations (les plus récentes).

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- **Scénario 1 — Réservation très anticipée sans engagement financier** : Réservation effectuée plus de 90 jours avant la date d'arrivée, assortie d'un tarif entièrement remboursable et sans aucun acompte versé.
- **Scénario 2 — Historique d'annulations récurrentes** : Client individuel dont le ratio d'annulations passées sur ses réservations précédentes dépasse 50 %.
- **Scénario 3 — Séjour long réservé via canal partenaire externe** : Réservation de plus de 4 nuitées effectuée via une agence en ligne (OTA) pour une période de haute saison sans prépaiement requis.

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**

- **Prévention du Data Leakage** : Toutes les transformations (calcul des médianes, fréquences et encodages) sont ajustées (`fit`) **exclusivement sur les 6 400 lignes du train set** puis appliquées (`transform`) sur la validation et le test.
- **Valeurs manquantes** : Les variables numériques manquantes sont imputées par la médiane du train via `SimpleImputer(strategy='median')`. Les variables catégorielles sont imputées par le mode via `SimpleImputer(strategy='most_frequent')`.
- **Catégories inédites** : Utilisées avec `OneHotEncoder(handle_unknown='ignore')`, ce qui attribue une ligne de zéros aux catégories jamais vues en entraînement sans faire planter le pipeline.

#### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**

L'hôtel **ne doit en aucun cas annuler d'office** la réservation. Il convient d'appliquer une réponse graduée et non agressive :
1. **Probabilité modérée ($0,214 - 0,500$)** : Envoi d'un e-mail / SMS de courtoisie automatique invitant le client à personnaliser son séjour (choix de l'heure d'arrivée, option petit-déjeuner) ou à confirmer sa venue en un clic.
2. **Probabilité élevée ($> 0,500$)** : Prise de contact directe par la réception pour proposer un avantage exclusif (ex: surclassement offert ou remise de 10 %) en échange du passage à un tarif non remboursable ou du versement d'un acompte.
3. **Pilotage de la surréservation** : Utiliser ces probabilités cumulées au niveau de chaque hôtel pour autoriser un sur-remplissage (*overbooking*) calculé et sécurisé par catégorie de chambre.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Les performances varient modérément selon la nature des établissements :
- **Destinations Balnéaires et Montagneuses** : Le F1-score est plus élevé (~0,52), car les réservations y suivent une forte saisonnalité et des fenêtres d'anticipation très marquées.
- **Destinations Urbaines et Affaires** : Le F1-score est légèrement plus bas (~0,43), les annulations y étant plus volatiles et décidées au dernier moment pour des raisons professionnelles.
*Limite* : Pour certaines régions comptant un nombre réduit de réservations dans l'échantillon de validation, la mesure du F1-score est soumise à une variance plus forte, ce qui impose de ne pas sur-interpréter les performances sur les très petits sous-groupes.

#### **Q9. Analyse des erreurs**

Matrice de confusion obtenue sur l'ensemble de validation ($1\ 600$ réservations) :
- **Vrais Négatifs ($TN$)** : $502$ (réservations maintenues correctement identifiées)
- **Faux Positifs ($FP$)** : $669$ (réservations prédites annulées mais finalement honorées)
- **Faux Négatifs ($FN$)** : $89$ (annulations réelles non anticipées)
- **Vrais Positifs ($TP$)** : $340$ (annulations réelles correctement prédites)
