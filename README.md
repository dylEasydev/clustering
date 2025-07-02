# Application de Clustering de Paroles Musicales

Ce projet présente une application interactive développée avec Streamlit pour charger, prétraiter  les donnée scrapper du site [https://kamer-lyrics.net](kamerlyrics) et analyser des paroles de chansons via diverses méthodes de clustering. L'objectif est de découvrir des groupes (clusters) naturels au sein d'un ensemble de paroles, basés sur leur contenu textuel.


## Fonctionnalités

  * **Chargement de Données** : Charge automatiquement les fichiers `.txt` de paroles musicales depuis un dossier spécifié, les associant à un artiste basé sur le nom du fichier.
  * **Prétraitement du Texte** : Nettoie les paroles en miniscule, supprime les caractères non alphabétiques, filtre les mots vides (stopwords en français, anglais et un ensemble personnalisé (stop word camerounais )), effectue la racinisation (stemming) et la lemmatisation.
  * **Vectorisation TF-IDF** : Convertit le texte nettoyé en représentations numériques (Term Frequency-Inverse Document Frequency) pour l'analyse.
  * **Réduction de Dimension PCA** : Réduit la complexité des données TF-IDF à 2 dimensions pour faciliter la visualisation.
  * **Visualisations Clés** :
      * **WordCloud Global** : Affiche les mots les plus fréquents dans l'ensemble du corpus.
      * **Projection PCA** : Visualise la répartition des documents dans l'espace réduit.
  * **Modèles de Clustering** : Applique et visualise les résultats de trois algorithmes de clustering populaires :
      * **K-Means** : clustering par partitionnement 
      * **Gaussian Mixture Model (GMM)**: clustering par mélange
      * **Agglomerative Clustering (Hiérarchique)** : clustering hieararchique 
  * **Évaluation des Clusters** : Affiche le score de Calinski-Harabasz (rapport entre inertie inter-cluster et inertie intra-cluster " plus il est grand , mieux les cluster sont former ") . pour chaque modèle de clustering, et l'inertie intra-cluster pour K-Means, ainsi que la log-vraisemblance pour GMM.
  * **Analyse Détaillée par Cluster** : Permet de consulter les documents originaux, l'artiste associé et le label de cluster attribué par chaque méthode.
  * **Comparaison des Clusters** : Visualise les résultats des trois méthodes de clustering côte à côte sur la projection PCA pour une comparaison facile.
  * **Interface Interactive** : Grâce à Streamlit, l'application est facile à utiliser avec des boutons et un curseur pour ajuster le nombre de clusters.


##  Démarrage Rapide

Suivez ces étapes pour lancer l'application sur votre machine locale.

### Prérequis

Assurez-vous d'avoir Python 3.8+ installé.

### 1\. Cloner le dépôt

Ouvrez votre terminal ou invite de commande et clonez le dépôt GitHub :

```bash
git clone https://github.com/dylEasydev/clustering.git
cd clustering
```

### 2\. Créer l'environnement des données
Récuperer le dossier zipper des lyrics scrapper  `lyrics.zip` disponible [](ici) 
que vous devez en suite extraire dans le dossier `./clustering` ou .

Créez un dossier nommé `lyrics` à la racine de votre projet, puis à l'intérieur de ce dossier `lyrics`, créez un sous-dossier nommé `lyrics_Artiste`.

Structure attendue :

```
clustering/
├── lyrics/
│   └── lyrics_Artiste/
│       ├── Artiste1_Chanson1.txt
│       ├── Artiste1_Chanson2.txt
│       └── Artiste2_Chanson1.txt
├── app.py
├── requirements.txt
└── README.md
```

Placez vos fichiers de paroles (`.txt`) dans le dossier `lyrics/lyrics_Artiste`. Assurez-vous que le nom de chaque fichier commence par le nom de l'artiste, suivi d'un underscore, comme par exemple `Adele_RollingInTheDeep.txt`.

### 3\. Installer les dépendances
installez toutes les bibliothèques nécessaires à partir du fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

### 4\. Lancer l'application Streamlit

Une fois les dépendances installées, vous pouvez lancer l'application :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur web par défaut.


##  sur Streamlit Cloud

Cette application est conçue pour être facilement déployable sur [Streamlit Cloud](https://streamlit.io/cloud).

1.  **Assurez-vous que `requirements.txt` est à jour** : Le fichier `requirements.txt` à la racine de votre dépôt doit lister toutes les dépendances Python nécessaires (Streamlit, NLTK, scikit-learn, pandas, matplotlib, seaborn, wordcloud, numpy).
2.  **Configuration des données NLTK** : Le script `app.py` contient des blocs de code pour télécharger automatiquement les ressources NLTK nécessaires (`stopwords`, `punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) la première fois que l'application est lancée sur un nouvel environnement (y compris Streamlit Cloud). Cela évite d'avoir à les gérer manuellement.
3.  **Structure du dossier des paroles** : Assurez-vous que le dossier `lyrics/lyrics_Artiste` contenant vos fichiers `.txt` est également commité et poussé vers votre dépôt GitHub. Streamlit Cloud clonera l'intégralité du dépôt.

Voici le  lien de notre application déploier [https://dyleasydev-clustering-app-bmmiyz.streamlit.app/](kamerlyrics-cluster)

## 🛠️ Technologies Utilisées

  * **Python 3.13**
  * **Streamlit** : Pour l'interface utilisateur interactive.
  * **NLTK (Natural Language Toolkit)** : Pour le prétraitement du texte (tokenisation, mots vides, lemmatisation, racinisation).
  * **scikit-learn** : Pour la vectorisation TF-IDF, la réduction de dimension PCA et les algorithmes de clustering (K-Means, GMM, AgglomerativeClustering).
  * **Pandas** : Pour la manipulation des données.
  * **Matplotlib** / **Seaborn** : Pour les visualisations graphiques.
  * **WordCloud** : Pour la génération de nuages de mots.
  * **NumPy** : Pour les opérations numériques.


## Auteur
