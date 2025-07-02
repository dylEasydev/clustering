import os
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score
import numpy as np
import pickle
import streamlit as st

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("Téléchargement des données 'punkt' de NLTK...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    st.info("Téléchargement des données 'omw-1.4' de NLTK (pour WordNetLemmatizer)...")
    nltk.download('omw-1.4')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    st.info("Téléchargement des données 'punkt_tab' de NLTK...")
    nltk.download('punkt_tab')

# Class PretraitementLyrics  pour charger et prétraiter les donnée
class PretraitementLyrics:
    def __init__(self, dossier):
        self.dossier = dossier

        # chargements des stopwords
        self.stopwords_fr = stopwords.words('french')
        self.stopwords_en = stopwords.words('english')
        self.stopwords_cmr = ["a","les","parol","parole","paroles","aah","ekié",
                              "aa","aaa","aahh","abel","abo","abok","achouk","ador","adou","zazou",
                              "affair","afric","african","afriqu","ahhhhh","aich","aid","ohhhhh","ohhhhhhh",
                              "combi","diman","bro","massah","wamda","per","moto","ah","oh","eh","owoo","owooh","é","yo","refrain","couplet"]
        # fusion de l'ensembles des stopwords 
        self.all_stopwords = list(set(self.stopwords_fr + self.stopwords_en + [s.lower() for s in self.stopwords_cmr]))
        self.dataSet = pd.DataFrame()

        self.vectorizer = None 
        self.X_tfidf = None   
        self.feature_names = None 
        self.X_pca = None      

        self.stemmer = SnowballStemmer("french") 
        self.lemmatizer = WordNetLemmatizer()    

    def nettoyer_texte(self, texte):
        tokens = word_tokenize(texte.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.all_stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def charger_lyrics(self):
        if not os.path.exists(self.dossier):
            st.error(f"Erreur : Le dossier spécifié '{self.dossier}' n'existe pas. Veuillez créer ce dossier et y placer vos fichiers de paroles (.txt).")
            return

        folder = Path(self.dossier)
        texts = [f.read_text(encoding='utf-8') for f in folder.iterdir() if f.is_file()]      
        labels = [f.name.split("_")[0] for f in folder.iterdir() if f.is_file()]
        self.dataSet = pd.DataFrame({'lyrics':texts , 'artiste':labels})
        found_files = True
       
        if not found_files:
            st.warning(f"Aucun fichier .txt trouvé dans le dossier '{self.dossier}'.")
        st.write(f"{len(self.dataSet["lyrics"])} documents chargés et prétraités.")


    def vectoriser(self):
        if self.dataSet.empty:
            st.warning("Aucun document à vectoriser. Exécutez d'abord charger_lyrics().")
            self.X_tfidf = None
            self.feature_names = None
            return
        self.dataSet["lyrics_clean"] = self.dataSet["lyrics"].apply(self.nettoyer_texte)
        self.vectorizer = TfidfVectorizer(stop_words=self.all_stopwords, max_features=5000)
        self.X_tfidf = self.vectorizer.fit_transform(self.dataSet["lyrics_clean"])
        self.feature_names = self.vectorizer.get_feature_names_out()
        st.write("TF-IDF vectorisation terminée. Shape:", self.X_tfidf.shape)

    def afficher_wordcloud(self):
        if self.dataSet.empty:
            st.warning("Aucun document trouvé. Exécutez d'abord charger_lyrics().")
            return
        texte_total = " ".join(self.dataSet["lyrics"])
        if not texte_total.strip():
            st.warning("Le texte total est vide après prétraitement. Impossible de générer un WordCloud.")
            return

        wc = WordCloud(width=800, height=400, stopwords=self.all_stopwords, background_color='white').generate(texte_total)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title("WordCloud Global")
        st.pyplot(fig)
        plt.close(fig)

    def reduire_pca(self):
        if self.X_tfidf is None:
            st.warning("TF-IDF non calculé. Exécutez d'abord vectoriser().")
            self.X_pca = None
            return
        if self.X_tfidf.shape[0] < 2 or self.X_tfidf.shape[1] < 2:
            st.warning("Nombre insuffisant de documents ou de caractéristiques pour PCA. Nécessite au moins 2.")
            self.X_pca = None
            return
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(self.X_tfidf.toarray())
        st.write("PCA réduction de dimension terminée. Shape:", self.X_pca.shape)

    def exporter_dataframe_tfidf(self):
        if self.dataSet.empty:
            st.warning("TF-IDF non calculé. Exécutez d'abord vectoriser().")
            return pd.DataFrame()
        return self.dataSet
    
    def draw_dataset(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(x=self.X_pca[:, 0], y=self.X_pca[:, 1], ax=ax, legend='full')
        ax.set_title(f"Projection des donnée")
        st.pyplot(fig)
        plt.close(fig) 

# Class EntrainementModeles pour le clustering et evaluation des models
class EntrainementModeles:
    def __init__(self, pretraitement):
        self.pretraitement = pretraitement

        self.kmeans = None
        self.gmm = None
        self.hierarchical_clustering = None 

    def _check_tfidf_and_pca(self):
        if self.pretraitement.X_tfidf is None:
            st.error("TF-IDF non calculé. Veuillez charger et préparer les données d'abord.")
            return False
        if self.pretraitement.X_pca is None:
            st.error("PCA non calculée. Veuillez charger et préparer les données d'abord.")
            return False
        if self.pretraitement.X_tfidf.shape[0] < 2:
            st.error("Nombre insuffisant de documents pour le clustering. Nécessite au moins 2.")
            return False
        return True

    def clustering_kmeans(self, n_clusters=3):
        
        st.subheader("=== Démarrage du clustering KMeans ===")
        if not self._check_tfidf_and_pca():
            return

        try:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) 
            self.pretraitement.dataSet["kmeans_labels"] = self.kmeans.fit_predict(self.pretraitement.X_tfidf)
            inertie = self.kmeans.inertia_
            score = calinski_harabasz_score(self.pretraitement.X_tfidf.toarray(), self.pretraitement.dataSet["kmeans_labels"])
            st.write(f"CH Score (KMeans) = {score:.2f} Inertie intra-cluster : {inertie:.2f}")

            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x=self.pretraitement.X_pca[:, 0], y=self.pretraitement.X_pca[:, 1],
                            hue=self.pretraitement.dataSet["kmeans_labels"], palette='tab10', ax=ax, legend='full')
            ax.set_title(f"Clustering TF-IDF avec KMeans (k={n_clusters})")
            st.pyplot(fig)
            plt.close(fig) 

            
            centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        except Exception as e:
            st.error(f"Erreur lors de l'exécution de KMeans : {e}")


    def clustering_gmm(self, n_clusters=3):
        st.subheader("=== Démarrage du clustering GMM ===")
        if not self._check_tfidf_and_pca():
            return

        try:
            st.write("Clustering avec GMM...")
            self.gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            self.pretraitement.dataSet["gmm_labels"] = self.gmm.fit_predict(self.pretraitement.X_tfidf.toarray())
            score = calinski_harabasz_score(self.pretraitement.X_tfidf.toarray(), self.pretraitement.dataSet["gmm_labels"])
            log_likelihood = self.gmm.score(self.pretraitement.X_tfidf.toarray()) 
            st.write(f"Log-vraisemblance: {log_likelihood:.2f} CH Score (GMM) = {score:.2f}")

            # affichage de l'espace PCA
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x=self.pretraitement.X_pca[:, 0], y=self.pretraitement.X_pca[:, 1],
                            hue=self.pretraitement.dataSet["gmm_labels"], palette='Set2', ax=ax, legend='full')
            ax.set_title(f"Clustering TF-IDF avec GMM (k={n_clusters})")
            st.pyplot(fig) 
            plt.close(fig)
        except Exception as e:
            st.error(f"Erreur lors de l'exécution de GMM : {e}")

    def clustering_hierarchique(self, n_clusters=3, linkage='ward'):
        st.subheader("=== Démarrage du clustering Hiérarchique ===")
        if not self._check_tfidf_and_pca():
            return

        try:
            st.write("Clustering Hiérarchique...")
            self.hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            self.pretraitement.dataSet["hierachical_labels"] = self.hierarchical_clustering.fit_predict(self.pretraitement.X_tfidf.toarray())
            score = calinski_harabasz_score(self.pretraitement.X_tfidf.toarray(), self.pretraitement.dataSet["hierachical_labels"])
            st.write(f"CH Score (Hiérarchique) = {score:.2f}")

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x=self.pretraitement.X_pca[:, 0], y=self.pretraitement.X_pca[:, 1],
                            hue=self.pretraitement.dataSet["hierachical_labels"], palette='viridis', ax=ax, legend='full')
            ax.set_title(f"Clustering TF-IDF avec Hiérarchique (k={n_clusters})")
            st.pyplot(fig)
            plt.close(fig) 
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du clustering hiérarchique : {e}")

    def afficher_documents_par_cluster(self, methode='kmeans', n_chars=200):
        if self.pretraitement.dataSet is None:
            st.warning("Aucun document ou label d'artiste disponible dans le prétraitement. Veuillez charger les données.")
            return
        st.subheader(f"Documents par Cluster ({methode.capitalize()})")
        if methode == 'kmeans':
            st.dataframe(self.pretraitement.dataSet[["lyrics" , "artiste" , "kmeans_labels"]])
        elif methode == 'gmm':
            st.dataframe(self.pretraitement.dataSet[["lyrics" , "artiste" , "gmm_labels"]])
        elif methode == 'hierarchical':
            st.dataframe(self.pretraitement.dataSet[["lyrics" , "artiste" , "hierachical_labels"]])
        else:
            st.error("Méthode invalide. Choisissez 'kmeans', 'gmm' ou 'hierarchical'.")
            return


   
    def afficher_comparaison_clusters(self):
        
        st.subheader("Comparaison des Clusters (Visualisation PCA)")
        if self.pretraitement.X_pca is None:
            st.error("PCA non trouvée. Exécutez d'abord reduire_pca().")
            return

        if self.pretraitement.dataSet["hierachical_labels"].empty or self.pretraitement.dataSet["gmm_labels"].empty or self.pretraitement.dataSet["kmeans_labels"].empty:
            st.warning("Tous les clusters (KMeans, GMM, Hiérarchique) doivent être générés avant d'afficher la comparaison.")
            return

       
        if self.pretraitement.X_pca.shape[0] < 1:
            st.warning("Pas assez de points de données pour afficher les graphiques de comparaison.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        if not self.pretraitement.dataSet["kmeans_labels"].empty:
            sns.scatterplot(ax=axes[0], x=self.pretraitement.X_pca[:, 0], y=self.pretraitement.X_pca[:, 1],
                            hue=self.pretraitement.dataSet["kmeans_labels"], palette='tab10', legend='full')
            axes[0].set_title("KMeans Clustering")
        else:
            axes[0].set_title("KMeans (Non exécuté)")
            axes[0].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

        # GMM 
        if not self.pretraitement.dataSet["gmm_labels"].empty:
            sns.scatterplot(ax=axes[1], x=self.pretraitement.X_pca[:, 0], y=self.pretraitement.X_pca[:, 1],
                            hue=self.pretraitement.dataSet["gmm_labels"], palette='Set2', legend='full')
            axes[1].set_title("GMM Clustering")
        else:
            axes[1].set_title("GMM (Non exécuté)")
            axes[1].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

        # Hierarchical 
        if not self.pretraitement.dataSet["hierachical_labels"].empty:
            sns.scatterplot(ax=axes[2], x=self.pretraitement.X_pca[:, 0], y=self.pretraitement.X_pca[:, 1],
                            hue=self.pretraitement.dataSet["hierachical_labels"], palette='viridis', legend='full')
            axes[2].set_title("Hierarchical Clustering")
        else:
            axes[2].set_title("Hierarchical (Non exécuté)")
            axes[2].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)

        st.pyplot(fig)
        plt.close(fig) 


# --- Streamlit Application ---
st.set_page_config(layout="wide") 
st.title("Application de Clustering de Paroles Musicales")


# entrer le chemin du fdossier du datasets
data_folder_input = "lyrics/lyrics_Artiste"

# Slider for number of clusters
n_clusters_input = 3


if 'pretraitement' not in st.session_state:
    st.session_state.pretraitement = PretraitementLyrics(data_folder_input)
if 'entrainement' not in st.session_state:
    st.session_state.entrainement = EntrainementModeles(st.session_state.pretraitement)




# --- Session Principale ---
st.header("Flux d'Analyse")

# Section 1: chargement des donnée
st.subheader("1. Chargement et Préparation des Données")
if st.button("Charger et Préparer les Données"):
    with st.spinner("Chargement et prétraitement des données..."):
        try:
            st.session_state.pretraitement.charger_lyrics()
            if not st.session_state.pretraitement.dataSet.empty: 
                st.session_state.pretraitement.vectoriser()
                st.session_state.pretraitement.reduire_pca()
                st.success("Données chargées et préparées avec succès !")
            else:
                st.warning("Aucun document chargé. Vérifiez le dossier des paroles et son contenu.")
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la préparation des données : {e}")

# Section 2: Visualizations
st.subheader("2. Visualisations")
if st.button("Afficher WordCloud Global"):
    if not st.session_state.pretraitement.dataSet.empty:
        st.session_state.pretraitement.afficher_wordcloud()
    else:
        st.warning("Veuillez charger et préparer les données d'abord.")

if st.button("Afficher les données nettoyer"):
    if not st.session_state.pretraitement.dataSet.empty:
        st.dataframe(st.session_state.pretraitement.dataSet[["artiste","lyrics_clean"]])
    else:
        st.warning("Veuillez charger et préparer les données d'abord.")

if st.button("Afficher les données "):
    if not st.session_state.pretraitement.dataSet.empty:
        st.session_state.pretraitement.draw_dataset()
    else:
        st.warning("Veuillez charger et préparer les données d'abord.")
# Section 3: Clustering Model Execution
st.subheader("3. Exécution des Modèles de Clustering")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Exécuter KMeans"):
        if st.session_state.pretraitement.X_tfidf is not None:
            st.session_state.entrainement.clustering_kmeans(n_clusters=n_clusters_input)
        else:
            st.warning("Veuillez charger et préparer les données d'abord.")

with col2:
    if st.button("Exécuter GMM"):
        if st.session_state.pretraitement.X_tfidf is not None:
            st.session_state.entrainement.clustering_gmm(n_clusters=n_clusters_input)
            st.subheader(f"Résultats du Clustering GMM (k={n_clusters_input})")
        else:
            st.warning("Veuillez charger et préparer les données d'abord.")

with col3:
    if st.button("Exécuter Hiérarchique"):
        if st.session_state.pretraitement.X_tfidf is not None:
            st.session_state.entrainement.clustering_hierarchique(n_clusters=n_clusters_input)
        else:
            st.warning("Veuillez charger et préparer les données d'abord.")


# Section 4: Analyse détaillé
st.subheader("4. Analyse Détaillée des Clusters")

cluster_method_display = st.selectbox(
    "Sélectionnez la méthode de clustering à afficher en détail :",
    ('kmeans', 'gmm', 'hierarchical'),
    index=1 
)

if st.button(f"Afficher Documents par Cluster ({cluster_method_display.capitalize()})"):
    st.session_state.entrainement.afficher_documents_par_cluster(methode=cluster_method_display)

# Section 5: Comparison des clustering 
st.subheader("5. Comparaison des Méthodes de Clustering")
if st.button("Afficher Comparaison des Clusters (Visualisation PCA)"):
    st.session_state.entrainement.afficher_comparaison_clusters()