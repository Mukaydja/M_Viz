# am-fcp.py

# --- IMPORTS (ajouts pour l'authentification) ---
import uuid
import json
import time
import os
import hashlib
import hmac
import re

# --- IMPORTS EXISTANTS (inchangés, à conserver) ---
import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import patheffects
from matplotlib.patches import Patch # Pour la légende
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import base64

# --- FONCTIONS D'AUTHENTIFICATION (NOUVEAU) ---
# Chemins vers les fichiers "base de données" simulée
USER_DB_FILE = "registered_users.json" # Stocke les identifiants et mots de passe hashés
AUTHORIZED_USERS_FILE = "authorized_users.json" # Stocke les identifiants autorisés
# Mot de passe admin (devrait être dans un secret, mais pour simplifier...)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123") # Changez-le dans les secrets de déploiement

def hash_password(password):
    """Hash un mot de passe avec SHA-256 (basique, pas bcrypt mais fonctionnel pour cet exemple)."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_registered_users():
    """Charge la liste des utilisateurs enregistrés depuis un fichier JSON."""
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_registered_user(username, password_hash):
    """Sauvegarde un nouvel utilisateur enregistré dans le fichier JSON."""
    users = load_registered_users()
    users[username] = password_hash
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def load_authorized_users():
    """Charge la liste des utilisateurs autorisés depuis un fichier JSON."""
    try:
        with open(AUTHORIZED_USERS_FILE, "r") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_authorized_users(users_set):
    """Sauvegarde la liste des utilisateurs autorisés dans un fichier JSON."""
    with open(AUTHORIZED_USERS_FILE, "w") as f:
        json.dump(list(users_set), f)

def is_valid_username(username):
    """Vérifie si un nom d'utilisateur est valide."""
    # Doit contenir entre 3 et 20 caractères, uniquement lettres, chiffres, underscore ou tiret
    return bool(re.match(r"^[a-zA-Z0-9_-]{3,20}$", username))

def is_user_authorized():
    """
    Vérifie si l'utilisateur actuel est connecté et autorisé.
    Pour l'administrateur, vérifie le mot de passe admin stocké.
    Pour les utilisateurs normaux, vérifie s'ils sont dans la liste des autorisés.
    """
    # Vérifier si l'utilisateur est connecté (session_state)
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        return False
    
    # Vérifier si c'est l'administrateur (spécial ou enregistré)
    if 'is_admin' in st.session_state and st.session_state['is_admin']:
        return True
        
    # Vérifier si c'est un utilisateur normal connecté avec un identifiant
    if 'username' in st.session_state:
        authorized_users = load_authorized_users()
        return st.session_state['username'] in authorized_users
    
    return False

def login_page():
    """Affiche la page de connexion/inscription."""
    st.title("⚽ Accès à l'Outil de Visualisation Footballistique")
    
    st.markdown("""
    ### Bienvenue !
    Cet outil vous permet d'analyser et de visualiser vos données footballistiques.
    
    **Fonctionnalités :**
    - Upload de fichiers CSV
    - Visualisation des événements sur le terrain
    - Génération de heatmaps
    - Création de rapports PDF détaillés
    
    **Accès restreint :** Un abonnement mensuel est requis pour utiliser cet outil.
    """)

    st.markdown("---")
    st.subheader("🔐 Accès Utilisateur")
    
    # Tabs pour Inscription / Connexion
    tab1, tab2 = st.tabs(["📝 Nouvel Utilisateur", "🔑 Déjà Inscrit"])
    
    with tab1:
        st.markdown("**Créez un compte pour commencer.**")
        new_username = st.text_input("Choisissez votre Identifiant Unique (3-20 caractères, lettres, chiffres, _ ou -)", key="new_username")
        new_user_password = st.text_input("Choisissez un mot de passe", type="password", key="new_password")
        if st.button("Créer un Compte"):
            if not new_username or not new_user_password:
                st.error("Veuillez remplir tous les champs.")
            elif not is_valid_username(new_username):
                st.error("Identifiant invalide. Il doit contenir 3 à 20 caractères, uniquement des lettres, chiffres, underscores (_) ou tirets (-).")
            else:
                registered_users = load_registered_users()
                authorized_users = load_authorized_users()
                
                if new_username in registered_users or new_username in authorized_users:
                    st.error("Cet identifiant est déjà pris. Veuillez en choisir un autre.")
                else:
                    password_hash = hash_password(new_user_password)
                    save_registered_user(new_username, password_hash)
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = new_username
                    st.success(f"Compte créé avec succès ! Bienvenue, {new_username} !")
                    st.info(f"Votre **Identifiant Unique** est : `{new_username}`")
                    st.warning("⚠️ **Important :** Conservez cet identifiant. Vous en aurez besoin pour activer votre abonnement.")
                    st.markdown("**Prochaine étape :** Procédez au paiement mensuel.")
                    # Bouton pour aller vers le paiement (à configurer avec votre lien Stripe)
                    st.link_button("💳 Procéder au Paiement Mensuel", "https://buy.stripe.com/test_8x228qfqfbeS5Zed0ldEs00")
                    st.markdown("*(Pour l'instant, le paiement est simulé. Contactez l'administrateur avec votre ID pour activation.)*")
                    time.sleep(2) # Laisser le message s'afficher
                    # On ne recharge pas ici pour montrer le message

    with tab2:
        st.markdown("**Connectez-vous avec votre identifiant unique.**")
        username_input = st.text_input("Votre Identifiant Unique", key="login_username")
        password_input = st.text_input("Votre Mot de passe", type="password", key="login_password")
        if st.button("Se Connecter"):
            if username_input and password_input:
                # Cas spécial : Connexion administrateur directe (si ADMIN_USERNAME est utilisé)
                # if username_input == ADMIN_USERNAME and password_input == ADMIN_PASSWORD:
                #     st.session_state['logged_in'] = True
                #     st.session_state['username'] = ADMIN_USERNAME
                #     st.session_state['is_admin'] = True # Marquer comme admin
                #     st.success(f"Connexion administrateur réussie ! Bienvenue, {ADMIN_USERNAME} !")
                #     time.sleep(1)
                #     st.rerun() # Recharger pour afficher l'application principale
                
                # Cas normal : Vérification des utilisateurs enregistrés
                registered_users = load_registered_users()
                password_hash = hash_password(password_input)
                
                if username_input in registered_users and registered_users[username_input] == password_hash:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username_input
                    # Ne pas marquer comme admin ici
                    
                    authorized_users = load_authorized_users()
                    if username_input in authorized_users:
                        st.success(f"Connexion réussie ! Bienvenue, {username_input} !")
                        time.sleep(1)
                        st.rerun() # Recharger pour afficher l'application principale
                    else:
                        # L'utilisateur existe (normalement) mais n'est pas encore autorisé
                        st.session_state['logged_in'] = True # On le connecte temporairement
                        st.session_state['username'] = username_input
                        st.info(f"Bonjour, {username_input} ! Votre identifiant est reconnu.")
                        st.warning("🔒 Votre compte n'est pas encore activé. Veuillez vérifier que votre abonnement est en cours.")
                        # Bouton pour aller vers le paiement
                        st.link_button("💳 Vérifier/Procéder au Paiement", "https://buy.stripe.com/test_8x228qfqfbeS5Zed0ldEs00")
                        st.markdown("*(Si vous avez déjà payé, contactez l'administrateur avec votre ID.)*")
                        # On ne recharge pas ici pour montrer le message
                else:
                    st.error("Identifiant ou mot de passe incorrect.")
            else:
                st.error("Veuillez entrer votre identifiant et votre mot de passe.")

    # --- Section Admin (pour vous) ---
    st.markdown("---")
    with st.expander("🛠️ Panneau d'Administration (Pour vous)"):
        admin_password_input = st.text_input("Mot de passe administrateur", type="password")
        # Vérifier si le mot de passe est correct
        if admin_password_input == ADMIN_PASSWORD:
            st.success("✅ Accès administrateur")
            
            st.subheader("Ajouter un utilisateur autorisé")
            user_to_add = st.text_input("Identifiant de l'utilisateur à autoriser (doit exister)")
            if st.button("Autoriser l'Utilisateur"):
                if user_to_add:
                    registered_users = load_registered_users()
                    if user_to_add in registered_users:
                        authorized_users = load_authorized_users()
                        authorized_users.add(user_to_add)
                        save_authorized_users(authorized_users)
                        st.success(f"Utilisateur `{user_to_add}` autorisé avec succès !")
                    else:
                        st.warning("Cet identifiant n'existe pas dans la base des utilisateurs enregistrés.")
                else:
                    st.warning("Veuillez entrer un identifiant.")
            
            if st.button("Voir la liste des utilisateurs autorisés"):
                authorized_users = load_authorized_users()
                if authorized_users:
                    st.write("Utilisateurs autorisés :")
                    # Afficher dans un textarea pour faciliter la copie
                    st.text_area("Liste des ID", value="\n".join(authorized_users), height=150, key="admin_user_list")
                else:
                    st.info("Aucun utilisateur autorisé pour le moment.")
            
            user_to_remove = st.text_input("Retirer un ID utilisateur")
            if st.button("Retirer l'Utilisateur"):
                if user_to_remove:
                    authorized_users = load_authorized_users()
                    if user_to_remove in authorized_users:
                        authorized_users.remove(user_to_remove)
                        save_authorized_users(authorized_users)
                        st.success(f"Utilisateur `{user_to_remove}` retiré avec succès.")
                        # Si l'utilisateur retiré est l'utilisateur actuel, le déconnecter
                        if 'username' in st.session_state and st.session_state['username'] == user_to_remove:
                            st.session_state['logged_in'] = False
                            if 'username' in st.session_state: del st.session_state['username']
                            if 'is_admin' in st.session_state: del st.session_state['is_admin']
                            st.info("Votre accès a été révoqué. Vous avez été déconnecté.")
                    else:
                        st.warning("ID utilisateur non trouvé.")
                else:
                    st.warning("Veuillez entrer un ID utilisateur.")
        elif admin_password_input:
            st.error("🔐 Mot de passe administrateur incorrect.")

# --- VÉRIFICATION D'ACCÈS AU DÉBUT DU SCRIPT (NOUVEAU) ---
# Vérifier si l'utilisateur est autorisé au début de l'exécution
if not is_user_authorized():
    # Si l'utilisateur n'est pas connecté ou pas autorisé, afficher la page de login
    login_page()
    
    # Permettre la connexion temporaire pour voir le message
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        # Afficher un message personnalisé
        if 'username' in st.session_state:
            st.markdown(f"Bonjour, **{st.session_state['username']}** !")
        
        # Afficher le bouton de déconnexion
        if st.button("Se Déconnecter"):
            # Nettoyer la session
            keys_to_delete = [key for key in st.session_state.keys() if key.startswith('logged_in') or key.startswith('user_') or key.startswith('is_')]
            for key in keys_to_delete:
                del st.session_state[key]
            st.success("Vous avez été déconnecté.")
            st.rerun()
    
    st.stop() # Arrêter l'exécution du reste du code

# --- CONTENU DE L'APPLICATION PRINCIPALE (votre code existant commence ici) ---

# Afficher un message de bienvenue personnalisé
if 'username' in st.session_state:
    st.set_page_config(page_title=f"Visualisation Foot - {st.session_state['username']}", layout="wide")
    st.title(f"⚽ Outil de Visualisation de Données Footballistiques - Bienvenue, {st.session_state['username']} !")
else:
    st.set_page_config(page_title="Visualisation Foot", layout="wide")
    st.title("Outil de Visualisation de Données Footballistiques")

# --- Upload CSV ---
st.sidebar.header("📁 Données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])
if not uploaded_file:
    st.warning("Veuillez importer un fichier CSV.")
    st.stop()
# --- Lecture du CSV avec feedback ---
with st.spinner("Lecture du fichier CSV..."):
    try:
        content = uploaded_file.read().decode('utf-8')
        sep = ';' if ';' in content.splitlines()[0] else ','
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du CSV : {e}")
        st.stop()
# --- Aperçu des données ---
with st.expander("🔍 Aperçu des données importées"):
    st.write("**Structure du DataFrame :**")
    st.write(df.dtypes)
    st.write("**Premières lignes :**")
    st.dataframe(df.head())
# --- Nettoyage des colonnes nécessaires ---
required_columns = ['Team', 'Player', 'Event', 'X', 'Y', 'X2', 'Y2']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Colonnes manquantes dans le fichier CSV : {missing_columns}")
    st.stop()
df = df[required_columns]
df = df.dropna(subset=['Player', 'Event', 'X', 'Y']).reset_index(drop=True)
# --- Conversion des colonnes numériques ---
for col in ['X', 'Y', 'X2', 'Y2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# --- Conversion des coordonnées si en 0–100 ---
# Ajout d'une vérification plus robuste
max_coord = df[['X', 'Y', 'X2', 'Y2']].max().max()
if pd.notna(max_coord) and max_coord <= 105 and max_coord > 50: # Hypothèse raisonnable
    st.info("Coordonnées détectées en format 0-100. Conversion vers 0-120/0-80.")
    df['X'] *= 1.2
    df['X2'] *= 1.2
    df['Y'] *= 0.8
    df['Y2'] *= 0.8
# --- Nettoyage des textes ---
df['Team'] = df['Team'].fillna('AS Monaco').astype(str).str.strip().str.title()
df['Player'] = df['Player'].fillna('').astype(str).str.strip().str.title()
df['Event'] = (
    df['Event']
    .fillna('')
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r'\s+', ' ', regex=True)
    .str.title()
)
# --- FILTRES ---
st.sidebar.header("🔍 Filtres Principaux")
event_options = sorted(df['Event'].dropna().unique())
player_options = sorted(df['Player'].dropna().unique())
team_options = sorted(df['Team'].dropna().unique())
selected_teams = st.sidebar.multiselect("Équipes", team_options, default=team_options)
selected_players = st.sidebar.multiselect("Joueurs", player_options, default=player_options)
displayed_events = st.sidebar.multiselect(
    "Événements à afficher",
    options=event_options,
    default=["Pass"] if "Pass" in event_options else event_options[:1]
)

# --- Classification par zones de terrain (AVANT les filtres pour avoir les options de filtre) ---
# *** MODIFIÉE : SANS SURFACE DE RÉPARATION ***
def classify_zone(x, y):
    """
    Classifie un point (x, y) dans une zone du terrain.
    Les zones sont mutuellement exclusives.
    Ordre de priorité: Haute > Médiane > Basse
    INVERSÉ : Basse et Haute sont interchangées
    """
    # 1. Haute (approche de la surface adverse, mais en dehors de la Surface Rép.)
    # INVERSÉ : Haute est maintenant près du fond de son propre camp (ancienne Basse)
    if 0 <= x < 36:
        return 'Haute'
    # 2. Médiane
    elif 36 <= x <= 90:
        return 'Médiane'
    # 3. Basse (fond de son propre camp)
    # INVERSÉ : Basse est maintenant près de la surface adverse (ancienne Haute)
    elif 90 < x <= 120: # Étendu à 120
        return 'Basse'
    else:
        # Pour les cas limites ou erreurs, on met par défaut dans une zone centrale
        return 'Médiane' # ou 'Inconnue'
# Ajouter temporairement la colonne Zone pour obtenir les options de filtre
df['Zone_temp'] = df.apply(lambda row: classify_zone(row['X'], row['Y']), axis=1)
zone_options = sorted(df['Zone_temp'].dropna().unique())
del df['Zone_temp'] # Supprimer la colonne temporaire
# Filtre par zone dans la sidebar (avant les options de visualisation)
# Note : La "Surface de Réparation" n'est plus une option
selected_zones = st.sidebar.multiselect(
    "Zones du Terrain",
    options=zone_options,
    default=zone_options,
    help="Filtrer les événements par zone où ils ont eu lieu."
)
# --- Options de Visualisation ---
st.sidebar.header("⚙️ Options d'Affichage")
show_legend = st.sidebar.checkbox("Afficher la légende des événements", value=True)
# Nouvelle option pour la palette de couleurs avec des noms lisibles
# Dictionnaire de correspondance Nom Affiché -> Nom Matplotlib
PALETTE_OPTIONS = {
    'Par défaut (Couleurs spécifiques + Tab20)': 'Par défaut',
    'Tab20 (Couleurs vives)': 'tab20',
    'Tab20b (Couleurs vives, variante B)': 'tab20b',
    'Set1 (Couleurs distinctes)': 'Set1',
    'Set2 (Couleurs pastel)': 'Set2',
    'Viridis (Dégradé violet-jaune)': 'viridis',
    'Plasma (Dégradé violet-rose-jaune)': 'plasma',
    'Coolwarm (Dégradé bleu-rouge)': 'coolwarm',
    'Pastel1 (Couleurs pastel vives)': 'Pastel1',
    'Dark2 (Couleurs sombres)': 'Dark2'
}
# Récupérer la liste des noms affichés pour le selectbox
display_names_list = list(PALETTE_OPTIONS.keys())
# Options avancées dans un expander
with st.sidebar.expander("➕ Options Avancées"):
    st.markdown("**Flèches (Passes, etc.)**")
    arrow_width = st.slider("Épaisseur des flèches", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="arrow_width")
    arrow_head_scale = st.slider("Taille de la tête des flèches", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="arrow_head_scale")
    arrow_alpha = st.slider("Opacité des flèches", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="arrow_alpha")
    st.markdown("**Points (Tirs, Tacles, etc.)**")
    point_size = st.slider("Taille des points", min_value=20, max_value=200, value=80, step=10, key="point_size")
    scatter_alpha = st.slider("Opacité des points", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="scatter_alpha")
    st.markdown("**Heatmap**")
    heatmap_alpha = st.slider("Opacité de la heatmap", min_value=0.1, max_value=1.0, value=0.85, step=0.05, key="heatmap_alpha")
    # NOUVEAUTÉS : Options avancées pour la heatmap (TYPE DE STATISTIQUE SUPPRIMÉ)
    st.markdown("**Options Avancées Heatmap**")
    # Option pour masquer/afficher les labels
    show_heatmap_labels = st.checkbox("Afficher les labels sur la heatmap", value=True, key="show_heatmap_labels")
    # NOUVEAUTÉ : Option pour masquer les labels 0%
    hide_zero_percent_labels = st.checkbox(
        "Masquer les labels 0%",
        value=True, # Activé par défaut
        help="Masquer les pourcentages à 0% sur la heatmap pour une meilleure lisibilité."
    )
    st.markdown("**Palette de Couleurs**")
    selected_palette_display_name = st.selectbox(
        "Palette de couleurs",
        options=display_names_list,
        index=0,
        help="Choisissez une palette de couleurs pour les types d'événements."
    )
    # Obtenir le nom technique de la palette à utiliser
    color_palette_name = PALETTE_OPTIONS[selected_palette_display_name]
# --- Filtres globaux ---
df_filtered = df[
    df['Team'].isin(selected_teams) &
    df['Player'].isin(selected_players) &
    df['Event'].isin(displayed_events)
]
# Appliquer le filtre par zone
df_filtered = df_filtered[
    df_filtered.apply(lambda row: classify_zone(row['X'], row['Y']), axis=1).isin(selected_zones)
]
df_event = df_filtered
# --- Vérification après filtrage ---
if df_event.empty:
    st.warning("Aucun événement ne correspond aux filtres sélectionnés.")
    st.stop()
# --- Ajout de la colonne Zone définitive ---
df_event['Zone'] = df_event.apply(lambda row: classify_zone(row['X'], row['Y']), axis=1)
# --- TABLEAU QUANTITATIF PAR TYPE ET ZONE ---
st.header("Quantité d'Événements par Type et Zone")
zone_counts = df_event.groupby(['Event', 'Zone']).size().unstack(fill_value=0)
zone_counts['Total'] = zone_counts.sum(axis=1)
zone_counts = zone_counts.sort_values(by='Total', ascending=False)
st.dataframe(zone_counts)
# --- TABLEAU DES POURCENTAGES PAR ZONE ---
st.subheader("Pourcentage d'Événements par Zone")
zone_total = df_event['Zone'].value_counts().reset_index()
zone_total.columns = ['Zone', 'Total']
total_events = zone_total['Total'].sum()
zone_total['Pourcentage'] = (zone_total['Total'] / total_events * 100).round(1)
# Affichage du tableau avec mise en couleur
st.dataframe(
    zone_total.style.background_gradient(cmap='Reds', subset=['Pourcentage']).format({"Pourcentage": "{:.1f}%"})
)
# Visualisation sur terrain avec répartition en zones
st.markdown("---")
st.header("Visualisations sur Terrain - Analyse par Zones")
# --- Paramètres communs pour les deux terrains ---
common_pitch_params = {
    'pitch_color': 'white',
    'line_color': 'black',
    'linewidth': 1,
    'line_zorder': 2
}
fig_size = (8, 5.5) # Taille uniforme et légèrement ajustée
# Définition des zones pour la visualisation (MODIFIÉE - SANS SURFACE DE RÉPARATION)
# INVERSÉ : Les rectangles correspondent à la logique de classification interchangée
zones_rects = {
    'Haute': (0, 0, 36, 80),      # Ancienne Basse
    'Médiane': (36, 0, 54, 80),   # 90-36=54
    'Basse': (90, 0, 30, 80),     # Ancienne Haute (120-90=30, ajusté)
    # 'Surface Rép.' n'est plus définie ici
}
zone_colors = {
    'Haute': '#87CEEB',          # Bleu clair
    'Médiane': '#98FB98',        # Vert pâle
    'Basse': '#FFD700',          # Or
    # 'Surface Rép.': '#FF6347'     # Rouge tomate
}
col_a, col_b = st.columns(2)
# --- Terrain avec pourcentages ---
with col_a:
    pitch_zone = Pitch(**common_pitch_params)
    fig_zone, ax_zone = pitch_zone.draw(figsize=fig_size)
    fig_zone.set_facecolor('white') # Définir la couleur de fond de la figure
    zone_percents = df_event['Zone'].value_counts(normalize=True).to_dict()
    for zone, (x, y, w, h) in zones_rects.items():
        percent = zone_percents.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=zone_colors.get(zone, '#DDDDDD'), alpha=0.7)
        ax_zone.add_patch(rect)
        # Ajustement de la taille du texte pour une meilleure lisibilité
        # Centrer le texte dans le rectangle
        ax_zone.text(x + w/2, y + h/2, f"{zone}\n{percent*100:.1f}%", ha='center', va='center', fontsize=8, weight='bold')
    ax_zone.set_title("Répartition en Pourcentages", fontsize=12, weight='bold', pad=10) # pad pour l'espacement
    st.pyplot(fig_zone)
# --- Terrain avec nombre d'événements ---
with col_b:
    pitch_count = Pitch(**common_pitch_params) # Utilisation des mêmes paramètres
    fig_count, ax_count = pitch_count.draw(figsize=fig_size)
    fig_count.set_facecolor('white') # Définir la couleur de fond de la figure
    zone_counts_dict = df_event['Zone'].value_counts().to_dict()
    for zone, (x, y, w, h) in zones_rects.items():
        count = zone_counts_dict.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=zone_colors.get(zone, '#DDDDDD'), alpha=0.7)
        ax_count.add_patch(rect)
        # Ajustement de la taille du texte pour une meilleure lisibilité et cohérence
        # Centrer le texte dans le rectangle
        ax_count.text(x + w/2, y + h/2, f"{zone}\n{count} evt", ha='center', va='center', fontsize=8, weight='bold')
    ax_count.set_title("Nombre d'Événements", fontsize=12, weight='bold', pad=10) # pad pour l'espacement
    st.pyplot(fig_count)
# --- VISUALISATION TERRAIN ---
st.markdown("---")
st.subheader("Visualisations sur Terrain")
# Générer le dictionnaire de couleurs en fonction du choix de l'utilisateur
def get_event_colors(event_list, palette_name, base_colors_dict):
    if palette_name == 'Par défaut':
        # Utiliser les couleurs de base + une palette pour les autres
        cmap_for_others = cm.get_cmap('tab20', max(1, len(event_list)))
        generated_colors = {event: mcolors.to_hex(cmap_for_others(i)) for i, event in enumerate([e for e in event_list if e not in base_colors_dict])}
        return {**base_colors_dict, **generated_colors}
    else:
        # Utiliser la palette sélectionnée pour tous les événements
        try:
            cmap_selected = cm.get_cmap(palette_name, max(1, len(event_list)))
            return {event: mcolors.to_hex(cmap_selected(i)) for i, event in enumerate(event_list)}
        except ValueError:
            # En cas d'erreur (palette non trouvée), retomber sur 'tab20'
            st.warning(f"Palette '{palette_name}' non trouvée. Utilisation de 'tab20'.")
            cmap_fallback = cm.get_cmap('tab20', max(1, len(event_list)))
            return {event: mcolors.to_hex(cmap_fallback(i)) for i, event in enumerate(event_list)}
# Définir les couleurs de base
base_colors = {
    'Shot': '#FF4B4B', 'Pass': '#6C9AC3', 'Dribble': '#FFA500',
    'Cross': '#92c952', 'Tackle': '#A52A2A', 'Interception': '#FFD700',
    'Clearance': '#00CED1'
}
# Générer le dictionnaire final des couleurs
# Utilise color_palette_name obtenu depuis le selectbox
event_colors = get_event_colors(event_options, color_palette_name, base_colors)
col1, col2 = st.columns(2)
with col1:
    with st.spinner("Génération de la visualisation des événements..."):
        pitch = Pitch(pitch_color='white', line_color='black', linewidth=1)
        fig1, ax1 = pitch.draw(figsize=(10, 6))
        legend_elements = [] # Pour stocker les éléments de légende
        for event_type in displayed_events:
            event_data = df_event[df_event['Event'] == event_type]
            color = event_colors.get(event_type, '#333333')
            has_xy2 = event_data[['X2', 'Y2']].notna().all(axis=1)
            # Utiliser les valeurs personnalisées pour les flèches
            if has_xy2.any():
                pitch.arrows(
                    event_data[has_xy2]['X'], event_data[has_xy2]['Y'],
                    event_data[has_xy2]['X2'], event_data[has_xy2]['Y2'],
                    color=color, width=arrow_width, headwidth=3 * arrow_head_scale, headlength=2 * arrow_head_scale,
                    alpha=arrow_alpha, ax=ax1
                )
            # Utiliser les valeurs personnalisées pour les points
            if (~has_xy2).any():
                pitch.scatter(
                    event_data[~has_xy2]['X'], event_data[~has_xy2]['Y'],
                    ax=ax1, fc=color, ec='black', lw=0.5, s=point_size, alpha=scatter_alpha
                )
            if show_legend:
                 legend_elements.append(Patch(facecolor=color, label=event_type))
        ax1.set_title("Visualisation des Événements", fontsize=12, weight='bold')
        fig1.set_facecolor('white')
        # Ajuster la mise en page AVANT d'ajouter la légende pour éviter la compression
        if show_legend and legend_elements:
            # Ajuster la taille de la figure pour accommoder la légende
            fig1.set_size_inches(12, 6)  # Élargir la figure
            ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            # Ajuster les marges pour que le terrain reste à la même taille
            plt.subplots_adjust(right=0.82)  # Laisser de la place à droite pour la légende
        else:
            plt.tight_layout()
        st.pyplot(fig1)
with col2:
    with st.spinner("Génération de la heatmap..."):
        pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2)
        fig2, ax2 = pitch.draw(figsize=(10, 6))
        fig2.set_facecolor('white')
        df_filtered_hm = df_event if len(displayed_events) != 1 else df_event[df_event['Event'] == displayed_events[0]]
        if not df_filtered_hm.empty:
            # Suppression de heatmap_statistic, utilisation directe de 'count'
            bin_statistic = pitch.bin_statistic(
                df_filtered_hm['X'], df_filtered_hm['Y'], statistic='count', bins=(6, 5), normalize=True
            )
            pitch.heatmap(bin_statistic, ax=ax2, cmap='Reds', edgecolor='white', alpha=heatmap_alpha)
            
            # >>>>> SOLUTION RADICALE : Utiliser pitch.label_heatmap <<<<<
            # Remplacer la boucle personnalisée problématique par l'appel standard
            if show_heatmap_labels:
                # Configurer les options d'affichage des labels
                # Masquer les 0% si demandé
                if hide_zero_percent_labels:
                    # Créer un masque pour les valeurs à 0
                    labels = bin_statistic['statistic']
                    labels = np.where(labels == 0, np.nan, labels)
                    # Passer le masque à label_heatmap
                    pitch.label_heatmap(bin_statistic, ax=ax2, str_format='{:.0%}', 
                                        fontsize=12, ha='center', va='center', 
                                        exclude_zeros=True, # Cette option masque les 0 et les NaN
                                        color='black')
                else:
                    # Afficher tous les labels
                    pitch.label_heatmap(bin_statistic, ax=ax2, str_format='{:.0%}', 
                                        fontsize=12, ha='center', va='center', 
                                        color='black')
                    
                # Appliquer les effets de contour si nécessaire
                # Note: label_heatmap ne permet pas directement d'ajouter path_effects.
                # On peut le faire manuellement après si nécessaire, mais cela complique le code.
                # Pour simplifier, on laisse comme ça. Si vous voulez les contours, 
                # il faudra revenir à une boucle personnalisée mais avec une approche différente.
                
        ax2.set_title("Heatmap des Événements", fontsize=12, weight='bold')
        st.pyplot(fig2)
st.markdown("---")
# --- CARTES COMBINÉES PAR TYPE D'ÉVÉNEMENT ---
if not df_event.empty:
    with st.expander("📊 Carte combinée par type d'événement", expanded=True):
        st.subheader("Carte combinée par type d'événement")
        grouped = [displayed_events[i:i + 3] for i in range(0, len(displayed_events), 3)]
        combined_images = []
        for group in grouped:
            row = st.container().columns(len(group))
            for i, event_type in enumerate(group):
                df_type = df_event[df_event['Event'] == event_type]
                if df_type.empty:
                    row[i].info(f"Aucun événement pour {event_type}")
                    continue
                color = event_colors.get(event_type, '#333333')
                pitch = Pitch(pitch_color='white', line_color='black', line_zorder=2)
                fig, ax = pitch.draw(figsize=(6, 4))
                arrows_data = df_type[df_type[['X2', 'Y2']].notna().all(axis=1)]
                if not arrows_data.empty:
                    # Utiliser les valeurs personnalisées pour les flèches
                    pitch.arrows(
                        arrows_data['X'], arrows_data['Y'],
                        arrows_data['X2'], arrows_data['Y2'],
                        ax=ax, zorder=10, color=color, alpha=arrow_alpha, width=arrow_width,
                        headwidth=3 * arrow_head_scale, headlength=2 * arrow_head_scale
                    )
                points_data = df_type[df_type[['X2', 'Y2']].isna().any(axis=1)]
                if not points_data.empty:
                     # Utiliser les valeurs personnalisées pour les points
                    pitch.scatter(
                        points_data['X'], points_data['Y'], ax=ax,
                        fc=color, marker='o', s=point_size, ec='black', lw=1,
                        alpha=scatter_alpha, zorder=5
                    )
                bin_stat = pitch.bin_statistic(df_type['X'], df_type['Y'], bins=(6, 5), normalize=True)
                event_cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'YlGnBu', 'PuRd']
                cmap_name = event_cmaps[i % len(event_cmaps)]
                pitch.heatmap(bin_stat, ax=ax, cmap=cmap_name, edgecolor='white', alpha=heatmap_alpha)
                
                # >>>>> SOLUTION RADICALE : Utiliser pitch.label_heatmap <<<<<
                # Remplacer la boucle personnalisée problématique par l'appel standard
                if show_heatmap_labels:
                    # Configurer les options d'affichage des labels
                    # Masquer les 0% si demandé
                    if hide_zero_percent_labels:
                        # Créer un masque pour les valeurs à 0
                        labels = bin_stat['statistic']
                        labels = np.where(labels == 0, np.nan, labels)
                        # Passer le masque à label_heatmap
                        pitch.label_heatmap(bin_stat, ax=ax, str_format='{:.0%}', 
                                            fontsize=10, ha='center', va='center', 
                                            exclude_zeros=True, # Cette option masque les 0 et les NaN
                                            color='black')
                    else:
                        # Afficher tous les labels
                        pitch.label_heatmap(bin_stat, ax=ax, str_format='{:.0%}', 
                                            fontsize=10, ha='center', va='center', 
                                            color='black')
                                            
                ax.set_title(event_type, color='black', fontsize=12, weight='bold')
                fig.set_facecolor('white')
                row[i].pyplot(fig)
                
                # Sauvegarder l'image pour le PDF
                with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    combined_images.append((event_type, tmpfile.name))
                plt.close(fig)  # Fermer la figure pour libérer la mémoire
# --- TÉLÉCHARGEMENT DU RAPPORT PDF FINAL ---
# Placer le bouton dans la sidebar, en bas
st.sidebar.markdown("---")
if st.sidebar.button("📥 Télécharger le rapport PDF complet"):
    with st.spinner("Génération du rapport PDF..."):
        # Format A3 paysage (420mm x 297mm) pour plus d'espace
        pdf = FPDF(orientation='L', unit='mm', format='A3')
        pdf.set_auto_page_break(auto=False)
        
        def add_footer():
            pdf.set_y(-10)
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 5, f"Page {pdf.page_no()}", 0, 0, 'C')

        # Préparer toutes les images avant de créer le PDF
        temp_files = []
        try:
            # --- Page 1: Page de garde ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 24)
            pdf.ln(40)
            pdf.cell(0, 15, "Rapport de Visualisation Footballistique", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", '', 14)
            pdf.cell(0, 10, f"Équipes : {', '.join(selected_teams)}", ln=True, align='C')
            pdf.ln(5)
            pdf.cell(0, 10, f"Joueurs : {', '.join(selected_players)}", ln=True, align='C')
            pdf.ln(5)
            pdf.cell(0, 10, f"Zones : {', '.join(selected_zones)}", ln=True, align='C')
            pdf.ln(5)
            pdf.cell(0, 10, f"Événements analysés : {', '.join(displayed_events)}", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 10, f"Nombre total d'événements : {len(df_event)}", ln=True, align='C')
            add_footer()

            # --- Page 2: Tableaux et terrains zonaux ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 12, "Analyse par Zones du Terrain", ln=True, align='C')
            pdf.ln(8)
            
            # Tableau quantitatif (partie supérieure gauche)
            pdf.set_xy(20, 30)  # Position fixe pour éviter les chevauchements
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(180, 8, "Événements par Type et Zone", ln=True)
            pdf.ln(3)
            
            # En-têtes du tableau
            col_names = ['Event'] + [col for col in zone_counts.columns if col != 'Total'] + ['Total']
            col_widths = [35] + [25] * (len(col_names) - 2) + [25]
            pdf.set_fill_color(220, 220, 220)
            for i, name in enumerate(col_names):
                pdf.cell(col_widths[i], 8, str(name), border=1, fill=True, align='C')
            pdf.ln()
            
            # Données du tableau
            pdf.set_font("Arial", size=10)
            pdf.set_fill_color(255, 255, 255)
            for idx, row in zone_counts.iterrows():
                pdf.cell(col_widths[0], 8, str(idx)[:25], border=1)  # Limiter la longueur
                for i, col in enumerate([col for col in zone_counts.columns if col != 'Total']):
                    pdf.cell(col_widths[i+1], 8, str(int(row[col])), border=1, align='C')
                pdf.cell(col_widths[-1], 8, str(int(row['Total'])), border=1, align='C')
                pdf.ln()
                
            # Tableau des pourcentages (partie supérieure droite avec position fixe)
            pdf.set_xy(250, 30)  # Position fixe à droite
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(125, 8, "Répartition par Zone (%)", ln=True)
            pdf.ln(3)
            
            pdf.set_xy(250, 46)  # Position fixe pour les en-têtes
            pdf.set_font("Arial", 'B', 10)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(50, 8, "Zone", border=1, fill=True, align='C')
            pdf.cell(35, 8, "Total", border=1, fill=True, align='C')
            pdf.cell(40, 8, "Pourcentage", border=1, fill=True, align='C')
            pdf.ln()
            
            pdf.set_font("Arial", size=10)
            pdf.set_fill_color(255, 255, 255)
            current_y = 54  # Y fixe pour commencer les données
            for _, row in zone_total.iterrows():
                pdf.set_xy(250, current_y)
                pdf.cell(50, 8, str(row['Zone']), border=1)
                pdf.cell(35, 8, str(row['Total']), border=1, align='C')
                pdf.cell(40, 8, f"{row['Pourcentage']:.1f}%", border=1, align='C')
                current_y += 8

            # Sauvegarder les terrains zonaux avec une meilleure résolution
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_zone_pct:
                fig_zone.savefig(tmp_zone_pct.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_zone_pct.name)
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_zone_count:
                fig_count.savefig(tmp_zone_count.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_zone_count.name)

            # Placer les terrains dans la partie inférieure (plus grands)
            terrain_y = 140
            pdf.image(tmp_zone_pct.name, x=50, y=terrain_y, w=140, h=90)
            pdf.image(tmp_zone_count.name, x=220, y=terrain_y, w=140, h=90)
            
            # Légendes sous les terrains
            pdf.set_xy(50, terrain_y + 95)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(140, 6, "Répartition en Pourcentages", align='C')
            pdf.set_xy(220, terrain_y + 95)
            pdf.cell(140, 6, "Nombre d'Événements", align='C')
            add_footer()

            # --- Page 3: Visualisations des terrains ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 12, "Visualisations sur Terrain", ln=True, align='C')
            pdf.ln(10)
            
            # Sauvegarder les visualisations avec une meilleure résolution
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_fig1:
                fig1.savefig(tmp_fig1.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_fig1.name)
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_fig2:
                fig2.savefig(tmp_fig2.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_fig2.name)

            # Placer les visualisations côte à côte avec plus d'espace
            terrain_width = 180
            terrain_height = 120
            pdf.image(tmp_fig1.name, x=30, y=40, w=terrain_width, h=terrain_height)
            pdf.image(tmp_fig2.name, x=230, y=40, w=terrain_width, h=terrain_height)
            
            # Légendes sous les terrains
            pdf.set_xy(30, 165)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(terrain_width, 8, "Événements sur le terrain", align='C')
            pdf.set_xy(230, 165)
            pdf.cell(terrain_width, 8, "Heatmap des événements", align='C')
            add_footer()

            # --- Page 4: Cartes combinées ---
            if combined_images:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18)
                pdf.cell(0, 12, "Cartes Combinées par Type d'Événement", ln=True, align='C')
                pdf.ln(10)
                
                # Organiser les images en grille adaptée au format A3
                img_width = 120
                img_height = 80
                margin_x = 30
                margin_y = 40
                cols = 3
                spacing_x = 15
                spacing_y = 20
                
                for i, (event_type, img_path) in enumerate(combined_images):
                    if i >= 6:  # Limite à 6 images par page (2 lignes de 3)
                        break
                    col = i % cols
                    row = i // cols
                    x = margin_x + col * (img_width + spacing_x)
                    y = margin_y + row * (img_height + spacing_y)
                    pdf.image(img_path, x=x, y=y, w=img_width, h=img_height)
                    
                    # Titre sous l'image
                    pdf.set_xy(x, y + img_height + 3)
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(img_width, 6, event_type, align='C')
                add_footer()

            # Générer et télécharger le PDF
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdf.output(tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as file:
                    pdf_data = file.read()
                    st.download_button(
                        "📄 Télécharger le PDF final du rapport",
                        data=pdf_data,
                        file_name="rapport_foot_A3.pdf",
                        mime="application/pdf"
                    )
                # Nettoyer le fichier PDF temporaire
                os.unlink(tmp_pdf.name)
        finally:
            # Nettoyer tous les fichiers temporaires
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            for _, img_path in combined_images:
                try:
                    os.unlink(img_path)
                except:
                    pass
