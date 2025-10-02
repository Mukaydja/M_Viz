# am-fcp.py — Version corrigée : zones inversées logiquement (Haute = offensive)

# --- IMPORTS (inchangés) ---
import uuid
import json
import time
import os
import hashlib
import re
import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import patheffects
from matplotlib.patches import Patch
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import base64

# --- AUTHENTIFICATION (inchangée, conservée comme dans ton fichier) ---
USER_DB_FILE = "registered_users.json"
AUTHORIZED_USERS_FILE = "authorized_users.json"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "MonSuperMotDePasseAdmin123!")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_registered_users():
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_registered_user(username, password_hash):
    users = load_registered_users()
    users[username] = password_hash
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def load_authorized_users():
    try:
        with open(AUTHORIZED_USERS_FILE, "r") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_authorized_users(users_set):
    with open(AUTHORIZED_USERS_FILE, "w") as f:
        json.dump(list(users_set), f)

def is_user_authorized():
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        return False
    if 'username' not in st.session_state:
        return False
    return st.session_state['username'] in load_authorized_users()

def is_valid_username(username):
    return bool(re.match(r"^[a-zA-Z0-9_-]{3,20}$", username))

def login_page():
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
    tab1, tab2 = st.tabs(["📝 Nouvel Utilisateur", "🔑 Déjà Inscrit"])
    with tab1:
        st.markdown("**Créez un compte pour commencer.**")
        new_username = st.text_input("Choisissez votre Identifiant Unique (3-20 caractères, lettres, chiffres, _ ou -)", key="new_username")
        new_user_password = st.text_input("Choisissez un mot de passe", type="password", key="new_password")
        if st.button("Créer un Compte"):
            if not new_username or not new_user_password:
                st.error("Veuillez remplir tous les champs.")
            elif not is_valid_username(new_username):
                st.error("Identifiant invalide.")
            else:
                registered_users = load_registered_users()
                authorized_users = load_authorized_users()
                if new_username in registered_users or new_username in authorized_users:
                    st.error("Cet identifiant est déjà pris.")
                else:
                    password_hash = hash_password(new_user_password)
                    save_registered_user(new_username, password_hash)
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = new_username
                    st.success(f"Compte créé avec succès ! Bienvenue, {new_username} !")
                    st.info(f"Votre **Identifiant Unique** est : `{new_username}`")
                    st.warning("⚠️ **Important :** Conservez cet identifiant.")
                    st.link_button("💳 Procéder au Paiement Mensuel", "https://buy.stripe.com/test_aFa9AS91R82G2N29O9dEs01")
                    time.sleep(2)
    with tab2:
        st.markdown("**Connectez-vous avec votre identifiant unique.**")
        username_input = st.text_input("Votre Identifiant Unique", key="login_id")
        password_input = st.text_input("Votre Mot de passe", type="password", key="login_password")
        if st.button("Se Connecter"):
            if username_input and password_input:
                registered_users = load_registered_users()
                password_hash = hash_password(password_input)
                if username_input in registered_users and registered_users[username_input] == password_hash:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username_input
                    st.success(f"Connexion réussie ! Bienvenue, {username_input} !")
                    time.sleep(1)
                    st.rerun()
                else:
                    if username_input in registered_users:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username_input
                        st.info(f"Bonjour, {username_input} ! Votre identifiant est reconnu.")
                        st.warning("🔒 Votre compte n'est pas encore activé.")
                        st.link_button("💳 Vérifier/Procéder au Paiement", "https://buy.stripe.com/test_aFa9AS91R82G2N29O9dEs01")
                    else:
                        st.error("Identifiant ou mot de passe incorrect.")
            else:
                st.error("Veuillez entrer un identifiant et un mot de passe.")
    st.markdown("---")
    with st.expander("🛠️ Panneau d'Administration (Pour vous)"):
        admin_password = st.text_input("Mot de passe administrateur", type="password")
        if admin_password == ADMIN_PASSWORD:
            st.success("Accès administrateur")
            admin_tab1, admin_tab2 = st.tabs(["➕ Autoriser un Utilisateur", "🗑️ Retirer un Utilisateur"])
            with admin_tab1:
                user_id_to_add = st.text_input("Identifiant de l'utilisateur à autoriser", key="admin_add_user")
                if st.button("Autoriser l'Utilisateur"):
                    if user_id_to_add:
                        authorized_users = load_authorized_users()
                        if user_id_to_add in authorized_users:
                            st.warning(f"L'utilisateur `{user_id_to_add}` est déjà autorisé.")
                        else:
                            authorized_users.add(user_id_to_add)
                            save_authorized_users(authorized_users)
                            st.success(f"Utilisateur `{user_id_to_add}` **autorisé avec succès** !")
                            if user_id_to_add not in load_registered_users():
                                st.info(f"L'utilisateur `{user_id_to_add}` n'existe pas encore.")
                    else:
                        st.warning("Veuillez entrer un identifiant.")
            with admin_tab2:
                user_id_to_remove = st.text_input("Identifiant de l'utilisateur à retirer", key="admin_remove_user")
                if st.button("Retirer l'Utilisateur"):
                    if user_id_to_remove:
                        authorized_users = load_authorized_users()
                        if user_id_to_remove in authorized_users:
                            authorized_users.remove(user_id_to_remove)
                            save_authorized_users(authorized_users)
                            st.success(f"Utilisateur `{user_id_to_remove}` retiré avec succès.")
                            if 'username' in st.session_state and st.session_state['username'] == user_id_to_remove:
                                st.session_state['logged_in'] = False
                                if 'username' in st.session_state: del st.session_state['username']
                                st.info("Votre accès a été révoqué.")
                        else:
                            st.warning("ID utilisateur non trouvé.")
                    else:
                        st.warning("Veuillez entrer un ID utilisateur.")
            if st.button("👀 Voir la liste des utilisateurs autorisés"):
                authorized_users = load_authorized_users()
                if authorized_users:
                    st.text_area("Liste des ID", value="\n".join(sorted(authorized_users)), height=150)
                else:
                    st.info("Aucun utilisateur autorisé.")
        elif admin_password:
            st.error("Mot de passe administrateur incorrect.")

# --- VÉRIFICATION D'ACCÈS ---
if not is_user_authorized():
    login_page()
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        if 'username' in st.session_state:
            st.markdown(f"Bonjour, **{st.session_state['username']}** !")
        if st.button("Se Déconnecter"):
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('logged_in') or k.startswith('user_')]
            for k in keys_to_delete:
                del st.session_state[k]
            st.success("Vous avez été déconnecté.")
            st.rerun()
    st.stop()

# --- PAGE PRINCIPALE ---
if 'username' in st.session_state:
    st.set_page_config(page_title=f"Visualisation Foot - {st.session_state['username']}", layout="wide")
    st.title(f"⚽ Outil de Visualisation de Données Footballistiques - Bienvenue, {st.session_state['username']} !")
else:
    st.set_page_config(page_title="Visualisation Foot", layout="wide")
    st.title("Outil de Visualisation de Données Footballistiques")

# --- UPLOAD CSV ---
st.sidebar.header("📁 Données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])
if not uploaded_file:
    st.warning("Veuillez importer un fichier CSV.")
    st.stop()

# --- LECTURE DU CSV ---
with st.spinner("Lecture du fichier CSV..."):
    try:
        content = uploaded_file.read().decode('utf-8')
        sep = ';' if ';' in content.splitlines()[0] else ','
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du CSV : {e}")
        st.stop()

# --- APERÇU DES DONNÉES ---
with st.expander("🔍 Aperçu des données importées"):
    st.write("**Structure du DataFrame :**")
    st.write(df.dtypes)
    st.write("**Premières lignes :**")
    st.dataframe(df.head())

# --- VÉRIFICATION DES COLONNES ---
required_columns = ['Team', 'Player', 'Event', 'X', 'Y', 'X2', 'Y2']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Colonnes manquantes dans le fichier CSV : {missing_columns}")
    st.stop()
df = df[required_columns]
df = df.dropna(subset=['Player', 'Event', 'X', 'Y']).reset_index(drop=True)

# --- CONVERSION DES COORDONNÉES ---
for col in ['X', 'Y', 'X2', 'Y2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

max_coord = df[['X', 'Y', 'X2', 'Y2']].max().max()
if pd.notna(max_coord) and max_coord <= 105 and max_coord > 50:
    st.info("Coordonnées détectées en format 0-100. Conversion vers 0-120/0-80.")
    df['X'] *= 1.2
    df['X2'] *= 1.2
    df['Y'] *= 0.8
    df['Y2'] *= 0.8

# --- NETTOYAGE DES TEXTES ---
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

# --- ✅ CORRECTION MAJEURE : CLASSIFICATION DES ZONES (LOGIQUE FOOTBALLISTIQUE) ---
def classify_zone(x, y):
    """
    Classification CORRECTE :
    - Surface Rép. : x > 102 et 18 < y < 62 → dans la surface adverse
    - Haute        : 80 < x <= 102 → tiers offensif (proche surface adverse)
    - Médiane      : 40 <= x <= 80 → milieu
    - Basse        : x < 40 → tiers défensif (proche de sa propre surface)
    """
    if x > 102 and 18 < y < 62:
        return 'Surface Rép.'
    elif x > 80:
        return 'Haute'
    elif x >= 40:
        return 'Médiane'
    else:
        return 'Basse'

# --- Ajout temporaire pour le filtre ---
df['Zone_temp'] = df.apply(lambda row: classify_zone(row['X'], row['Y']), axis=1)
zone_options = sorted(df['Zone_temp'].dropna().unique())
del df['Zone_temp']

selected_zones = st.sidebar.multiselect(
    "Zones du Terrain",
    options=zone_options,
    default=zone_options,
    help="Filtrer les événements par zone où ils ont eu lieu."
)

# --- OPTIONS D'AFFICHAGE ---
st.sidebar.header("⚙️ Options d'Affichage")
show_legend = st.sidebar.checkbox("Afficher la légende des événements", value=True)

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
display_names_list = list(PALETTE_OPTIONS.keys())

with st.sidebar.expander("➕ Options Avancées"):
    arrow_width = st.slider("Épaisseur des flèches", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="arrow_width")
    arrow_head_scale = st.slider("Taille de la tête des flèches", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="arrow_head_scale")
    arrow_alpha = st.slider("Opacité des flèches", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="arrow_alpha")
    point_size = st.slider("Taille des points", min_value=20, max_value=200, value=80, step=10, key="point_size")
    scatter_alpha = st.slider("Opacité des points", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="scatter_alpha")
    heatmap_alpha = st.slider("Opacité de la heatmap", min_value=0.1, max_value=1.0, value=0.85, step=0.05, key="heatmap_alpha")
    heatmap_statistic = st.selectbox("Type de statistique", options=['count', 'density'], index=0)
    show_heatmap_labels = st.checkbox("Afficher les labels sur la heatmap", value=True, key="show_heatmap_labels")
    hide_zero_percent_labels = st.checkbox("Masquer les labels 0%", value=True)
    selected_palette_display_name = st.selectbox("Palette de couleurs", options=display_names_list, index=0)
    color_palette_name = PALETTE_OPTIONS[selected_palette_display_name]

# --- APPLICATION DES FILTRES ---
df_filtered = df[
    df['Team'].isin(selected_teams) &
    df['Player'].isin(selected_players) &
    df['Event'].isin(displayed_events)
]
df_filtered = df_filtered[
    df_filtered.apply(lambda row: classify_zone(row['X'], row['Y']), axis=1).isin(selected_zones)
]
df_event = df_filtered.copy()

if df_event.empty:
    st.warning("Aucun événement ne correspond aux filtres sélectionnés.")
    st.stop()

df_event['Zone'] = df_event.apply(lambda row: classify_zone(row['X'], row['Y']), axis=1)

# --- TABLEAUX STATISTIQUES ---
st.header("Quantité d'Événements par Type et Zone")
zone_counts = df_event.groupby(['Event', 'Zone']).size().unstack(fill_value=0)
zone_counts['Total'] = zone_counts.sum(axis=1)
zone_counts = zone_counts.sort_values(by='Total', ascending=False)
st.dataframe(zone_counts)

st.subheader("Pourcentage d'Événements par Zone")
zone_total = df_event['Zone'].value_counts().reset_index()
zone_total.columns = ['Zone', 'Total']
total_events = zone_total['Total'].sum()
zone_total['Pourcentage'] = (zone_total['Total'] / total_events * 100).round(1)
st.dataframe(zone_total.style.background_gradient(cmap='Reds', subset=['Pourcentage']).format({"Pourcentage": "{:.1f}%"}))

# --- ✅ CORRECTION : VISUALISATIONS PAR ZONES ---
st.markdown("---")
st.header("Visualisations sur Terrain - Analyse par Zones")

common_pitch_params = {
    'pitch_color': 'white',
    'line_color': 'black',
    'linewidth': 1,
    'line_zorder': 2
}
fig_size = (8, 5.5)

# ✅ RECTANGLES CORRIGÉS : Haute = offensive (x élevé)
zones_rects = {
    'Haute': (80, 0, 22, 80),       # x=80 à 102
    'Médiane': (40, 0, 40, 80),     # x=40 à 80
    'Basse': (0, 0, 40, 80),        # x=0 à 40
    'Surface Rép.': (102, 18, 18, 44)  # x=102 à 120, y=18 à 62
}

zone_colors = {
    'Haute': '#FFD700',          # Or → offensive
    'Médiane': '#98FB98',        # Vert → neutre
    'Basse': '#87CEEB',          # Bleu → défensive
    'Surface Rép.': '#FF6347'    # Rouge → danger
}

col_a, col_b = st.columns(2)

with col_a:
    pitch_zone = Pitch(**common_pitch_params)
    fig_zone, ax_zone = pitch_zone.draw(figsize=fig_size)
    fig_zone.set_facecolor('white')
    zone_percents = df_event['Zone'].value_counts(normalize=True).to_dict()
    for zone, (x, y, w, h) in zones_rects.items():
        percent = zone_percents.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=zone_colors.get(zone, '#DDDDDD'), alpha=0.7)
        ax_zone.add_patch(rect)
        ax_zone.text(x + w/2, y + h/2, f"{zone}\n{percent*100:.1f}%", ha='center', va='center', fontsize=8, weight='bold')
    ax_zone.set_title("Répartition en Pourcentages", fontsize=12, weight='bold', pad=10)
    st.pyplot(fig_zone)

with col_b:
    pitch_count = Pitch(**common_pitch_params)
    fig_count, ax_count = pitch_count.draw(figsize=fig_size)
    fig_count.set_facecolor('white')
    zone_counts_dict = df_event['Zone'].value_counts().to_dict()
    for zone, (x, y, w, h) in zones_rects.items():
        count = zone_counts_dict.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=zone_colors.get(zone, '#DDDDDD'), alpha=0.7)
        ax_count.add_patch(rect)
        ax_count.text(x + w/2, y + h/2, f"{zone}\n{count} evt", ha='center', va='center', fontsize=8, weight='bold')
    ax_count.set_title("Nombre d'Événements", fontsize=12, weight='bold', pad=10)
    st.pyplot(fig_count)

# --- VISUALISATIONS PRINCIPALES (inchangées) ---
st.markdown("---")
st.subheader("Visualisations sur Terrain")

base_colors = {
    'Shot': '#FF4B4B', 'Pass': '#6C9AC3', 'Dribble': '#FFA500',
    'Cross': '#92c952', 'Tackle': '#A52A2A', 'Interception': '#FFD700',
    'Clearance': '#00CED1'
}

def get_event_colors(event_list, palette_name, base_colors_dict):
    if palette_name == 'Par défaut':
        cmap_for_others = cm.get_cmap('tab20', max(1, len(event_list)))
        generated_colors = {event: mcolors.to_hex(cmap_for_others(i)) for i, event in enumerate([e for e in event_list if e not in base_colors_dict])}
        return {**base_colors_dict, **generated_colors}
    else:
        try:
            cmap_selected = cm.get_cmap(palette_name, max(1, len(event_list)))
            return {event: mcolors.to_hex(cmap_selected(i)) for i, event in enumerate(event_list)}
        except ValueError:
            cmap_fallback = cm.get_cmap('tab20', max(1, len(event_list)))
            return {event: mcolors.to_hex(cmap_fallback(i)) for i, event in enumerate(event_list)}

event_colors = get_event_colors(event_options, color_palette_name, base_colors)

col1, col2 = st.columns(2)

with col1:
    with st.spinner("Génération de la visualisation des événements..."):
        pitch = Pitch(pitch_color='white', line_color='black', linewidth=1)
        fig1, ax1 = pitch.draw(figsize=(10, 6))
        legend_elements = []
        for event_type in displayed_events:
            event_data = df_event[df_event['Event'] == event_type]
            color = event_colors.get(event_type, '#333333')
            has_xy2 = event_data[['X2', 'Y2']].notna().all(axis=1)
            if has_xy2.any():
                pitch.arrows(
                    event_data[has_xy2]['X'], event_data[has_xy2]['Y'],
                    event_data[has_xy2]['X2'], event_data[has_xy2]['Y2'],
                    color=color, width=arrow_width, headwidth=3 * arrow_head_scale,
                    headlength=2 * arrow_head_scale, alpha=arrow_alpha, ax=ax1
                )
            if (~has_xy2).any():
                pitch.scatter(
                    event_data[~has_xy2]['X'], event_data[~has_xy2]['Y'],
                    ax=ax1, fc=color, ec='black', lw=0.5, s=point_size, alpha=scatter_alpha
                )
            if show_legend:
                legend_elements.append(Patch(facecolor=color, label=event_type))
        ax1.set_title("Visualisation des Événements", fontsize=12, weight='bold')
        fig1.set_facecolor('white')
        if show_legend and legend_elements:
            fig1.set_size_inches(12, 6)
            ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
            plt.subplots_adjust(right=0.82)
        else:
            plt.tight_layout()
        st.pyplot(fig1)

        # Légende dans la sidebar
        if show_legend:
            with st.sidebar:
                st.markdown("### 🎨 Légende des événements")
                for event in displayed_events:
                    color = event_colors.get(event, '#333333')
                    st.markdown(f"<span style='color:{color}; font-weight:bold;'>●</span> {event}", unsafe_allow_html=True)

with col2:
    with st.spinner("Génération de la heatmap..."):
        pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2)
        fig2, ax2 = pitch.draw(figsize=(10, 6))
        fig2.set_facecolor('white')
        df_hm = df_event if len(displayed_events) != 1 else df_event[df_event['Event'] == displayed_events[0]]
        if not df_hm.empty:
            bin_statistic = pitch.bin_statistic(df_hm['X'], df_hm['Y'], statistic=heatmap_statistic, bins=(6, 5), normalize=True)
            pitch.heatmap(bin_statistic, ax=ax2, cmap='Reds', edgecolor='white', alpha=heatmap_alpha)
            if show_heatmap_labels:
                pitch.label_heatmap(
                    bin_statistic, ax=ax2, str_format='{:.0%}', fontsize=12,
                    ha='center', va='center', exclude_zeros=hide_zero_percent_labels, color='black'
                )
        ax2.set_title("Heatmap des Événements", fontsize=12, weight='bold')
        st.pyplot(fig2)

st.markdown("---")

# --- CARTES COMBINÉES ET PDF (inchangés) ---
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
                    pitch.arrows(
                        arrows_data['X'], arrows_data['Y'],
                        arrows_data['X2'], arrows_data['Y2'],
                        ax=ax, zorder=10, color=color, alpha=arrow_alpha, width=arrow_width,
                        headwidth=3 * arrow_head_scale, headlength=2 * arrow_head_scale
                    )
                points_data = df_type[df_type[['X2', 'Y2']].isna().any(axis=1)]
                if not points_data.empty:
                    pitch.scatter(
                        points_data['X'], points_data['Y'], ax=ax,
                        fc=color, marker='o', s=point_size, ec='black', lw=1,
                        alpha=scatter_alpha, zorder=5
                    )
                bin_stat = pitch.bin_statistic(df_type['X'], df_type['Y'], bins=(6, 5), normalize=True)
                event_cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'YlGnBu', 'PuRd']
                cmap_name = event_cmaps[i % len(event_cmaps)]
                pitch.heatmap(bin_stat, ax=ax, cmap=cmap_name, edgecolor='white', alpha=heatmap_alpha)
                if show_heatmap_labels:
                    pitch.label_heatmap(
                        bin_stat, ax=ax, str_format='{:.0%}', fontsize=10,
                        ha='center', va='center', exclude_zeros=hide_zero_percent_labels, color='black'
                    )
                ax.set_title(event_type, color='black', fontsize=12, weight='bold')
                fig.set_facecolor('white')
                row[i].pyplot(fig)
                with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    combined_images.append((event_type, tmpfile.name))
                plt.close(fig)

# --- TÉLÉCHARGEMENT PDF ---
st.sidebar.markdown("---")
if st.sidebar.button("📥 Télécharger le rapport PDF complet"):
    with st.spinner("Génération du rapport PDF..."):
        pdf = FPDF(orientation='L', unit='mm', format='A3')
        pdf.set_auto_page_break(auto=False)

        def add_footer():
            pdf.set_y(-10)
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 5, f"Page {pdf.page_no()}", 0, 0, 'C')

        temp_files = []
        try:
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

            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 12, "Analyse par Zones du Terrain", ln=True, align='C')
            pdf.ln(8)

            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_zone_pct:
                fig_zone.savefig(tmp_zone_pct.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_zone_pct.name)
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_zone_count:
                fig_count.savefig(tmp_zone_count.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_zone_count.name)

            terrain_y = 60
            pdf.image(tmp_zone_pct.name, x=50, y=terrain_y, w=140, h=90)
            pdf.image(tmp_zone_count.name, x=220, y=terrain_y, w=140, h=90)
            pdf.set_xy(50, terrain_y + 95)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(140, 6, "Répartition en Pourcentages", align='C')
            pdf.set_xy(220, terrain_y + 95)
            pdf.cell(140, 6, "Nombre d'Événements", align='C')
            add_footer()

            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 12, "Visualisations sur Terrain", ln=True, align='C')
            pdf.ln(10)

            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_fig1:
                fig1.savefig(tmp_fig1.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_fig1.name)
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp_fig2:
                fig2.savefig(tmp_fig2.name, bbox_inches='tight', dpi=200, facecolor='white')
                temp_files.append(tmp_fig2.name)

            terrain_width = 180
            terrain_height = 120
            pdf.image(tmp_fig1.name, x=30, y=40, w=terrain_width, h=terrain_height)
            pdf.image(tmp_fig2.name, x=230, y=40, w=terrain_width, h=terrain_height)
            pdf.set_xy(30, 165)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(terrain_width, 8, "Événements sur le terrain", align='C')
            pdf.set_xy(230, 165)
            pdf.cell(terrain_width, 8, "Heatmap des événements", align='C')
            add_footer()

            if combined_images:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18)
                pdf.cell(0, 12, "Cartes Combinées par Type d'Événement", ln=True, align='C')
                pdf.ln(10)
                img_width = 120
                img_height = 80
                margin_x = 30
                margin_y = 40
                cols = 3
                spacing_x = 15
                spacing_y = 20
                for i, (event_type, img_path) in enumerate(combined_images[:6]):
                    col = i % cols
                    row = i // cols
                    x = margin_x + col * (img_width + spacing_x)
                    y = margin_y + row * (img_height + spacing_y)
                    pdf.image(img_path, x=x, y=y, w=img_width, h=img_height)
                    pdf.set_xy(x, y + img_height + 3)
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(img_width, 6, event_type, align='C')
                add_footer()

            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdf.output(tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as file:
                    st.download_button(
                        "📄 Télécharger le PDF final du rapport",
                        data=file.read(),
                        file_name="rapport_foot_A3.pdf",
                        mime="application/pdf"
                    )
                os.unlink(tmp_pdf.name)

        finally:
            for f in temp_files + [img for _, img in combined_images]:
                try:
                    os.unlink(f)
                except:
                    pass
