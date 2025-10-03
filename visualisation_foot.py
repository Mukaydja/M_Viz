# am-fcp.py
# --- IMPORTS (ajouts pour l'authentification) ---
import uuid
import json
import time
import os
import hashlib
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
from matplotlib.patches import Patch
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import base64

# === PORTAIL D'ACCÈS OBLIGATOIRE AVEC VÉRIFICATION D'EMAIL (MX + OTP) ===
from datetime import datetime, timedelta
import smtplib, ssl, random, string
try:
    import dns.resolver  # pip install dnspython
except Exception:
    dns = None  # si non dispo, on gérera proprement

CONTACTS_PATH = os.path.join("data", "contacts.csv")
os.makedirs(os.path.dirname(CONTACTS_PATH), exist_ok=True)

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
OTP_TTL_MINUTES = 10
OTP_LENGTH = 6

def load_contacts():
    if os.path.exists(CONTACTS_PATH):
        try:
            return pd.read_csv(CONTACTS_PATH, dtype=str)
        except Exception:
            return pd.DataFrame(columns=["id","nom","prenom","email","email_sha256","created_at"])
    else:
        return pd.DataFrame(columns=["id","nom","prenom","email","email_sha256","created_at"])

def save_contact(nom: str, prenom: str, email: str):
    dfc = load_contacts()
    email_norm = email.strip().lower()
    email_hash = hashlib.sha256(email_norm.encode()).hexdigest()

    # doublon par email (insensible à la casse)
    if not dfc[dfc["email"].str.lower() == email_norm].empty:
        return True, "Bienvenue à nouveau 👋", email_hash

    new_row = {
        "id": str(uuid.uuid4()),
        "nom": nom.strip(),
        "prenom": prenom.strip(),
        "email": email_norm,                   # supprimez cette colonne si vous ne voulez garder que le hash
        "email_sha256": email_hash,            # identifiant non réversible
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    dfc = pd.concat([dfc, pd.DataFrame([new_row])], ignore_index=True)
    dfc.to_csv(CONTACTS_PATH, index=False)
    return True, "Coordonnées enregistrées ✅", email_hash

def has_mx_record(email: str) -> bool:
    """Vérifie qu'un enregistrement MX existe pour le domaine."""
    try:
        domain = email.split("@", 1)[1].strip()
        if not domain:
            return False
        if dns is None:
            # dnspython absent : on ne bloque pas, mais on conseille de l'installer
            return True
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:
        return False

def gen_otp(n: int = OTP_LENGTH) -> str:
    return "".join(random.choices(string.digits, k=n))

def send_otp_email(to_email: str, otp: str) -> tuple[bool, str]:
    """
    Envoie un OTP par SMTP (config via st.secrets). Retourne (ok, message).
    Secrets attendus :
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, APP_NAME (facultatif)
    """
    host = st.secrets.get("SMTP_HOST")
    port = int(st.secrets.get("SMTP_PORT", 465))
    user = st.secrets.get("SMTP_USER")
    pwd  = st.secrets.get("SMTP_PASS")
    from_addr = st.secrets.get("SMTP_FROM", user)
    app_name = st.secrets.get("APP_NAME", "Votre application")

    if not all([host, port, user, pwd, from_addr]):
        return False, "Configuration SMTP manquante. Définissez SMTP_* dans st.secrets."

    subject = f"[{app_name}] Votre code de vérification"
    body = (
        f"Bonjour,\n\n"
        f"Voici votre code de vérification : {otp}\n"
        f"Il expire dans {OTP_TTL_MINUTES} minutes.\n\n"
        f"--\n{app_name}"
    )
    msg = f"From: {from_addr}\r\nTo: {to_email}\r\nSubject: {subject}\r\n\r\n{body}"

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(user, pwd)
            server.sendmail(from_addr, [to_email], msg.encode("utf-8"))
        return True, "Un code vous a été envoyé par email."
    except Exception as e:
        return False, f"Échec d'envoi du code : {e}"

def start_otp_flow(nom: str, prenom: str, email: str):
    """Démarre le flux OTP (génère, envoie, stocke en session)."""
    otp = gen_otp()
    st.session_state["pending_gate"] = {
        "nom": nom.strip(),
        "prenom": prenom.strip(),
        "email": email.strip().lower(),
        "otp": otp,
        "otp_expires_at": (datetime.utcnow() + timedelta(minutes=OTP_TTL_MINUTES)).isoformat(),
        "tries": 0
    }
    ok, msg = send_otp_email(email.strip(), otp)
    if ok:
        st.info(msg)
    else:
        st.error(msg)

def render_otp_form():
    """Affiche le formulaire de saisie OTP si un OTP est en attente."""
    if "pending_gate" not in st.session_state:
        return False

    pending = st.session_state["pending_gate"]
    expires_at = datetime.fromisoformat(pending["otp_expires_at"])
    remaining = int((expires_at - datetime.utcnow()).total_seconds() // 60)

    st.markdown("### ✉️ Vérification de votre email")
    st.caption(f"Un code a été envoyé à **{pending['email']}**. Il expirera dans {remaining} minute(s).")

    with st.form("otp_form", clear_on_submit=False):
        otp_input = st.text_input("Code de vérification (6 chiffres) *", max_chars=OTP_LENGTH)
        c1, c2 = st.columns([1,1])
        with c1:
            validate = st.form_submit_button("Valider")
        with c2:
            resend = st.form_submit_button("Renvoyer le code")

    if resend:
        # régénérer et renvoyer un nouveau code
        start_otp_flow(pending["nom"], pending["prenom"], pending["email"])
        st.stop()

    if validate:
        if datetime.utcnow() > expires_at:
            st.error("Code expiré. Un nouveau code vous a été envoyé.")
            start_otp_flow(pending["nom"], pending["prenom"], pending["email"])
            st.stop()

        if otp_input and otp_input.strip() == pending["otp"]:
            # Vérification réussie -> on enregistre l'utilisateur et on ouvre l'accès
            ok, msg, email_hash = save_contact(pending["nom"], pending["prenom"], pending["email"])
            if ok:
                st.session_state["gate_passed"] = True
                st.session_state["user_email_hash"] = email_hash
                st.session_state["username"] = f"{pending['prenom'].title()} {pending['nom'].upper()}"
                # nettoyer l'OTP de la session
                del st.session_state["pending_gate"]
                st.success("Email vérifié ✅")
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
                st.stop()
        else:
            pending["tries"] += 1
            st.error("Code invalide.")
            if pending["tries"] >= 5:
                st.warning("Trop d'essais. Un nouveau code a été envoyé.")
                start_otp_flow(pending["nom"], pending["prenom"], pending["email"])
            st.stop()

    return True

def require_user_gate():
    # Si un OTP est en cours, afficher le formulaire OTP
    if render_otp_form():
        return

    st.markdown("## 🔐 Accès")
    with st.container():
        with st.form("gate_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("Nom *")
            with col2:
                prenom = st.text_input("Prénom *")
            email = st.text_input("Email *", placeholder="ex: nom@domaine.com")

            consent = st.checkbox(
                "J’accepte que mes informations (nom, prénom, email) soient utilisées pour personnaliser l’application.",
                value=True
            )

            submitted = st.form_submit_button("Recevoir un code et continuer")
            if submitted:
                if not nom or not prenom or not email:
                    st.error("Merci de remplir tous les champs obligatoires (*)")
                elif not re.match(EMAIL_REGEX, email.strip()):
                    st.error("Adresse email invalide.")
                elif not consent:
                    st.warning("Vous devez accepter pour continuer.")
                elif not has_mx_record(email.strip()):
                    st.error("Le domaine de l'email ne semble pas exister (MX introuvable).")
                else:
                    # L'email ressemble à un vrai + MX OK => démarrer l'envoi du code
                    start_otp_flow(nom, prenom, email)
                    st.stop()

# --- Exiger l'identification avant tout le contenu ---
if not st.session_state.get("gate_passed"):
    require_user_gate()
    st.stop()

# === (Optionnel) EXPORT ADMIN SEULEMENT ===
"""
with st.sidebar.expander("🔑 Zone admin", expanded=False):
    admin_key = st.text_input("Admin key", type="password")
    expected = st.secrets.get("ADMIN_KEY", None)
    if expected and admin_key == expected:
        dfc = load_contacts()
        csv_bytes = dfc.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Télécharger contacts.csv", data=csv_bytes, file_name="contacts.csv", mime="text/csv")
        st.caption(f"{len(dfc)} contact(s) enregistrés.")
    elif admin_key:
        st.error("Clé admin invalide.")
"""
# === PORTAIL SIMPLE : FORMAT + MX + ANTI-DOMAINE JETABLE (SANS ENVOI D'EMAIL) ===
from datetime import datetime
try:
    import dns.resolver  # facultatif : ajoute 'dnspython>=2.6' dans requirements.txt pour activer la vérification MX
except Exception:
    dns = None

# Emplacement du fichier de contacts (privé)
CONTACTS_PATH = os.path.join("data", "contacts.csv")
os.makedirs(os.path.dirname(CONTACTS_PATH), exist_ok=True)

# Validation de base du format email
EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

# Liste succincte de domaines jetables courants (tu peux en ajouter au besoin)
DISPOSABLE_DOMAINS = {
    "mailinator.com","10minutemail.com","10minutemail.net","10minutemail.co.uk",
    "tempmail.com","tempmailo.com","tempmailaddress.com","guerrillamail.com",
    "yopmail.com","trashmail.com","getnada.com","sharklasers.com",
    "maildrop.cc","moakt.com","throwawaymail.com","dispostable.com"
}

def load_contacts() -> pd.DataFrame:
    """Charge le CSV de contacts (ou initialise un DF propre)."""
    cols = ["id","nom","prenom","email","email_sha256","created_at"]
    if os.path.exists(CONTACTS_PATH):
        try:
            dfc = pd.read_csv(CONTACTS_PATH, dtype=str)
            # s'assure des colonnes
            for c in cols:
                if c not in dfc.columns:
                    dfc[c] = ""
            return dfc[cols]
        except Exception:
            return pd.DataFrame(columns=cols)
    else:
        return pd.DataFrame(columns=cols)

def save_contact(nom: str, prenom: str, email: str):
    """Ajoute l'utilisateur si nouveau (doublon sur email), sinon laisse passer."""
    dfc = load_contacts()
    email_norm = email.strip().lower()
    email_hash = hashlib.sha256(email_norm.encode()).hexdigest()

    if "email" in dfc.columns and not dfc[dfc["email"].str.lower() == email_norm].empty:
        return True, "Bienvenue à nouveau 👋", email_hash

    new_row = {
        "id": str(uuid.uuid4()),
        "nom": nom.strip(),
        "prenom": prenom.strip(),
        "email": email_norm,            # supprime cette clé si tu ne veux jamais stocker l'email en clair
        "email_sha256": email_hash,     # identifiant non réversible
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    dfc = pd.concat([dfc, pd.DataFrame([new_row])], ignore_index=True)
    dfc.to_csv(CONTACTS_PATH, index=False)
    return True, "Coordonnées enregistrées ✅", email_hash

def has_mx_record(email: str) -> bool:
    """Vérifie qu'un enregistrement MX existe pour le domaine de l'email."""
    try:
        domain = email.split("@", 1)[1].strip().lower()
        if not domain:
            return False
        if dns is None:
            # Si dnspython n'est pas installé, on ne bloque pas (tu peux durcir en retournant False ici si tu préfères)
            return True
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:
        return False

def require_user_gate():
    """Affiche le portail et bloque l'app tant que l'utilisateur n'est pas identifié."""
    st.markdown("## 🔐 Accès")
    with st.container():
        with st.form("gate_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("Nom *")
            with col2:
                prenom = st.text_input("Prénom *")
            email = st.text_input("Email *", placeholder="ex: nom@domaine.com")

            submitted = st.form_submit_button("Continuer")
            if submitted:
                # 1) champs requis
                if not nom or not prenom or not email:
                    st.error("Merci de remplir tous les champs obligatoires (*)")
                    st.stop()

                # 2) format email
                email_norm = email.strip().lower()
                if not re.match(EMAIL_REGEX, email_norm):
                    st.error("Adresse email invalide.")
                    st.stop()

                # 3) anti-domaine jetable
                domain = email_norm.split("@", 1)[1]
                if domain in DISPOSABLE_DOMAINS:
                    st.error("Les emails jetables ne sont pas autorisés.")
                    st.stop()

                # 4) MX du domaine (si dnspython installé)
                if not has_mx_record(email_norm):
                    st.error("Le domaine de l'email ne semble pas exister (MX introuvable).")
                    st.stop()

                # 5) OK → enregistrement + accès
                ok, msg, email_hash = save_contact(nom, prenom, email_norm)
                if ok:
                    st.session_state["gate_passed"] = True
                    st.session_state["user_email_hash"] = email_hash
                    st.session_state["username"] = f"{prenom.strip().title()} {nom.strip().upper()}"
                    st.success(msg)
                    try:
                        st.rerun()  # Streamlit récents
                    except AttributeError:
                        st.experimental_rerun()  # fallback anciennes versions
                    st.stop()

# 🔒 Bloquer l'accès tant que non identifié
if not st.session_state.get("gate_passed"):
    require_user_gate()
    st.stop()

# (Optionnel) Zone admin d'export CSV — à décommenter si tu veux te garder une porte d'export privée
"""
with st.sidebar.expander("🔑 Zone admin (privée)", expanded=False):
    admin_key = st.text_input("Admin key", type="password")
    expected = st.secrets.get("ADMIN_KEY", None)  # définis ADMIN_KEY dans .streamlit/secrets.toml
    if expected and admin_key == expected:
        dfc = load_contacts()
        st.download_button(
            "Télécharger contacts.csv",
            data=dfc.to_csv(index=False).encode("utf-8-sig"),
            file_name="contacts.csv",
            mime="text/csv"
        )
        st.caption(f"{len(dfc)} contact(s) enregistrés.")
    elif admin_key:
        st.error("Clé admin invalide.")
"""

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

# --- ✅ CORRECTION : CLASSIFICATION EN 3 ZONES (Haute/Basse inversées) ---
def classify_zone(x, y):
    """
    Classification en 3 zones :
    - Haute : x < 40 (désormais zone côté gauche/défensive)
    - Médiane : 40 <= x <= 80
    - Basse : x > 80 (désormais zone côté droit/offensive)
    """
    if x < 40:
        return 'Haute'
    elif x <= 80:
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
    arrow_width = st.slider("Épaisseur des flèches", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    arrow_head_scale = st.slider("Taille de la tête des flèches", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    arrow_alpha = st.slider("Opacité des flèches", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    point_size = st.slider("Taille des points", min_value=20, max_value=200, value=80, step=10)
    scatter_alpha = st.slider("Opacité des points", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    heatmap_alpha = st.slider("Opacité de la heatmap", min_value=0.1, max_value=1.0, value=0.85, step=0.05)
    heatmap_statistic = st.selectbox("Type de statistique", options=['count', 'density'], index=0)
    show_heatmap_labels = st.checkbox("Afficher les labels sur la heatmap", value=True)
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

# --- VISUALISATIONS PAR ZONES (3 ZONES SEULEMENT) ---
st.markdown("---")
st.header("Visualisations sur Terrain - Analyse par Zones")

common_pitch_params = {
    'pitch_color': 'white',
    'line_color': 'black',
    'linewidth': 1,
    'line_zorder': 2
}
fig_size = (8, 5.5)

# ✅ RECTANGLES CORRIGÉS : 3 zones, Haute/Basse inversées pour correspondre à la classification
zones_rects = {
    'Haute': (0, 0, 40, 80),        # x=0 à 40   (désormais Haute)
    'Médiane': (40, 0, 40, 80),     # x=40 à 80
    'Basse': (80, 0, 40, 80)        # x=80 à 120 (désormais Basse)
}

zone_colors = {
    'Haute': '#FFD700',          # Or
    'Médiane': '#98FB98',        # Vert
    'Basse': '#87CEEB'           # Bleu
}

col_a, col_b = st.columns(2)

with col_a:
    pitch_zone = Pitch(**common_pitch_params)
    fig_zone, ax_zone = pitch_zone.draw(figsize=fig_size)
    fig_zone.set_facecolor('white')
    zone_percents = df_event['Zone'].value_counts(normalize=True).to_dict()
    for zone, (x, y, w, h) in zones_rects.items():
        percent = zone_percents.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black',
                             facecolor=zone_colors.get(zone, '#DDDDDD'), alpha=0.7)
        ax_zone.add_patch(rect)
        ax_zone.text(x + w/2, y + h/2, f"{zone}\n{percent*100:.1f}%", ha='center', va='center',
                     fontsize=8, weight='bold')
    ax_zone.set_title("Répartition en Pourcentages", fontsize=12, weight='bold', pad=10)
    st.pyplot(fig_zone)

with col_b:
    pitch_count = Pitch(**common_pitch_params)
    fig_count, ax_count = pitch_count.draw(figsize=fig_size)
    fig_count.set_facecolor('white')
    zone_counts_dict = df_event['Zone'].value_counts().to_dict()
    for zone, (x, y, w, h) in zones_rects.items():
        count = zone_counts_dict.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black',
                             facecolor=zone_colors.get(zone, '#DDDDDD'), alpha=0.7)
        ax_count.add_patch(rect)
        ax_count.text(x + w/2, y + h/2, f"{zone}\n{count} evt", ha='center', va='center',
                      fontsize=8, weight='bold')
    ax_count.set_title("Nombre d'Événements", fontsize=12, weight='bold', pad=10)
    st.pyplot(fig_count)

# --- VISUALISATIONS PRINCIPALES ---
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
        ax1.set_title("Visualisation des Événements", fontsize=12, weight='bold')
        fig1.set_facecolor('white')
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
            bin_statistic = pitch.bin_statistic(
                df_hm['X'], df_hm['Y'], statistic=heatmap_statistic, bins=(6, 5), normalize=True
            )
            pitch.heatmap(bin_statistic, ax=ax2, cmap='Reds', edgecolor='white', alpha=heatmap_alpha)
            if show_heatmap_labels:
                pitch.label_heatmap(
                    bin_statistic, ax=ax2, str_format='{:.0%}', fontsize=12,
                    ha='center', va='center', exclude_zeros=hide_zero_percent_labels, color='black'
                )
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
