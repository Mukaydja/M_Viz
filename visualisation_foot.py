import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import patheffects
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import os

# --- Configuration de la page ---
st.set_page_config(page_title="Visualisation Foot", layout="wide")
st.title("Outil de Visualisation de Données Footballistiques")

# --- Upload CSV ---
st.sidebar.header("Importer un fichier CSV")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])

if not uploaded_file:
    st.warning("Veuillez importer un fichier CSV.")
    st.stop()

# --- Lecture du CSV ---
try:
    content = uploaded_file.read().decode('utf-8')
    sep = ';' if ';' in content.splitlines()[0] else ','
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=sep)
except Exception as e:
    st.error(f"Erreur lors de la lecture du CSV : {e}")
    st.stop()

# --- Nettoyage des colonnes nécessaires ---
df = df[['Team', 'Player', 'Event', 'X', 'Y', 'X2', 'Y2']]
df = df.dropna(subset=['Player', 'Event', 'X', 'Y']).reset_index(drop=True)

# --- Conversion des colonnes numériques ---
for col in ['X', 'Y', 'X2', 'Y2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Conversion des coordonnées si en 0–100 ---
if df[['X', 'Y', 'X2', 'Y2']].max().max() <= 100:
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
st.sidebar.header("Filtres")
event_options = sorted(df['Event'].dropna().unique())
player_options = sorted(df['Player'].dropna().unique())
team_options = sorted(df['Team'].dropna().unique())

selected_players = st.sidebar.multiselect("Joueurs", player_options, default=player_options)
selected_teams = st.sidebar.multiselect("Équipes", team_options, default=team_options)

st.sidebar.markdown("### Événements")
displayed_events = st.sidebar.multiselect(
    "Événements à afficher",
    options=event_options,
    default=["Pass"] if "Pass" in event_options else event_options[:1]
)

# --- Filtres globaux ---
df_filtered = df[
    df['Team'].isin(selected_teams) &
    df['Player'].isin(selected_players) &
    df['Event'].isin(displayed_events)
]
df_event = df_filtered

# --- Classification par zones de terrain ---
def classify_zone(x, y):
    if x > 102 and 24 < y < 56:
        return 'Surface Rép.'
    elif x > 84:
        return 'Haute'
    elif x < 36:
        return 'Basse'
    else:
        return 'Médiane'

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

# --- Le reste du code est inchangé et peut être copié du bloc initial fourni ---
# Pour alléger la cellule ici, on écrit uniquement la première moitié
