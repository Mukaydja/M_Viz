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

# Visualisation sur terrain avec répartition en zones
col_a, col_b = st.columns(2)

# --- Terrain avec pourcentages ---
with col_a:
    pitch_zone = Pitch(pitch_color='white', line_color='black', linewidth=1)
    fig_zone, ax_zone = pitch_zone.draw(figsize=(8, 5))
    fig_zone.set_facecolor('white')

    zone_percents = df_event['Zone'].value_counts(normalize=True).to_dict()

    zones_rects = {
        'Basse': (0, 0, 36, 80),
        'Médiane': (36, 0, 48, 80),
        'Haute': (84, 0, 36, 80),
        'Surface Rép.': (102, 24, 18, 32)
    }

    colors = {'Basse': '#AED6F1', 'Médiane': '#A9DFBF', 'Haute': '#F9E79F', 'Surface Rép.': '#F5B7B1'}

    for zone, (x, y, w, h) in zones_rects.items():
        percent = zone_percents.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=colors.get(zone, '#DDDDDD'), alpha=0.5)
        ax_zone.add_patch(rect)
        offset = -7 if zone == 'Surface Rép.' else 0
        ax_zone.text(x + w/2, y + h/2 + offset, f"{zone}\n{percent*100:.1f}%", ha='center', va='center', fontsize=10, weight='bold')

    ax_zone.set_title("Répartition en Pourcentages", fontsize=12, weight='bold')
    st.pyplot(fig_zone)

# --- Terrain avec nombre d'événements ---
with col_b:
    pitch_count = Pitch(pitch_color='white', line_color='black', linewidth=1)
    fig_count, ax_count = pitch_count.draw(figsize=(8, 5))
    fig_count.set_facecolor('white')

    zone_counts_dict = df_event['Zone'].value_counts().to_dict()

    for zone, (x, y, w, h) in zones_rects.items():
        count = zone_counts_dict.get(zone, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor=colors.get(zone, '#DDDDDD'), alpha=0.5)
        ax_count.add_patch(rect)
        offset = -7 if zone == 'Surface Rép.' else 0
        ax_count.text(x + w/2, y + h/2 + offset, f"{zone}\n{count} evt", ha='center', va='center', fontsize=10, weight='bold')

    ax_count.set_title("Nombre d'Événements", fontsize=12, weight='bold')
    st.pyplot(fig_count)

# --- VISUALISATION TERRAIN ---
st.subheader("Visualisations sur Terrain")
base_colors = {
    'Shot': '#FF4B4B', 'Pass': '#6C9AC3', 'Dribble': '#FFA500',
    'Cross': '#92c952', 'Tackle': '#A52A2A', 'Interception': '#FFD700',
    'Clearance': '#00CED1'
}
cmap = cm.get_cmap('tab20', len(event_options))
generated_colors = {event: mcolors.to_hex(cmap(i)) for i, event in enumerate(event_options)}
event_colors = {**base_colors, **generated_colors}

col1, col2 = st.columns(2)

with col1:
    pitch = Pitch(pitch_color='white', line_color='black', linewidth=1)
    fig1, ax1 = pitch.draw(figsize=(10, 6))
    for event_type in displayed_events:
        event_data = df_event[df_event['Event'] == event_type]
        color = event_colors.get(event_type, '#333333')
        has_xy2 = event_data[['X2', 'Y2']].notna().all(axis=1)
        pitch.arrows(
            event_data[has_xy2]['X'], event_data[has_xy2]['Y'],
            event_data[has_xy2]['X2'], event_data[has_xy2]['Y2'],
            color=color, width=2, headwidth=4, headlength=6, alpha=0.8, ax=ax1
        )
        pitch.scatter(
            event_data[~has_xy2]['X'], event_data[~has_xy2]['Y'],
            ax=ax1, fc=color, ec='black', lw=0.5, s=80, alpha=0.8
        )
    ax1.set_title("Visualisation des Événements", fontsize=12, weight='bold')
    fig1.set_facecolor('white')
    st.pyplot(fig1)

with col2:
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2)
    fig2, ax2 = pitch.draw(figsize=(10, 6))
    fig2.set_facecolor('white')
    df_filtered_hm = df_event if len(displayed_events) != 1 else df_event[df_event['Event'] == displayed_events[0]]
    if not df_filtered_hm.empty:
        bin_statistic = pitch.bin_statistic(
            df_filtered_hm['X'], df_filtered_hm['Y'], statistic='count', bins=(6, 5), normalize=True
        )
        pitch.heatmap(bin_statistic, ax=ax2, cmap='Reds', edgecolor='white', alpha=0.85)
        path_eff = [patheffects.withStroke(linewidth=0.6, foreground="black")]

        pitch.label_heatmap(
            bin_statistic, str_format='{:.0%}',
            color='black',
            fontsize=12,
            ax=ax2,
            ha='center',
            va='center',
            path_effects=path_eff
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
                        ax=ax, zorder=10, color=color, alpha=0.8, width=2,
                        headwidth=4, headlength=6
                    )

                points_data = df_type[df_type[['X2', 'Y2']].isna().any(axis=1)]
                if not points_data.empty:
                    pitch.scatter(
                        points_data['X'], points_data['Y'], ax=ax,
                        fc=color, marker='o', s=80, ec='black', lw=1,
                        alpha=0.5, zorder=5
                    )

                bin_stat = pitch.bin_statistic(df_type['X'], df_type['Y'], bins=(6, 5), normalize=True)
                event_cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'YlGnBu', 'PuRd']
                cmap_name = event_cmaps[i % len(event_cmaps)]
                pitch.heatmap(bin_stat, ax=ax, cmap=cmap_name, edgecolor='white', alpha=0.6)

                pitch.label_heatmap(
                    bin_stat, str_format='{:.0%}',
                    color='black',
                    fontsize=10,
                    ax=ax,
                    path_effects=[patheffects.withStroke(linewidth=0.6, foreground="black")]
                )

                ax.set_title(event_type, color='black', fontsize=12, weight='bold')
                fig.set_facecolor('white')
                row[i].pyplot(fig)

                # Sauvegarder l'image pour le PDF
                with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    combined_images.append((event_type, tmpfile.name))
                
                plt.close(fig)  # Fermer la figure pour libérer la mémoire

# --- TÉLÉCHARGEMENT DU RAPPORT PDF FINAL ---
if st.button("📥 Télécharger le rapport PDF complet"):
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
