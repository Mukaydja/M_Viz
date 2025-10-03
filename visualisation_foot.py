# am-fcp.py
# --- IMPORTS (ajouts pour l'authentification + cookies) ---
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
from datetime import datetime, timedelta

# --- IMPORT POUR LES COOKIES (connexion persistante) ---
try:
    from streamlit_cookies_manager import CookieManager
    cookies = CookieManager()
    _COOKIES_AVAILABLE = True
except Exception as e:
    st.warning("Cookies non disponibles (streamlit-cookies-manager manquant). Connexion non persistante.")
    _COOKIES_AVAILABLE = False

# =========================================================
# =============== CONFIG GÉNÉRALE & PAGE ==================
# =========================================================
st.set_page_config(page_title="Visualisation Foot", layout="wide")

# Choisis le mode d'authentification : "simple" (par défaut) ou "otp"
AUTH_MODE = "simple"   # "simple" | "otp"

# Fichiers / constantes communes
CONTACTS_PATH = os.path.join("data", "contacts.csv")
os.makedirs(os.path.dirname(CONTACTS_PATH), exist_ok=True)

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
DISPOSABLE_DOMAINS = {
    "mailinator.com","10minutemail.com","10minutemail.net","10minutemail.co.uk",
    "tempmail.com","tempmailo.com","tempmailaddress.com","guerrillamail.com",
    "yopmail.com","trashmail.com","getnada.com","sharklasers.com",
    "maildrop.cc","moakt.com","throwawaymail.com","dispostable.com"
}

# Vérification MX (facultative si dnspython n'est pas installé)
try:
    import dns.resolver
    _DNS_AVAILABLE = True
except Exception:
    _DNS_AVAILABLE = False

def has_mx_record(email: str) -> bool:
    try:
        domain = email.split("@", 1)[1].strip().lower()
        if not domain:
            return False
        if not _DNS_AVAILABLE:
            return True
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:
        return False

def load_contacts() -> pd.DataFrame:
    cols = ["id","nom","prenom","email","email_sha256","created_at"]
    if os.path.exists(CONTACTS_PATH):
        try:
            dfc = pd.read_csv(CONTACTS_PATH, dtype=str)
            for c in cols:
                if c not in dfc.columns:
                    dfc[c] = ""
            return dfc[cols]
        except Exception:
            return pd.DataFrame(columns=cols)
    else:
        return pd.DataFrame(columns=cols)

def save_contact(nom: str, prenom: str, email: str):
    dfc = load_contacts()
    email_norm = email.strip().lower()
    email_hash = hashlib.sha256(email_norm.encode()).hexdigest()
    if "email" in dfc.columns and not dfc[dfc["email"].str.lower() == email_norm].empty:
        return True, "Bienvenue à nouveau 👋", email_hash
    new_row = {
        "id": str(uuid.uuid4()),
        "nom": nom.strip(),
        "prenom": prenom.strip(),
        "email": email_norm,
        "email_sha256": email_hash,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    dfc = pd.concat([dfc, pd.DataFrame([new_row])], ignore_index=True)
    dfc.to_csv(CONTACTS_PATH, index=False)
    return True, "Coordonnées enregistrées ✅", email_hash

# =========================================================
# ================== MODE OTP (inchangé) ==================
# ... (le code OTP reste identique — on le garde tel quel)
# (Je le conserve ici pour complétude, mais tu peux le laisser tel quel)

import socket, ssl, smtplib, json, random, string
try:
    import requests
    _REQ_AVAILABLE = True
except Exception:
    _REQ_AVAILABLE = False

OTP_TTL_MINUTES = 10
OTP_LENGTH = 6

def _gen_otp(n: int = OTP_LENGTH) -> str:
    return "".join(random.choices(string.digits, k=n))

def _resolve_host_or_msg(host: str) -> tuple[bool, str]:
    try:
        socket.getaddrinfo(host, None)
        return True, ""
    except Exception as e:
        return False, f"Impossible de résoudre le nom d’hôte '{host}'. Erreur: {e}"

def send_otp_email(to_email: str, otp: str) -> tuple[bool, str]:
    backend = (st.secrets.get("EMAIL_BACKEND") or "smtp").lower()
    app_name = st.secrets.get("APP_NAME", "Votre application")
    subject = f"[{app_name}] Votre code de vérification"
    body = (
        f"Bonjour,\n\n"
        f"Voici votre code de vérification : {otp}\n"
        f"Il expire dans {OTP_TTL_MINUTES} minutes.\n\n"
        f"--\n{app_name}"
    )

    if backend == "smtp":
        host = st.secrets.get("SMTP_HOST")
        port = int(st.secrets.get("SMTP_PORT", 465))
        user = st.secrets.get("SMTP_USER")
        pwd  = st.secrets.get("SMTP_PASS")
        from_addr = st.secrets.get("SMTP_FROM", user)
        security = (st.secrets.get("SMTP_SECURITY") or "SSL").upper()
        missing = [k for k,v in {"SMTP_HOST":host,"SMTP_PORT":port,"SMTP_USER":user,"SMTP_PASS":pwd}.items() if not v]
        if missing:
            return False, f"Config SMTP incomplète (manque: {', '.join(missing)})."

        ok_res, msg_res = _resolve_host_or_msg(host)
        if not ok_res:
            if st.secrets.get("DEBUG_OTP", False):
                st.warning(f"[DEV] {msg_res} — OTP affiché ci-dessous.")
                st.code(otp)
                return True, "Mode DEV : OTP affiché (SMTP non résolu)."
            return False, msg_res

        msg_bytes = f"From: {from_addr}\r\nTo: {to_email}\r\nSubject: {subject}\r\n\r\n{body}".encode("utf-8")
        try:
            if security == "SSL":
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as server:
                    server.login(user, pwd)
                    server.sendmail(from_addr, [to_email], msg_bytes)
            elif security == "STARTTLS":
                with smtplib.SMTP(host, port, timeout=20) as server:
                    server.ehlo(); server.starttls(context=ssl.create_default_context()); server.ehlo()
                    server.login(user, pwd)
                    server.sendmail(from_addr, [to_email], msg_bytes)
            else:
                with smtplib.SMTP(host, port, timeout=20) as server:
                    server.login(user, pwd)
                    server.sendmail(from_addr, [to_email], msg_bytes)
            return True, "Un code vous a été envoyé par email."
        except Exception as e:
            if st.secrets.get("DEBUG_OTP", False):
                st.warning(f"[DEV] Envoi SMTP impossible ({e}). OTP affiché ci-dessous.")
                st.code(otp)
                return True, "Mode DEV : OTP affiché (SMTP KO)."
            return False, f"Échec d'envoi SMTP : {e}"

    if backend == "sendgrid":
        if not _REQ_AVAILABLE:
            return False, "SendGrid nécessite 'requests'."
        api_key = st.secrets.get("SENDGRID_API_KEY")
        from_addr = st.secrets.get("SMTP_FROM")
        if not api_key or not from_addr:
            return False, "Config SendGrid incomplète."
        url = "https://api.sendgrid.com/v3/mail/send"
        payload = {
            "personalizations": [{"to":[{"email": to_email}], "subject": subject}],
            "from": {"email": from_addr},
            "content": [{"type": "text/plain", "value": body}]
        }
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, data=json.dumps(payload), timeout=20)
            if 200 <= r.status_code < 300:
                return True, "Un code vous a été envoyé par email."
            else:
                if st.secrets.get("DEBUG_OTP", False):
                    st.warning(f"[DEV] SendGrid {r.status_code}: {r.text}. OTP affiché.")
                    st.code(otp)
                    return True, "Mode DEV : OTP affiché (SendGrid KO)."
                return False, f"SendGrid {r.status_code}: {r.text}"
        except Exception as e:
            if st.secrets.get("DEBUG_OTP", False):
                st.warning(f"[DEV] Envoi SendGrid impossible ({e}). OTP affiché.")
                st.code(otp)
                return True, "Mode DEV : OTP affiché (SendGrid KO)."
            return False, f"Échec d'envoi via SendGrid : {e}"

    if backend == "mailgun":
        if not _REQ_AVAILABLE:
            return False, "Mailgun nécessite 'requests'."
        domain = st.secrets.get("MAILGUN_DOMAIN")
        api_key = st.secrets.get("MAILGUN_API_KEY")
        from_addr = st.secrets.get("SMTP_FROM")
        if not domain or not api_key or not from_addr:
            return False, "Config Mailgun incomplète."
        url = f"https://api.mailgun.net/v3/{domain}/messages"
        data = {"from": from_addr, "to": [to_email], "subject": subject, "text": body}
        try:
            r = requests.post(url, auth=("api", api_key), data=data, timeout=20)
            if 200 <= r.status_code < 300:
                return True, "Un code vous a été envoyé par email."
            else:
                if st.secrets.get("DEBUG_OTP", False):
                    st.warning(f"[DEV] Mailgun {r.status_code}: {r.text}. OTP affiché.")
                    st.code(otp)
                    return True, "Mode DEV : OTP affiché (Mailgun KO)."
                return False, f"Mailgun {r.status_code}: {r.text}"
        except Exception as e:
            if st.secrets.get("DEBUG_OTP", False):
                st.warning(f"[DEV] Envoi Mailgun impossible ({e}). OTP affiché.")
                st.code(otp)
                return True, "Mode DEV : OTP affiché (Mailgun KO)."
            return False, f"Échec d'envoi via Mailgun : {e}"

    return False, f"EMAIL_BACKEND inconnu: {backend}. Utilise 'smtp', 'sendgrid' ou 'mailgun'."

def start_otp_flow(nom: str, prenom: str, email: str):
    otp = _gen_otp()
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
        if st.secrets.get("DEBUG_OTP", False):
            st.code(otp)

def render_otp_form() -> bool:
    if "pending_gate" not in st.session_state:
        return False

    pending = st.session_state["pending_gate"]
    expires_at = datetime.fromisoformat(pending["otp_expires_at"])
    remaining = max(0, int((expires_at - datetime.utcnow()).total_seconds() // 60))

    st.markdown("### ✉️ Vérification de votre email")
    st.caption(f"Un code a été envoyé à **{pending['email']}**. Il expirera dans {remaining} minute(s).")

    with st.form("otp_form", clear_on_submit=False):
        otp_input = st.text_input("Code de vérification (6 chiffres) *", max_chars=OTP_LENGTH)
        c1, c2 = st.columns(2)
        with c1:
            validate = st.form_submit_button("Valider")
        with c2:
            resend = st.form_submit_button("Renvoyer le code")

    if resend:
        start_otp_flow(pending["nom"], pending["prenom"], pending["email"])
        st.stop()

    if validate:
        if datetime.utcnow() > expires_at:
            st.error("Code expiré. Un nouveau code a été envoyé.")
            start_otp_flow(pending["nom"], pending["prenom"], pending["email"])
            st.stop()

        if otp_input and otp_input.strip() == pending["otp"]:
            ok, msg, email_hash = save_contact(pending["nom"], pending["prenom"], pending["email"])
            if ok:
                st.session_state["gate_passed"] = True
                st.session_state["user_email_hash"] = email_hash
                st.session_state["username"] = f"{pending['prenom'].title()} {pending['nom'].upper()}"
                # 🔑 ÉCRIRE LE COOKIE POUR LA PROCHAINE VISITE
                if _COOKIES_AVAILABLE:
                    cookies["user_email_hash"] = email_hash
                    cookies.save()
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

# =========================================================
# =================== PORTAIL D’ACCÈS =====================
# =========================================================
def require_user_gate():
    st.markdown("## 🔐 Accès")

    # 🔍 VÉRIFIER SI LE COOKIE CONTIENT UN HASH VALIDE
    if _COOKIES_AVAILABLE and "user_email_hash" in cookies and cookies["user_email_hash"]:
        email_hash = cookies["user_email_hash"]
        dfc = load_contacts()
        if "email_sha256" in dfc.columns and not dfc[dfc["email_sha256"] == email_hash].empty:
            user_row = dfc[dfc["email_sha256"] == email_hash].iloc[0]
            st.session_state["gate_passed"] = True
            st.session_state["user_email_hash"] = email_hash
            st.session_state["username"] = f"{user_row['prenom'].title()} {user_row['nom'].upper()}"
            st.success(f"Re-bonjour, {st.session_state['username']} ! 👋")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
            st.stop()

    # Si OTP actif et un code est en cours → afficher le formulaire OTP
    if AUTH_MODE == "otp" and render_otp_form():
        return

    with st.container():
        with st.form("gate_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("Nom *")
            with col2:
                prenom = st.text_input("Prénom *")
            email = st.text_input("Email *", placeholder="ex: nom@domaine.com")

            if AUTH_MODE == "otp":
                consent = st.checkbox(
                    "J’accepte que mes informations (nom, prénom, email) soient utilisées pour personnaliser l’application.",
                    value=True
                )

            button_label = "Recevoir un code et continuer" if AUTH_MODE == "otp" else "Continuer"
            submitted = st.form_submit_button(button_label)

            if submitted:
                if not nom or not prenom or not email:
                    st.error("Merci de remplir tous les champs obligatoires (*)"); st.stop()

                email_norm = email.strip().lower()
                if not re.match(EMAIL_REGEX, email_norm):
                    st.error("Adresse email invalide."); st.stop()

                domain = email_norm.split("@", 1)[1]
                if domain in DISPOSABLE_DOMAINS:
                    st.error("Les emails jetables ne sont pas autorisés."); st.stop()

                if not has_mx_record(email_norm):
                    st.error("Le domaine de l'email ne semble pas exister (MX introuvable)."); st.stop()

                if AUTH_MODE == "otp":
                    if not consent:
                        st.warning("Vous devez accepter pour continuer."); st.stop()
                    start_otp_flow(nom, prenom, email_norm)
                    st.stop()
                else:
                    ok, msg, email_hash = save_contact(nom, prenom, email_norm)
                    if ok:
                        st.session_state["gate_passed"] = True
                        st.session_state["user_email_hash"] = email_hash
                        st.session_state["username"] = f"{prenom.strip().title()} {nom.strip().upper()}"
                        # 🔑 ÉCRIRE LE COOKIE
                        if _COOKIES_AVAILABLE:
                            cookies["user_email_hash"] = email_hash
                            cookies.save()
                        st.success(msg)
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
                        st.stop()

# 🔒 Bloquer l'accès tant que non identifié
if not st.session_state.get("gate_passed"):
    require_user_gate()
    st.stop()

# --- ADMIN : Consulter / Exporter les contacts (protégé) ---
with st.sidebar.expander("🔑 Contacts (admin)", expanded=False):
    admin_key = st.text_input("Admin key", type="password")
    expected = st.secrets.get("ADMIN_KEY", None)

    if expected and admin_key == expected:
        try:
            dfc = load_contacts().copy()
        except Exception as e:
            st.error(f"Impossible de charger les contacts : {e}")
            dfc = pd.DataFrame(columns=["created_at","prenom","nom","email","email_sha256","id"])

        if "created_at" in dfc.columns:
            dfc["created_at"] = pd.to_datetime(dfc["created_at"], errors="coerce")
            dfc = dfc.sort_values("created_at", ascending=False)

        st.write(f"👥 {len(dfc)} contact(s) enregistré(s)")
        show_full = st.checkbox("Afficher les emails complets", value=False)
        df_view = dfc.copy()
        if "email" in df_view.columns and not show_full:
            df_view["email"] = df_view["email"].fillna("").str.replace(
                r"(^.).+(@.+$)", r"\1***\2", regex=True
            )

        cols_to_show = [c for c in ["created_at","prenom","nom","email"] if c in df_view.columns]
        st.dataframe(df_view[cols_to_show], use_container_width=True)

        with st.popover("🗓️ Filtrer par date (optionnel)"):
            min_d, max_d = None, None
            if "created_at" in dfc.columns and not dfc["created_at"].isna().all():
                min_d = pd.to_datetime(dfc["created_at"]).min().date()
                max_d = pd.to_datetime(dfc["created_at"]).max().date()
            if min_d and max_d:
                d1, d2 = st.date_input("Intervalle", (min_d, max_d))
                if isinstance(d1, pd.Timestamp): d1 = d1.date()
                if isinstance(d2, pd.Timestamp): d2 = d2.date()
                if d1 and d2:
                    mask = (pd.to_datetime(dfc["created_at"]).dt.date >= d1) & (pd.to_datetime(dfc["created_at"]).dt.date <= d2)
                    dfc_filtered = dfc.loc[mask].copy()
                else:
                    dfc_filtered = dfc
            else:
                dfc_filtered = dfc

            st.download_button(
                "📥 Télécharger (CSV, filtré si dates)",
                data=dfc_filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name="contacts.csv",
                mime="text/csv"
            )

        st.download_button(
            "⬇️ Télécharger (CSV complet)",
            data=dfc.to_csv(index=False).encode("utf-8-sig"),
            file_name="contacts_complet.csv",
            mime="text/csv"
        )

        st.caption("Astuce : les emails sont masqués par défaut. Coche l’option pour les voir en clair.")

    elif admin_key:
        st.error("Clé admin invalide.")

# =========================================================
# ===================== PAGE PRINCIPALE ===================
# =========================================================
if 'username' in st.session_state:
    st.title(f"⚽ Outil de Visualisation de Données Footballistiques - Bienvenue, {st.session_state['username']} !")
else:
    st.title("Outil de Visualisation de Données Footballistiques")

# --- UPLOAD CSV ---
st.sidebar.header("📁 Données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])
if not uploaded_file:
    st.warning("Veuillez importer un fichier CSV.")
    st.stop()

# ... (le reste du code de visualisation reste **inchangé** — tu peux le conserver tel quel)
# (Je ne le réécris pas ici pour ne pas alourdir, mais il suit exactement comme dans ton code initial)

# ⚠️ ⚠️ ⚠️ 
# → Colle ici **tout le reste de ton code original** à partir de "# --- LECTURE DU CSV ---"
# jusqu'à la fin.
# ⚠️ ⚠️ ⚠️
# =========================================================
# =================== PORTAIL D’ACCÈS =====================
# =========================================================
def require_user_gate():
    st.markdown("## 🔐 Accès")

    # Si OTP actif et un code est en cours → afficher le formulaire OTP
    if AUTH_MODE == "otp" and render_otp_form():
        return

    with st.container():
        with st.form("gate_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("Nom *")
            with col2:
                prenom = st.text_input("Prénom *")
            email = st.text_input("Email *", placeholder="ex: nom@domaine.com")

            if AUTH_MODE == "otp":
                consent = st.checkbox(
                    "J’accepte que mes informations (nom, prénom, email) soient utilisées pour personnaliser l’application.",
                    value=True
                )

            button_label = "Recevoir un code et continuer" if AUTH_MODE == "otp" else "Continuer"
            submitted = st.form_submit_button(button_label)

            if submitted:
                if not nom or not prenom or not email:
                    st.error("Merci de remplir tous les champs obligatoires (*)"); st.stop()

                email_norm = email.strip().lower()
                if not re.match(EMAIL_REGEX, email_norm):
                    st.error("Adresse email invalide."); st.stop()

                domain = email_norm.split("@", 1)[1]
                if domain in DISPOSABLE_DOMAINS:
                    st.error("Les emails jetables ne sont pas autorisés."); st.stop()

                if not has_mx_record(email_norm):
                    st.error("Le domaine de l'email ne semble pas exister (MX introuvable)."); st.stop()

                if AUTH_MODE == "otp":
                    if not consent:
                        st.warning("Vous devez accepter pour continuer."); st.stop()
                    start_otp_flow(nom, prenom, email_norm)
                    st.stop()
                else:
                    ok, msg, email_hash = save_contact(nom, prenom, email_norm)
                    if ok:
                        st.session_state["gate_passed"] = True
                        st.session_state["user_email_hash"] = email_hash
                        st.session_state["username"] = f"{prenom.strip().title()} {nom.strip().upper()}"
                        st.success(msg)
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
                        st.stop()

# 🔒 Bloquer l'accès tant que non identifié
if not st.session_state.get("gate_passed"):
    require_user_gate()
    st.stop()

# --- ADMIN : Consulter / Exporter les contacts (protégé) ---
with st.sidebar.expander("🔑 Contacts (admin)", expanded=False):
    # Saisie de la clé admin (à définir dans st.secrets)
    admin_key = st.text_input("Admin key", type="password")
    expected = st.secrets.get("ADMIN_KEY", None)  # Ex.: ADMIN_KEY="ta-cle-ultra-secrete" dans .streamlit/secrets.toml

    if expected and admin_key == expected:
        try:
            dfc = load_contacts().copy()
        except Exception as e:
            st.error(f"Impossible de charger les contacts : {e}")
            dfc = pd.DataFrame(columns=["created_at","prenom","nom","email","email_sha256","id"])

        # Tri du plus récent au plus ancien si la colonne existe
        if "created_at" in dfc.columns:
            dfc["created_at"] = pd.to_datetime(dfc["created_at"], errors="coerce")
            dfc = dfc.sort_values("created_at", ascending=False)

        # Masquer les emails par défaut (option pour afficher en clair)
        st.write(f"👥 {len(dfc)} contact(s) enregistré(s)")
        show_full = st.checkbox("Afficher les emails complets", value=False)
        df_view = dfc.copy()
        if "email" in df_view.columns and not show_full:
            df_view["email"] = df_view["email"].fillna("").str.replace(
                r"(^.).+(@.+$)", r"\1***\2", regex=True
            )

        # Colonnes affichées (ajuste si besoin)
        cols_to_show = [c for c in ["created_at","prenom","nom","email"] if c in df_view.columns]
        st.dataframe(df_view[cols_to_show], use_container_width=True)

        # Filtre date (optionnel mais pratique)
        with st.popover("🗓️ Filtrer par date (optionnel)"):
            min_d, max_d = None, None
            if "created_at" in dfc.columns and not dfc["created_at"].isna().all():
                min_d = pd.to_datetime(dfc["created_at"]).min().date()
                max_d = pd.to_datetime(dfc["created_at"]).max().date()
            if min_d and max_d:
                d1, d2 = st.date_input("Intervalle", (min_d, max_d))
                if isinstance(d1, pd.Timestamp): d1 = d1.date()
                if isinstance(d2, pd.Timestamp): d2 = d2.date()
                if d1 and d2:
                    mask = (pd.to_datetime(dfc["created_at"]).dt.date >= d1) & (pd.to_datetime(dfc["created_at"]).dt.date <= d2)
                    dfc_filtered = dfc.loc[mask].copy()
                else:
                    dfc_filtered = dfc
            else:
                dfc_filtered = dfc

            # Export filtré
            st.download_button(
                "📥 Télécharger (CSV, filtré si dates)",
                data=dfc_filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name="contacts.csv",
                mime="text/csv"
            )

        # Export complet (sans filtre)
        st.download_button(
            "⬇️ Télécharger (CSV complet)",
            data=dfc.to_csv(index=False).encode("utf-8-sig"),
            file_name="contacts_complet.csv",
            mime="text/csv"
        )

        st.caption("Astuce : les emails sont masqués par défaut. Coche l’option pour les voir en clair.")

    elif admin_key:
        st.error("Clé admin invalide.")

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

# --- CLASSIFICATION EN 3 ZONES ---
def classify_zone(x, y):
    if x < 40:
        return 'Haute'
    elif x <= 80:
        return 'Médiane'
    else:
        return 'Basse'

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
    arrow_width = st.slider("Épaisseur des flèches", 0.5, 5.0, 2.0, 0.5)
    arrow_head_scale = st.slider("Taille de la tête des flèches", 1.0, 10.0, 2.0, 0.5)
    arrow_alpha = st.slider("Opacité des flèches", 0.1, 1.0, 0.8, 0.1)
    point_size = st.slider("Taille des points", 20, 200, 80, 10)
    scatter_alpha = st.slider("Opacité des points", 0.1, 1.0, 0.8, 0.1)
    heatmap_alpha = st.slider("Opacité de la heatmap", 0.1, 1.0, 0.85, 0.05)
    heatmap_statistic = st.selectbox("Type de statistique", ['count', 'density'], index=0)
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

# --- VISUALISATIONS PAR ZONES ---
st.markdown("---")
st.header("Visualisations sur Terrain - Analyse par Zones")

common_pitch_params = {
    'pitch_color': 'white',
    'line_color': 'black',
    'linewidth': 1,
    'line_zorder': 2
}
fig_size = (8, 5.5)

zones_rects = {
    'Haute': (0, 0, 40, 80),
    'Médiane': (40, 0, 40, 80),
    'Basse': (80, 0, 40, 80)
}

zone_colors = {
    'Haute': '#FFD700',
    'Médiane': '#98FB98',
    'Basse': '#87CEEB'
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

            if 'combined_images' in locals() and combined_images:
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
            if 'combined_images' in locals():
                for _, img in combined_images:
                    try:
                        os.unlink(img)
                    except:
                        pass
            for f in temp_files:
                try:
                    os.unlink(f)
                except:
                    pass
