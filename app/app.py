import json
import os

import joblib
import pandas as pd
import streamlit as st

DOSSIER_APP = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PROJET = os.path.dirname(DOSSIER_APP)
CHEMIN_MODELE = os.path.join(DOSSIER_PROJET, "artifacts", "flight_price_model.joblib")
CHEMIN_META = os.path.join(DOSSIER_PROJET, "artifacts", "flight_price_model.meta.json")

st.set_page_config(page_title="Flight Price Predictor", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Chargement du modele en cours...")
def charger_modele(chemin_modele, chemin_meta):
    if not os.path.isfile(chemin_modele):
        st.error(f"Modele introuvable : {chemin_modele}")
        st.stop()
    if not os.path.isfile(chemin_meta):
        st.error(f"Metadonnees introuvables : {chemin_meta}")
        st.stop()
    try:
        pipeline = joblib.load(chemin_modele)
        with open(chemin_meta, encoding="utf-8") as fichier:
            meta = json.load(fichier)
    except Exception as erreur:
        st.error(f"Chargement impossible : {erreur}")
        st.stop()
    return pipeline, meta


pipeline, meta = charger_modele(CHEMIN_MODELE, CHEMIN_META)

CATEGORIES_ORDINALES = meta["ordre_categories_ordinales"]
CATEGORIES_NOMINALES = meta["categories_possibles_nominales"]
COLONNES_ATTENDUES = meta["features"]["colonnes_utilisees_par_le_transformer"]
METRIQUES = meta["metriques_test"]
IC95 = METRIQUES["IC95_MAE"]
PRIX_MIN, PRIX_MAX = 1105, 123071

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #f8fafc; }

header[data-testid="stHeader"] { background: transparent; }

.bandeau {
    background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 45%, #0ea5e9 100%);
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(30, 58, 138, 0.22);
}
.bandeau h1 { color: #ffffff; margin: 0 0 6px 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
.bandeau p  { color: #dbeafe; margin: 0; font-size: 15px; font-weight: 400; }

.carte {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}
.carte h4 { margin: 0 0 14px 0; font-size: 13px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.1px; color: #64748b; }

.tuile { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
         padding: 16px 18px; text-align: center;
         box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05); }
.tuile .val { font-size: 24px; font-weight: 800; color: #0f172a; line-height: 1.2; }
.tuile .lab { font-size: 11px; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.8px; color: #94a3b8; margin-top: 6px; }

.resultat { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            border-radius: 16px; padding: 28px 24px; text-align: center;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.28); }
.resultat .montant { font-size: 46px; font-weight: 800; color: #ffffff; line-height: 1.05; }
.resultat .devise { font-size: 14px; color: #93c5fd; font-weight: 600; margin-top: 8px; }
.resultat .fourchette { color: #cbd5e1; font-size: 13px; margin-top: 16px;
                        border-top: 1px solid rgba(255,255,255,0.18); padding-top: 14px; }

.vide { background: #ffffff; border: 2px dashed #cbd5e1; border-radius: 16px;
        padding: 46px 24px; text-align: center; color: #94a3b8; font-size: 15px; }
.vide .grand { font-size: 34px; margin-bottom: 10px; }

.stButton > button {
    width: 100%;
    background: linear-gradient(120deg, #1d4ed8 0%, #0ea5e9 100%);
    color: #ffffff; border: none; border-radius: 12px;
    padding: 13px 18px; font-weight: 700; font-size: 15px;
    box-shadow: 0 6px 18px rgba(29, 78, 216, 0.32);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover { transform: translateY(-1px);
                           box-shadow: 0 10px 24px rgba(29, 78, 216, 0.42); }

.stSelectbox label, .stNumberInput label { font-weight: 600; color: #334155; font-size: 13px; }
[data-testid="stForm"] { background: #ffffff; border: 1px solid #e2e8f0;
                         border-radius: 16px; padding: 18px;
                         box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05); }
[data-baseweb="select"] > div, [data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] { border-radius: 10px !important; }
hr { border-color: #e2e8f0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def nombre_fr(valeur, decimales=0):
    return f"{valeur:,.{decimales}f}".replace(",", " ")


def tuile(valeur, libelle):
    return (
        f'<div class="tuile"><div class="val">{valeur}</div>'
        f'<div class="lab">{libelle}</div></div>'
    )


st.markdown(
    f"""
    <div class="bandeau">
        <h1>Prediction du prix d'un billet d'avion</h1>
        <p>{meta['nom_modele']} &nbsp;|&nbsp; exporte le {meta['date_export']}
           &nbsp;|&nbsp; scikit-learn {meta['version_sklearn']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

colonnes_tuiles = st.columns(4)
colonnes_tuiles[0].markdown(
    tuile(nombre_fr(METRIQUES["MAE"]), "MAE test"), unsafe_allow_html=True
)
colonnes_tuiles[1].markdown(
    tuile(f"{METRIQUES['R2']:.4f}", "R2"), unsafe_allow_html=True
)
colonnes_tuiles[2].markdown(
    tuile(
        f"[{nombre_fr(IC95['borne_basse'])} - {nombre_fr(IC95['borne_haute'])}]",
        "IC 95 % MAE",
    ),
    unsafe_allow_html=True,
)
colonnes_tuiles[3].markdown(
    tuile(nombre_fr(METRIQUES["nombre_observations_test"]), "lignes test"),
    unsafe_allow_html=True,
)
st.write("")

col_form, col_res = st.columns([1.12, 1], gap="large")

ligne = None

with col_form:
    st.markdown(
        '<div class="carte"><h4>Caracteristiques du vol</h4></div>',
        unsafe_allow_html=True,
    )
    with st.form("formulaire_vol"):
        rang_1 = st.columns(3)
        airline = rang_1[0].selectbox("Compagnie", CATEGORIES_NOMINALES["airline"])
        source = rang_1[1].selectbox("Depart de", CATEGORIES_NOMINALES["source_city"])
        villes_arrivee = [v for v in CATEGORIES_NOMINALES["destination_city"] if v != source]
        destination = rang_1[2].selectbox("Arrivee a", villes_arrivee)

        rang_2 = st.columns(3)
        classe = rang_2[0].selectbox("Classe", CATEGORIES_ORDINALES["class"])
        escales = rang_2[1].selectbox("Escales", CATEGORIES_ORDINALES["stops"])
        depart = rang_2[2].selectbox("Creneau depart", CATEGORIES_NOMINALES["departure_time"])

        rang_3 = st.columns(2)
        duree = rang_3[0].number_input(
            "Duree du vol (h)", min_value=0.5, max_value=50.0, value=2.0,
            step=0.25, format="%.2f",
        )
        jours = rang_3[1].number_input(
            "Jours avant depart", min_value=1, max_value=60, value=30, step=1
        )
        arrivee = st.selectbox("Creneau arrivee", CATEGORIES_NOMINALES["arrival_time"])

        predict_pressed = st.form_submit_button("Estimer le prix")

with col_res:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    if not predict_pressed:
        st.markdown(
            '<div class="vide"><div class="grand">&#9992;</div>'
            "Renseignez le vol a gauche puis lancez l'estimation.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="carte"><h4>Fiabilite de la modelisation</h4>'
            f'<div style="font-size:14px;color:#334155;line-height:1.65">'
            f"Le modele a ete entraine sur {nombre_fr(meta['taille_dataset']['train'])} lignes "
            f"et evalue sur {nombre_fr(METRIQUES['nombre_observations_test'])} lignes "
            f"reservees ({meta['split']['test_size']:.0%} du jeu, random_state "
            f"{meta['split']['random_state']}).<br><br>"
            f"Erreur absolue mediane de l'ecart attendu : "
            f"<b>{nombre_fr(METRIQUES['MAE'])}</b> unites.<br><br>"
            f"Prix observes dans le jeu de donnees : de "
            f"{nombre_fr(PRIX_MIN)} a {nombre_fr(PRIX_MAX)}.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        saisie = {
            "duration": duree,
            "days_left": jours,
            "stops": escales,
            "class": classe,
            "airline": airline,
            "source_city": source,
            "destination_city": destination,
            "departure_time": depart,
            "arrival_time": arrivee,
        }
        manquantes = [c for c in COLONNES_ATTENDUES if c not in saisie]
        en_trop = [c for c in saisie if c not in COLONNES_ATTENDUES]

        if manquantes:
            st.error(f"Colonnes manquantes dans la saisie : {', '.join(manquantes)}")
        elif en_trop:
            st.error(f"Colonnes non attendues par le pipeline : {', '.join(en_trop)}")
        else:
            ligne = pd.DataFrame([saisie])[COLONNES_ATTENDUES]
            try:
                prix = float(pipeline.predict(ligne)[0])
            except Exception as erreur:
                st.error(f"Prediction impossible : {erreur}")
            else:
                bas, haut = prix - METRIQUES["MAE"], prix + METRIQUES["MAE"]
                position = (prix - PRIX_MIN) / (PRIX_MAX - PRIX_MIN) * 100
                position = min(max(position, 0.0), 100.0)
                st.markdown(
                    f"""
                    <div class="resultat">
                        <div class="montant">{nombre_fr(prix)}</div>
                        <div class="devise">prix estime du billet</div>
                        <div class="fourchette">
                            fourchette realiste : {nombre_fr(max(bas, PRIX_MIN))} -
                            {nombre_fr(haut)} unites<br>
                            incertitude du modele : +/- {nombre_fr(METRIQUES['MAE'])} unites
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown(
                    f"""
                    <div class="carte"><h4>Position dans le jeu de donnees</h4>
                    <div style="height:14px;border-radius:7px;
                                background:linear-gradient(90deg,#dbeafe 0%,#93c5fd 100%);
                                position:relative;margin:10px 0 8px 0;">
                        <div style="position:absolute;left:{position:.1f}%;top:-5px;
                                    width:4px;height:24px;border-radius:2px;
                                    background:#1d4ed8;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:12px;color:#64748b;">
                        <span>{nombre_fr(PRIX_MIN)}</span>
                        <span>{nombre_fr(PRIX_MAX)}</span>
                    </div></div>
                    """,
                    unsafe_allow_html=True,
                )

st.write("")
with st.expander("Detail de la requete envoyee au pipeline"):
    st.dataframe(
        ligne if ligne is not None else pd.DataFrame(columns=COLONNES_ATTENDUES),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Ordre des variables impose par artifacts/flight_price_model.meta.json "
        f"({len(COLONNES_ATTENDUES)} colonnes). Les colonnes non listees par le "
        f"ColumnTransformer sont ignorees."
    )

with st.sidebar:
    st.markdown("### A propos du modele")
    st.markdown(
        f"- **Type** : {meta['nom_modele']} (`{meta['type_estimator']}`)\n"
        f"- **Arbres** : {meta['parametres_estimator']['n_estimators']}\n"
        f"- **Cible** : `{meta['features']['cible']}`\n"
        f"- **RMSE test** : {nombre_fr(METRIQUES['RMSE'])}\n"
        f"- **Export** : {meta['date_export']}\n"
        f"- **Pipeline** : {' -> '.join(meta['etapes_pipeline'])}"
    )
    st.divider()
    st.markdown("### Variables du modele")
    st.markdown(
        f"**Numeriques** : {', '.join(meta['colonnes']['numeriques'])}\n\n"
        f"**Ordinales** : {', '.join(meta['colonnes']['ordinales'])}\n\n"
        f"**Nominales** : {', '.join(meta['colonnes']['nominales'])}"
    )
    st.divider()
    st.caption(
        "Projet de prediction du prix des billets d'avion. "
        "Application Streamlit branchee sur un pipeline scikit-learn exporte "
        "depuis le notebook d'analyse."
    )
