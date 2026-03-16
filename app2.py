import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


DECK_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "Unknown": 7,
}

SIDE_MAPPING = {
    "P - Babord": 0,
    "S - Tribord": 1,
}


st.set_page_config(
    page_title="Spaceship Titanic",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def result(is_win: bool) -> None:
    col1, col2, col3 = st.columns([1.6, 1.2, 1.6])

    if is_win:
        with col2:
            st.success("Bravo, tu as survécu(e) - vers l'infini et au-delà !")
            st.image("images/ludowinner.png", use_container_width=True)
    else:
        with col2:
            st.error("Oups, tu as été pulvérisé(e).")
            st.image("images/sylvainlose.png", use_container_width=True)


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    df_display = df.copy()
    for column in df_display.columns:
        if df_display[column].dtype == "object":
            df_display[column] = df_display[column].astype(str)
    return df_display


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw/train.csv")


@st.cache_data
def load_processed_data():
    return pd.read_csv("data/processed/df_titanic_clean.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/titanic_model.pkl")


try:
    df_raw = load_raw_data()
    df_processed = load_processed_data()
    model = load_model()
    data_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    data_loaded = False


st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Sommaire :",
    [
        "Accueil",
        "Analyse Exploratoire (EDA)",
        "Test du Modèle",
        "Conclusions & Perspectives",
    ],
)
st.sidebar.image("images/spaceship.png", use_container_width=True)


if menu == "Accueil":
    st.title("Spaceship Titanic")

    with st.expander("Vos missions : cliquer pour développer", expanded=True):
        st.markdown(
            """
            1. **Données** : charger le dataset brut dans `data/raw/train.csv`.
            2. **Nettoyage** : charger le dataset traité dans `data/processed/df_titanic_clean.csv`.
            3. **EDA** : comparer les données brutes et les données nettoyées.
            4. **Modélisation** : charger le modèle entraîné depuis `models/titanic_model.pkl`.
            5. **Prédiction** : tester la survie d'un passager avec le formulaire spatial.
            """
        )

    if data_loaded:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lignes brutes", df_raw.shape[0])
        m2.metric("Colonnes brutes", df_raw.shape[1])
        m3.metric("Lignes nettoyées", df_processed.shape[0])
        m4.metric("Colonnes nettoyées", df_processed.shape[1])

elif menu == "Analyse Exploratoire (EDA)":
    st.title("Analyse Exploratoire des Données (EDA)")
    st.caption(
        "Compare les données brutes et les données nettoyées avant de lancer une prédiction."
    )
    st.divider()

    if not data_loaded:
        st.warning("Les données ne sont pas chargées.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Aperçu brut", "Aperçu nettoyé", "Statistiques", "Visualisations"]
        )

        with tab1:
            st.subheader("Extrait du dataset brut")
            st.dataframe(make_arrow_compatible(df_raw.head()), use_container_width=True)

        with tab2:
            st.subheader("Extrait du dataset nettoyé")
            st.dataframe(
                make_arrow_compatible(df_processed.head()), use_container_width=True
            )

        with tab3:
            st.subheader("Description des variables")
            m1, m2 = st.columns(2)
            m1.metric("Valeurs manquantes brut", int(df_raw.isna().sum().sum()))
            m2.metric(
                "Valeurs manquantes nettoyées", int(df_processed.isna().sum().sum())
            )

            with st.expander("Statistiques du dataset brut", expanded=True):
                st.dataframe(
                    make_arrow_compatible(df_raw.describe(include="all").reset_index()),
                    use_container_width=True,
                )

            with st.expander("Statistiques du dataset nettoyé", expanded=True):
                st.dataframe(
                    make_arrow_compatible(
                        df_processed.describe(include="all").reset_index()
                    ),
                    use_container_width=True,
                )

        with tab4:
            st.markdown("### Répartition des passagers transportés")
            if "Transported" in df_raw.columns:
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                df_raw["Transported"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Répartition des passagers transportés")
                st.pyplot(fig, use_container_width=False)
            st.divider()

            st.markdown("### Transport selon la planète d'origine")
            if {"HomePlanet", "Transported"}.issubset(df_raw.columns):
                fig, ax = plt.subplots(figsize=(4.2, 2.6))
                pd.crosstab(df_raw["HomePlanet"], df_raw["Transported"]).plot(
                    kind="bar", ax=ax
                )
                ax.set_title("Transport selon la planète d'origine")
                ax.set_ylabel("Nombre de passagers")
                ax.set_xlabel("HomePlanet")
                st.pyplot(fig, use_container_width=False)
            st.divider()

            st.markdown("### Transport selon CryoSleep")
            if {"CryoSleep", "Transported"}.issubset(df_raw.columns):
                fig, ax = plt.subplots(figsize=(4.2, 2.6))
                pd.crosstab(df_raw["CryoSleep"], df_raw["Transported"]).plot(
                    kind="bar", ax=ax
                )
                ax.set_title("Transport selon CryoSleep")
                ax.set_ylabel("Nombre de passagers")
                ax.set_xlabel("CryoSleep")
                st.pyplot(fig, use_container_width=False)
            st.divider()

            st.markdown("### Probabilité d'être transporté selon CryoSleep")
            if {"CryoSleep", "Target"}.issubset(df_processed.columns):
                fig, ax = plt.subplots(figsize=(4.2, 2.6))
                sns.barplot(
                    data=df_processed.reset_index(drop=True),
                    x="CryoSleep",
                    y="Target",
                    ax=ax,
                )
                ax.set_title("Probabilité d'être transporté selon CryoSleep")
                ax.set_ylabel("Probabilité")
                ax.set_xlabel("CryoSleep")
                st.pyplot(fig, use_container_width=False)
            st.divider()

            st.markdown("### Transport selon la destination")
            if {"Destination", "Transported"}.issubset(df_raw.columns):
                fig, ax = plt.subplots(figsize=(4.2, 2.6))
                pd.crosstab(df_raw["Destination"], df_raw["Transported"]).plot(
                    kind="bar", ax=ax
                )
                ax.set_title("Transport selon la destination")
                ax.set_ylabel("Nombre de passagers")
                ax.set_xlabel("Destination")
                st.pyplot(fig, use_container_width=False)
            st.divider()

            st.markdown("### Transport selon le deck")
            if {"Cabin", "Transported"}.issubset(df_raw.columns):
                deck_df = df_raw[["Cabin", "Transported"]].copy()
                deck_df["Deck"] = deck_df["Cabin"].astype(str).str.split("/").str[0]
                fig, ax = plt.subplots(figsize=(4.2, 2.6))
                pd.crosstab(deck_df["Deck"], deck_df["Transported"]).plot(
                    kind="bar", ax=ax
                )
                ax.set_title("Transport selon le deck")
                ax.set_ylabel("Nombre de passagers")
                ax.set_xlabel("Deck")
                st.pyplot(fig, use_container_width=False)

elif menu == "Test du Modèle":
    st.title("Vérifie si tu aurais survécu au Spaceship Titanic")

    if not data_loaded:
        st.warning("Le modèle ou les données ne sont pas chargés.")
    else:
        with st.container(border=True):
            st.info("Renseigne tes informations personnelles et les détails du voyage.")

            # st.markdown("### Profil passager :")
            profil_col1, profil_col2 = st.columns(2)
            with profil_col1:
                # with st.container(border=True):
                    st.caption("Identité :")
                    age = st.slider("Âge", 0, 80, 40)
                    vip = st.segmented_control(
                        "Statut VIP",
                        options=[0, 1],
                        default=0,
                        key="vip_control",
                        format_func=lambda x: "Non" if x == 0 else "Oui",
                    )
            with profil_col2:
                with st.container(border=True):
                    st.caption("Conditions du voyage :")
                    cryosleep = st.segmented_control(
                        "Mode CryoSleep :",
                        options=[0, 1],
                        default=0,
                        key="cryosleep_control",
                        format_func=lambda x: "Non" if x == 0 else "Oui",
                    )
                    depenses_activees = st.segmented_control(
                        "Souhaites-tu dépenser ?",
                        options=[0, 1],
                        default=0,
                        key="spend_control",
                        format_func=lambda x: "Non" if x == 0 else "Oui",
                    )

            with st.form("formulaire_prediction", clear_on_submit=False):
                st.markdown("### Voyage :")
                voyage_col1, voyage_col2 = st.columns([1.1, 1.4])
                with voyage_col1:
                    with st.container(border=True):
                        homeplanet = st.segmented_control(
                            "Planète d'origine :",
                            options=["Earth", "Europa", "Mars", "Inconnue"],
                            default="Earth",
                            key="homeplanet_control",
                        )
                with voyage_col2:
                    with st.container(border=True):
                        cabin_top_col1, cabin_top_col2 = st.columns(2)
                        with cabin_top_col1:
                            cabin_num = st.number_input(
                                "Numéro de cabine :",
                                min_value=0,
                                max_value=2000,
                                value=0,
                                step=1,
                            )
                        with cabin_top_col2:
                            deck_label = st.selectbox(
                                "Deck :",
                                options=list(DECK_MAPPING.keys()),
                                index=0,
                            )
                        side_label = st.segmented_control(
                            "Côté :",
                            options=list(SIDE_MAPPING.keys()),
                            default="P - Babord",
                            key="side_control",
                        )

                st.markdown("### Dépenses à bord :")
                if depenses_activees == 1:
                    with st.container(border=True):
                        spend_col1, spend_col2, spend_col3, spend_col4, spend_col5 = st.columns(5)
                        with spend_col1:
                            room_service = st.number_input(
                                "RoomService :", min_value=0.0, value=0.0, step=10.0
                            )
                        with spend_col2:
                            food_court = st.number_input(
                                "FoodCourt :", min_value=0.0, value=0.0, step=10.0
                            )
                        with spend_col3:
                            shopping_mall = st.number_input(
                                "ShoppingMall :", min_value=0.0, value=0.0, step=10.0
                            )
                        with spend_col4:
                            spa = st.number_input(
                                "Spa :", min_value=0.0, value=0.0, step=10.0
                            )
                        with spend_col5:
                            vr_deck = st.number_input(
                                "VRDeck :", min_value=0.0, value=0.0, step=10.0
                            )
                else:
                    with st.container(border=True):
                        st.caption("Aucune dépense sélectionnée")
                    room_service = 0.0
                    food_court = 0.0
                    shopping_mall = 0.0
                    spa = 0.0
                    vr_deck = 0.0

                predire = st.form_submit_button(
                    "Appuie si tu veux connaître ton sort",
                    type="primary",
                    use_container_width=True,
                )

        if predire:
            try:
                expected_columns = list(model.feature_names_in_)
                input_data = pd.DataFrame(0, index=[0], columns=expected_columns)

                if "Age" in input_data.columns:
                    input_data.loc[0, "Age"] = age
                if "CryoSleep" in input_data.columns:
                    input_data.loc[0, "CryoSleep"] = cryosleep
                if "VIP" in input_data.columns:
                    input_data.loc[0, "VIP"] = vip
                if "Num" in input_data.columns:
                    input_data.loc[0, "Num"] = cabin_num
                if "Side" in input_data.columns:
                    input_data.loc[0, "Side"] = SIDE_MAPPING[side_label]
                if "Side_P" in input_data.columns:
                    input_data.loc[0, "Side_P"] = 1 if side_label == "P - Babord" else 0
                if "Side_S" in input_data.columns:
                    input_data.loc[0, "Side_S"] = 1 if side_label == "S - Tribord" else 0
                if "RoomService" in input_data.columns:
                    input_data.loc[0, "RoomService"] = room_service
                if "FoodCourt" in input_data.columns:
                    input_data.loc[0, "FoodCourt"] = food_court
                if "ShoppingMall" in input_data.columns:
                    input_data.loc[0, "ShoppingMall"] = shopping_mall
                if "Spa" in input_data.columns:
                    input_data.loc[0, "Spa"] = spa
                if "VRDeck" in input_data.columns:
                    input_data.loc[0, "VRDeck"] = vr_deck
                if "over_15" in input_data.columns:
                    input_data.loc[0, "over_15"] = 1 if age > 15 else 0
                if "Deck_encoded" in input_data.columns:
                    input_data.loc[0, "Deck_encoded"] = DECK_MAPPING[deck_label]

                if homeplanet == "Earth" and "HomePlanet_Earth" in input_data.columns:
                    input_data.loc[0, "HomePlanet_Earth"] = 1
                elif (
                    homeplanet == "Europa"
                    and "HomePlanet_Europa" in input_data.columns
                ):
                    input_data.loc[0, "HomePlanet_Europa"] = 1
                elif homeplanet == "Mars" and "HomePlanet_Mars" in input_data.columns:
                    input_data.loc[0, "HomePlanet_Mars"] = 1
                elif "HomePlanet_nan" in input_data.columns:
                    input_data.loc[0, "HomePlanet_nan"] = 1

                prediction = model.predict(input_data)
                result(int(prediction[0]) == 1)

            except Exception as e:
                st.error(f"Erreur lors de la prediction : {e}")

elif menu == "Conclusions & Perspectives":
    st.title("Conclusions et pistes d'amelioration")
    st.caption(
        "Cette section presente un bilan du projet Spaceship Titanic, de l'exploration des donnees jusqu'a l'utilisation du modele."
    )
    st.markdown(
        """
        ### Bilan du projet
        Ce projet Spaceship Titanic montre une demarche complete de data science :
        analyse des donnees, nettoyage, modelisation puis integration dans une application Streamlit.

        ### Ce que montre l'application
        L'application permet maintenant de comparer le dataset brut et le dataset nettoye,
        puis d'utiliser le modele pour produire une prediction a partir d'un formulaire.

        ### Perspectives d'amelioration
        Il reste possible d'ajouter les metriques du modele, une matrice de confusion,
        plus de visualisations et une meilleure mise en valeur des variables importantes.
        """
    )


st.divider()
st.caption("Spaceship Titanic - Par Christian - Mira 2025-2026")
