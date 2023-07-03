
# import libraries
import streamlit as st

# layout
st.set_page_config(layout="wide")

st.markdown(""" 
<style>

    /* above the header */
    [data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    }

    /* central part */
    [data-testid="stAppViewContainer"] {
    background-color: white;
    }

    /* sidebar menu */
    [data-testid="stSidebarNav"] {
    background-color: #044389;
    }

    /* sidebar*/
    [data-testid="stSidebar"] {
    background-color: #044389;
    }

    /* Sidebar links*/
    .css-17lntkn {
    color:white;
    font-size: 20px;
    }
    .css-17lntkn:hover {
    color:#FF4B4B !important;
    font-size: 20px;
    }
    .css-pkbazv {
    color:white;
    font-size: 20px;
    }
    .css-pkbazv:hover {
    color:#FF4B4B !important;
    font-size: 20px;
    }
    /* streamlit buttons*/
        /* arrow + 3 lines*/
    .css-fblp2m{
        color:#044389 !important;
    }
    .css-fblp2m:hover{
        color:#FF4B4B !important;
    }
        /* cross*/
    section[data-testid="stSidebar"] .css-fblp2m{
        color:white !important;
    }
    section[data-testid="stSidebar"] .css-fblp2m:hover{
        color:#FF4B4B !important;
    }

    /* Titles*/
    .css-10trblm {
    color:#044389;
    }

    /* Streamlit link*/
    .css-1vbd788 {
    color:#044389 !important;
    text-decoration-line:underline;
    }
    .css-1vbd788:hover {
    color:#FF4B4B !important;
    }

    /* Links*/
    a {
    color:#044389 !important;
    }
    a:hover {
    color:#FF4B4B !important;
    }
""",unsafe_allow_html=True)

# project
st.header("Le projet")
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("Nous sommes un groupe de cinq étudiants en Data Science en formation chez **[Jedha](https://www.jedha.co)** \
                et nous travaillons actuellement sur notre projet de fin d'études qui s'appelle **Main'Terprète**. Ce projet open-source vise à appliquer \
                nos compétences en vue de faciliter la compréhension de la langue des signes.")
    
    st.markdown("Notre application de traduction de la langue des signes par vidéo a pour but de simplifier les échanges \
                entre les personnes sourdes et les personnes entendantes. Cette application permet de filmer une personne en train \
                de signer un mot et de fournir ensuite une traduction en langue française. Nous sommes convaincus que **Main’Terprète** \
                contribuera à renforcer la communication entre les personnes sourdes et le reste de la société.")

with col2:
    st.markdown("")

# key numbers
st.header("Quelques chiffres clés")
col1, _ = st.columns([4,1])
with col1:
    st.markdown("L’Organisation mondiale de la Santé (OMS) estime que \
        466 millions de personnes souffrent de déficience auditive incapacitante dans le monde, \
        dont 34 millions d'enfants.")

    st.markdown("Pour la France, l'Inserm estime que la surdité affecte \
        6 % des 15-24 ans, et plus de 65 % des 65 ans et plus.")

    st.markdown("Toujours selon le même organisme, chaque année, près \
        d'un millier d'enfants naissent atteints de surdité dans le pays.")
    st.markdown("*Source : [Fondation pour la Recherche Médicale](https://www.frm.org/recherches-autres-maladies/surdite/focus-surdite)*")

# references
st.header("Références")
col1, _ = st.columns([4,1])
with col1:
    st.markdown("J. Fink, B. Frénay, L. Meurant and A. Cleve, LSFB-CONT and LSFB-ISOL: Two New \
        Datasets for Vision-Based Sign Language Recognition, 2021 International Joint Conference \
        on Neural Networks (IJCNN), Shenzhen, China, 2021, pp. 1-8.")

    st.markdown("Subramanian, B., Olimov, B., Naik, S.M. et al. An integrated mediapipe-optimized \
        GRU model for Indian sign language recognition. Sci Rep 12, 11964 (2022).")

# developers
st.header("Pour les développeurs")
col1, _ = st.columns([4,1])
with col1:
    st.markdown("Main'Terprète est un projet open-source. \
        Vous le trouverez sur [Github](https://github.com/Fre7r).")

    
  