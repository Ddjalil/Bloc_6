
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

# team
st.header("Une équipe d'étudiants issus de multiples secteurs")
col1, col2, _ = st.columns([3,2,2])
with col1:
    st.markdown("**📊 Delphine CESAR** - Marketing")
    st.markdown("**💸 Dalanda DIALLO** - Conseil en finance")
    st.markdown("**🛫 Djalil DJEBALI** - Logistique")
    st.markdown("**🔬 Céline MARTINEAU** - Recherche en biologie")
    st.markdown("**📽️ Bastien MONIN** - Audiovisuel")
with col2:
    st.markdown("[Linkedin](https://www.linkedin.com/in/delphinecesar/) | [Github](https://github.com/delphinecesar)")
    st.markdown("[Linkedin](https://fr.linkedin.com/in/mdalanda-diallo/) | [Github](https://github.com/DalandaDIALLO3)")
    st.markdown("[Linkedin](https://www.linkedin.com/in/djalil-djebali/) | [Github](https://github.com/Ddjalil)")
    st.markdown("[Linkedin](https://www.linkedin.com/in/cnmartineau/) | [Github](https://github.com/cnmartineau)")
    st.markdown("[Linkedin](https://www.linkedin.com/in/bastien-monin/) | [Github](https://github.com/Fre7r)")

# school
st.header("Une école pour se former à la Data Science")

col1, _ = st.columns([4,1])
with col1:
    st.markdown("[Jedha](https://www.jedha.co) est l'école de référence pour se former aux nouveaux métiers de la Tech.  \
            Avec 3 parcours d'excellence en Data Analysis, en Data Science & en Cybersécurité, Jedha démocratise \
            l'accès aux compétences les plus recherchées.")
    st.markdown("Depuis sa création en 2017, ses formations accélérées en 2 à 12 semaines ont permis à des milliers d'élèves \
            de booster leur carrière et de rejoindre les leaders de la Tech.")
    st.markdown("**Jedha en chiffres :**")
    st.markdown("* Meilleur bootcamp français en data science 2023 selon COURSE REPORT")
    st.markdown("* +1600 alumni | 87% d'insertion professionelle dans les 6 mois | 14 projets par formation")

# thanks
st.header("Remerciements")

col1, _ = st.columns([4,1])
with col1:
    st.markdown("Nous remercions Jedha, l'ensemble de nos professeurs et professeurs-assistants  \
                pour leur accompagnement tout au long de la formation et plus particulièrement \
                lors de la réalisation de ce projet.")