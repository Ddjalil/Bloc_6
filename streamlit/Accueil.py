# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tempfile
import cv2
import mediapipe as mp
import requests
import mediapipe.python.solutions.holistic as mp_holistic
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_utils as mp_drawing_styles
import time

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

    /*page Accueil*/
    /* "Lancer M'ain'Terprète" button*/
    .stButton button {
        background-color: white;
        color: #044389;
    }
    .stButton button:hover {
        background-color: white;
        color: #FF4B4B;
        border-color:#FF4B4B;
    }

    /* browse files button*/
    .css-1x8cf1d {
    color:#044389;
    border-color:#044389;
    }
    .css-1x8cf1d:hover {
    color:#FF4B4B;
    border-color:#FF4B4B;
    }

    /* translation*/
    h1 {
    color:#31333F !important;
    }

</style>       
""",unsafe_allow_html=True)

# banner
image = Image.open("banner.jpg")    
st.image(image)

# set text
st.write("<p style='font-size:26px;'>Bienvenue sur l'outil de traduction de la langue des signes</p>",
unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    st.write("<p style='font-size:18px;'>Le projet Main'terprète a été conçu en 2 semaines. Pour le moment, le modèle de traduction \
    n'a été conçu que pour une dizaine de mots mais nous travaillons activement à son amélioration.</p>",unsafe_allow_html=True)
    
st.markdown("")
st.markdown("")
st.markdown("")
st.subheader("Comment ça marche ?")
st.markdown("**C'est très simple, il te suffit de :**") 
st.markdown("1. filmer ton interlocuteur pendant qu'il signe un mot.  \n \
            Le sujet doit être à 2m de l'appareil et ses deux mains doivent être visibles,")
st.markdown("2. charger la vidéo sur le site,")
st.markdown("""3. appuyer sur le bouton "Lance Main'Terprète".""")
st.markdown("Nous nous occupons du reste !")
st.markdown("")

st.subheader("**Tu es prêt ? C'est par ici que ça se passe :point_down:**")

# upload file
st.markdown("Charge ta vidéo ici :")
col1, col2, col3 = st.columns([3,0.5,2])
with col1:
    uploaded_file = st.file_uploader(label = "yo", label_visibility = "collapsed")
    if uploaded_file is not None:

        # get temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False) 
        temp_file.write(uploaded_file.read())

        # display video
        st.video(temp_file.name)

# button to fetch the API
st.markdown("Appuie sur le bouton pour lancer la traduction : ")
if st.button("Lance Main'Terprète"):
    if uploaded_file is None:
        st.markdown("Tu n'as pas chargé de fichier au format demandé")

    elif uploaded_file is not None:
        with st.spinner('On te prépare ça...'):

            # instance of feature extractor
            holistic = mp_holistic.Holistic(
                static_image_mode = False,
                model_complexity = 2,
                refine_face_landmarks = True)

            video_current = cv2.VideoCapture(temp_file.name)

            # grab some parameters of video to use them for writing a new, processed video
            width = int(video_current.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_current.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_fps = video_current.get(cv2.CAP_PROP_FPS)

            # initialize variables to store landmarks
            pose_all = []
            face_all = []
            left_all = []
            right_all = []
        
            # initialize video writer to display landmarks on video    
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc_mp4, frame_fps,(width,height))

            # get landmarks per video frame
            while video_current.isOpened():
                success, image = video_current.read()
                if success:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)

                    # Draw landmark annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(
                        image,
                        results.right_hand_landmarks)
                    mp_drawing.draw_landmarks(
                        image,
                        results.left_hand_landmarks)

                    # write video
                    out.write(image)

                    # store pose
                    pose = []
                    if results.pose_landmarks is None:
                        continue
                    else:
                        for landmark in results.pose_landmarks.landmark:
                            pose.append(landmark.x)
                            pose.append(landmark.y)
                            pose.append(landmark.z)
                        pose_all.append(pose)

                    # store face
                    face = []
                    if results.face_landmarks is None:
                        continue
                    else:
                        for landmark in results.face_landmarks.landmark:
                            face.append(landmark.x)
                            face.append(landmark.y)
                            face.append(landmark.z)
                        face_all.append(face)

                    # store left hand
                    left = []
                    if results.left_hand_landmarks is None:
                        continue
                    else:
                        for landmark in results.left_hand_landmarks.landmark:
                            left.append(landmark.x)
                            left.append(landmark.y)
                            left.append(landmark.z)
                        left_all.append(left)

                    # store right hand
                    right = []
                    if results.right_hand_landmarks is None:
                        continue
                    else:
                        for landmark in results.right_hand_landmarks.landmark:
                            right.append(landmark.x)
                            right.append(landmark.y)
                            right.append(landmark.z)
                        right_all.append(right)
                
                # stop when video ends
                else:
                    video_current.release()
                    out.release()
                    break

            # store landmarks as dataframes
            pose_df = pd.DataFrame(pose_all)
            face_df = pd.DataFrame(face_all)
            left_df = pd.DataFrame(left_all)
            right_df = pd.DataFrame(right_all)


            # check if data exists
            if  (len(pose_all)==0) | (len(face_all)==0)|(len(left_all)==0)|(len(right_all)==0):
                st.error("Malheureusement nous ne sommes pas capables d'interpréter cette vidéo...  \n \
                         N'oublie pas qu'il faut que le sujet soit à 2m de l'appareil et que ses deux mains soient visibles.")
            else:
                # get full data
                data = pd.concat([pose_df, face_df, left_df, right_df], axis = 1)
                data.columns = range(0,data.shape[1])
            
            #Remove NaN from Data
            to_drop = []
            for i in range(data.shape[0]):
                if data.loc[i,:].isnull().any():
                    to_drop.append(i)
            data = data.drop(to_drop, axis = 0)

            # data length
            if data.shape[0] > 50:
                factor = int(np.ceil(data.shape[0] / 50))
                data = data.iloc[::factor,:]

            #Add padding
            pad_length = 50 - data.shape[0]
            pad = pd.DataFrame(0, index = range(pad_length), columns = data.columns)
            data = pd.concat([data, pad], axis = 0)
            data = data.to_numpy()
            data = data.reshape(1,50,1659)

            #Check is data still contains data
            if (data.shape[1] == 0) | (data.shape[2] < 1659):
                st.error("Malheureusement nous ne sommes pas capables d'interpréter cette vidéo...  \n \
                         N'oublie pas qu'il faut que le sujet soit à 2m de l'appareil et que ses deux mains soient visibles.")
            else:
                res = requests.post(url = "https://mainterprete-api.herokuapp.com/translate", json= {"data": data.tolist()})

                col1, _, col3 = st.columns([3,0.5,2])

                with col1:
                    # display video
                    st.subheader("Traitement de la vidéo")
                    video_file = open('output.mp4', 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                with col3:
                    st.subheader("Traduction")
                    st.title(res.content.decode("ascii").strip('"'))