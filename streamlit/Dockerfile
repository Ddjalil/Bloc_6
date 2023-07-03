FROM python:3.9
COPY . /home/app

WORKDIR /home/app

RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

RUN pip install -r requirements.txt
CMD streamlit run --server.port $PORT Accueil.py