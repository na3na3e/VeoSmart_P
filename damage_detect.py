import torch
import streamlit as st
from PIL import Image
import cv2
import os
import shutil
from zipfile import ZipFile
from glob import glob
import pandas as pd


st.set_page_config(page_title="Veosmart Damage Detection", page_icon=':car:')
st.title("Veosmart Damage Detection")

# Add logo to sidebar
st.sidebar.image("veosmart.png", width=280)

# Function for model integration with Streamlit
def convert_df(df):
    '''
    Convert dataframe to CSV
    '''
    return df.to_csv().encode('utf-8')

def image_dt(uploaded_file, model):
    '''
    Detect damages using the damage detection model on images.
    '''
    img1 = model(os.path.join('data/images/', uploaded_file.name))
    df = img1.pandas().xyxy[0]
    img1 = img1.save(f'data/images/output/{uploaded_file.name}')
    # Ajouter une nouvelle colonne appelée "Nouvelle Colonne"
    #suggestions = ["Remplacer"]
    df["Pièce"]=""
    df["Suggestion"] = ""  # Ajouter une nouvelle colonne "Suggestion"
    df['Chiffrage'] = ""
#changement de nom pour les pieces
    for index, row in df.iterrows():
        class_value = row['class']  
        if class_value in [0]:            
            df.at[index, 'Pièce'] = 'Dommage au pare-brise avant'
        if class_value in [1]:            
            df.at[index, 'Pièce'] = 'Dommage aux phares'
        if class_value in [2]:            
            df.at[index, 'Pièce'] = "Grosse bosse à l'arrière du pare-chocs"
        if class_value in [3]:            
            df.at[index, 'Pièce'] = 'Dommage au pare-brise arrière'
        if class_value in [4]:            
            df.at[index, 'Pièce'] = 'Bosse sur le marchepied'
        if class_value in [5]:            
            df.at[index, 'Pièce'] = 'Dommage au rétroviseur'
        if class_value in [6]:            
            df.at[index, 'Pièce'] = 'Dommage aux feux de signalisation'
        if class_value in [7]:            
            df.at[index, 'Pièce'] = 'Dommage aux feux arrière'
        if class_value in [8]:            
            df.at[index, 'Pièce'] = 'Bosse sur le capot'
        if class_value in [9]:            
            df.at[index, 'Pièce'] = 'Bosse sur la porte extérieure'
        if class_value in [10]:            
            df.at[index, 'Pièce'] = "Bosse sur l'aile"
        if class_value in [11]:            
            df.at[index, 'Pièce'] = 'Bosse sur le pare-chocs avant'
        if class_value in [12]:            
            df.at[index, 'Pièce'] = 'Bosse sur le panneau de carrosserie moyen'
        if class_value in [13]:            
            df.at[index, 'Pièce'] = "Bosse sur le pilier"
        if class_value in [14]:            
            df.at[index, 'Pièce'] = "Bosse sur l'aile"
        if class_value in [15]:            
            df.at[index, 'Pièce'] = 'Bosse sur le pare-chocs arrière'
        if class_value in [16]:            
            df.at[index, 'Pièce'] = 'Bosse sur le toit'


#estimation du choix de vie de la piece 
    for index, row in df.iterrows():
        class_value = row['class']
        if class_value in [0, 1, 2, 3, 5, 6, 7]:            
            df.at[index, 'Suggestion'] = 'Remplacer'
        elif class_value in [4, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            df.at[index, 'Suggestion'] = 'Réparer'


#estimation du prix 
    for index, row in df.iterrows():
        class_value = row['class']  
        if class_value in [0]:            
            df.at[index, 'Chiffrage'] = 1500
        if class_value in [1, 2,3]:            
            df.at[index, 'Chiffrage'] = 650
        if   class_value in [5, 6, 7]:            
            df.at[index, 'Chiffrage'] = 500
        if class_value in [ 8, 9, 10, 11]:            
            df.at[index, 'Chiffrage'] = 450
        if   class_value in [12, 14, 15, 16]:            
            df.at[index, 'Chiffrage'] = 500
        if   class_value in [4,13]:            
            df.at[index, 'Chiffrage'] = 200



    total_chiffrage = df['Chiffrage'].sum()
    df.loc['Total'] = [''] * 9 + [total_chiffrage]

    #supprimer names
    df.drop(columns=['name'], inplace=True)

    return df

def video_dt(uploaded_file, model):
    '''
    Detect damages using the damage detection model on videos.
    '''
    vid = cv2.VideoCapture(os.path.join("data/videos", uploaded_file.name))
    i = 0
    while True:
        hasFrames, image = vid.read()
        if not hasFrames:
            break  # no more frames to read
        
        if i % 1 == 0:
            re = model(image[..., ::-1])
            if re is not None:
                re.save(f'data/images/output/{uploaded_file.name}')
                image_path = os.path.join('result', f'im{i}.jpg')
                shutil.copy(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime), "image0.jpg"), image_path)
                print(f"Result image saved to {image_path}")

            
        i += 1
        print(i)
        if i >= 1000:
            break  # limit to 1000 frames
        
    images = [img for img in os.listdir('result') if img.endswith(".jpg")]
    temp_vid = uploaded_file.name.split('.')[0] + '.webm'
    video = cv2.VideoWriter(os.path.join('vid', temp_vid), fourcc=cv2.VideoWriter_fourcc(*'vp80'), fps=5, frameSize=(1280,720), isColor=True)
    for i in images:
        im = cv2.imread(os.path.join('result', i))
        im = cv2.resize(im, (1280,720))
   
        video.write(im)
        os.remove(os.path.join('result', i))

    video.release()

    # Create a zip file of the video for downloading
    old_path = os.getcwd()
    os.chdir('zipr')
    shutil.make_archive(uploaded_file.name.split('.')[0], 'zip', root_dir=os.path.join('vid', temp_vid))
    os.chdir(old_path)
    with ZipFile(os.path.join('zipr', uploaded_file.name.split('.')[0] + '.zip'), "w") as newzip:
        newzip.write(os.path.join('vid', temp_vid))

   

#-----Uploading Picture and video------# 

def multiple_image_dt(uploaded_files, model):
    '''
    Detect damages using the damage detection model on multiple images.
    '''
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img.save(f'data/images/{uploaded_file.name}')
        df = image_dt(uploaded_file, model)
        temp_img = uploaded_file.name.split('.')[0] + '.jpg'
        st.image(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime), temp_img))
        st.table(df)
        csv = convert_df(df)
        st.download_button("Download CSV", csv, "file.csv", "text/csv", key='download-csv')
        with open(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime), temp_img), "rb") as file:
            st.download_button("Download Image", data=file.read(), file_name='img.jpg', mime="image/jpeg")

def upload(model):
    source = ("Image Detection", "Video Detection", "Multiple Image Detection")
    confidence_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)
    model.conf = confidence_threshold  # Set confidence threshold for the model
    source_index = st.sidebar.selectbox("Select Input", range(len(source)), format_func=lambda x: source[x])
    uploaded_file = None
    uploaded_files = None
    
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            with st.spinner('Resource loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture.save(f'data/images/{uploaded_file.name}')

    elif source_index == 1:
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
        if uploaded_file is not None:
            with st.spinner('Resource loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data/videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
    
    elif source_index == 2:
        uploaded_files = st.sidebar.file_uploader("Upload Multiple Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files is not None:
            with st.spinner('Loading images...'):
                multiple_image_dt(uploaded_files, model)

    if st.button("Detect"):
        if source_index == 0:
            if uploaded_file is None:
                st.warning("Veuillez sélectionner une image.")
            else:
                with st.spinner('Detecting...'):
                    df = image_dt(uploaded_file, model)
                temp_img = uploaded_file.name.split('.')[0] + '.jpg'
                st.image(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime), temp_img))
                st.table(df)
                csv = convert_df(df)
                st.download_button("Download CSV", csv, "file.csv", "text/csv", key='download-csv')
                with open(os.path.join(max(glob("runs/detect/*/"), key=os.path.getmtime), temp_img), "rb") as file:
                    st.download_button("Download Image", data=file.read(), file_name='img.jpg', mime="image/jpeg")

        elif source_index == 1:
            if uploaded_file is None:
                st.warning("Veuillez sélectionner une vidéo.")
            else:
                with st.spinner('Detecting...'):
                    vid = video_dt(uploaded_file, model)
                temp_vid = uploaded_file.name.split('.')[0] + '.webm'
                temp_zip = uploaded_file.name.split('.')[0] + '.zip'
                video_file = open(os.path.join('vid', temp_vid), 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                with open(os.path.join('zipr', temp_zip), "rb") as fp:
                    st.download_button("Download Video in Zip format", data=fp.read(), file_name='result.zip', mime="application/zip")
            
        elif source_index == 2:
            if uploaded_files is None:
                st.warning("Veuillez sélectionner des images.")
            else:
                with st.spinner('Detecting...'):
                    multiple_image_dt(uploaded_files, model)



def pg1():

    # Using pytorch hub inference of Yolov5 for loading model
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"model/dam_det.pt", force_reload=True)
    model.conf = 0.5
    upload(model)
    pass



page_names_to_funcs = {
    "Vehicle Damage Detection": pg1,
}

selected_page = st.sidebar.radio("", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()