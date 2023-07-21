import os
import urllib.parse
import requests
import json
from PIL import Image
from io import BytesIO
from fastapi import FastAPI
import torch
from PIL import ImageDraw

app = FastAPI()


def detect_damage(image, model):
    '''
    Detect damages using the damage detection model on an image.
    '''
    class_names = ['class1', 'class2', 'class3']  # Replace with your actual class names

    # Apply your image detection logic with your model here
    output = model(image)  # Your image detection code goes here

    # Process the detection output and extract relevant information
    detections = []
    for detection in output.xyxy[0]:
        xmin, ymin, xmax, ymax, class_id, confidence = detection.tolist()
        class_name = class_names[int(class_id)]
        detection_info = {"class": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        detections.append(detection_info)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red")

    # Show the image with the bounding boxes
    image.show()

    return detections



@app.get("/detect_damage/")
async def detect_damage_api(image_url: str):
    # Téléchargement de l'image à partir de l'URL
    response = requests.get(image_url)
    image_data = response.content

    # Extraction du nom de fichier de l'URL
    parsed_url = urllib.parse.urlparse(image_url)
    image_filename = os.path.basename(parsed_url.path)

    # Chargement de votre modèle de détection d'image
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"model/dam_det.pt", force_reload=True)
    model.conf = 0.5

    # Convertir les données de l'image en format PIL
    image_pil = Image.open(BytesIO(image_data))

    # Appliquer la détection d'image sur l'image téléchargée
    detections = detect_damage(image_pil, model)

    # Enregistrer l'image avec les détections dans un fichier
    output_image_path = f"resu_api/{image_filename}"
    image_with_detections = model.show(image_pil, detections=detections)
    image_with_detections.save(output_image_path)

    # Enregistrer les détections dans un fichier JSON
    output_json_path = f"output/{image_filename}.json"
    with open(output_json_path, "w") as output_json_file:
        json.dump(detections, output_json_file)

    return {
        "image_url": image_url,
        "image_with_detections": output_image_path,
        "detections": detections,
        "json_path": output_json_path,
    }
