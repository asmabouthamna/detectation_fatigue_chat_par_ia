import cv2
import tensorflow as tf
import numpy as np
import base64
import paho.mqtt.client as mqtt
from websocket import create_connection

# Charger le modèle de classification
modele_classification = tf.keras.models.load_model('linear_model.keras')  # Remplacez par votre modèle de classification

# Paramètres MQTT
broker = "192.168.1.27"  # Adresse du broker MQTT
port = 1883  # Le port par défaut pour MQTT
topic = "camera/photo"  # Le topic sur lequel l'image a été envoyée
output_image_path = "received_photo.jpg"  # Le chemin où enregistrer l'image reçue

# Paramètres WebSocket
ws_url = "ws://localhost:8765"  # L'adresse du serveur WebSocket
ws = create_connection(ws_url)


# Fonction pour prédire l'état de santé du chat
def predire_etat_chat(image):
    try:
        image_redimensionnee = cv2.resize(image, (128, 128))
        image_normalisee = image_redimensionnee / 255.0
        image_prete = np.expand_dims(image_normalisee, axis=0)

        prediction = modele_classification.predict(image_prete)
        prob = float(prediction[0][0])
        print(f"Prédiction brute : {prob}")

        if prob > 0.4:
            label = "Chat malade"
            color = (0, 0, 255)
        else:
            label = "Chat en bonne santé"
            color = (0, 255, 0)

        print(f"Résultat de la classification : {label}")
        return label, color
    except Exception as e:
        print(f"Erreur lors de la prédiction de l'état du chat : {e}")
        return "Erreur", (0, 0, 255)

# Fonction pour traiter l'image et envoyer via WebSocket
def traiter_image(image_path):
    try:
        # Charger l'image pour la prédiction
        image = cv2.imread(image_path)
        label, color = predire_etat_chat(image)

        # Ajouter un label sur l'image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (10, 30), font, 1, color, 2, cv2.LINE_AA)

        # Sauvegarder l'image avec le label
        cv2.imwrite("processed_" + image_path, image)
        print(f"Image traitée et enregistrée sous processed_{image_path}")
    except Exception as e:
        print(f"Erreur lors du traitement de l'image : {e}")

# Fonction pour envoyer une image via WebSocket
def envoyer_image_websocket(image_path):
    try:
        # Charger et convertir l'image en base64
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Créer un message JSON avec l'image en base64
        message = {
            "type": "image",
            "data": f"data:image/jpeg;base64,{img_base64}"
        }

        # Envoyer l'image en base64 via WebSocket
        ws.send(str(message))
        print("Image envoyée via WebSocket")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'image via WebSocket : {e}")

# Fonction appelée lors de la connexion au broker MQTT
def on_connect(client, userdata, flags, rc):
    print(f"Connecté au broker avec le code {rc}")
    client.subscribe(topic)
    print(f"S'abonne au topic {topic}")

# Fonction appelée lorsqu'un message est reçu sur le topic
def on_message(client, userdata, msg):
    print(f"Message reçu sur le topic {msg.topic}")
    try:
        image_data = base64.b64decode(msg.payload)
        with open(output_image_path, "wb") as img_file:
            img_file.write(image_data)
        print(f"Image reçue et enregistrée sous {output_image_path}")

        # Traiter l'image et envoyer via WebSocket
        traiter_image(output_image_path)
        envoyer_image_websocket("processed_" + output_image_path)
    except Exception as e:
        print(f"Erreur lors du traitement de l'image reçue : {e}")

# Initialisation du client MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connexion au broker MQTT
try:
    client.connect(broker, port, 60)
    client.loop_forever()
except Exception as e:
    print(f"Erreur lors de la connexion au broker MQTT : {e}")
