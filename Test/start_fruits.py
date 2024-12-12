import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp

# Ruta de la carpeta de modelos
MODELS_DIR = "C:/Users/fabri/Documents/GitHub/linkproject_ia/h5_models"



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Función para cargar un modelo
def load_trained_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo {model_name} no existe en la ruta {MODELS_DIR}.")
    return load_model(model_path)

# Funciones de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

# Función principal para probar un modelo
def test_model(model_name):
    # Cargar modelo
    model = load_trained_model(model_name)
    print(f"Modelo {model_name} cargado con éxito.")

    # Configurar la captura de video
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detección de puntos clave
            image, results = mediapipe_detection(frame, holistic)

            # Dibujar puntos clave
            draw_styled_landmarks(image, results)

            # Preprocesar y predecir
            keypoints = extract_keypoints(results)
            keypoints = np.expand_dims(keypoints, axis=0)  # Agregar dimensión para el modelo
            prediction = model.predict(keypoints)
            action = np.argmax(prediction)  # Obtener la acción predicha

            # Mostrar la predicción
            cv2.putText(image, f"Predicción: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar en pantalla
            cv2.imshow('Feed de Video', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

if __name__ == "__main__":
    # Cambiar el nombre del modelo según sea necesario
    model_name = "modelprueba1.h5"  # Cambia a 'model2.h5' o 'model3.h5' para probar otros modelos
    test_model(model_name)
