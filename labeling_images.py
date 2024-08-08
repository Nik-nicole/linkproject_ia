""" Drawing funtions """

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (16, 117, 245),
    (16, 117, 245),
    (16, 117, 245),
    (16, 117, 245),
    (16, 117, 245),
    (16, 117, 245),
    (16, 117, 245),
]


def mediapipe_detection(image, model):
    """
    Procesa una imagen con un modelo de MediaPipe y devuelve la imagen en
    formato BGR junto con los resultados.
    Args:
        image (numpy.ndarray): Imagen en formato BGR.
        model (mediapipe model): Modelo de MediaPipe para procesar la imagen.
    Returns:
        tuple: Imagen en formato BGR y resultados del modelo.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    """
    Dibuja los puntos de referencia en la imagen utilizando los resultados del modelo.
    Args:
        image (ndarray): La imagen en la que se dibujarán los puntos de referencia.
        results (object): Un objeto de resultados del modelo que contiene las
                          landmarks faciales, de pose y de manos.
    Returns:
        None
    """
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def draw_styled_landmarks(image, results):
    """
    Dibuja los puntos de referencia en la imagen con un estilo específico.
    Args:
        image (ndarray): La imagen en la que se dibujarán los puntos de referencia.
        results (object): Un objeto de resultados del modelo que contiene las
                          landmarks faciales, de pose y de manos.

    Returns:
        None
    Notes:
        - Los puntos de referencia faciales se dibujan con un estilo verde
        para los puntos y un estilo amarillo para las conexiones.
        - Los puntos de referencia de pose se dibujan con un estilo rojo
        para los puntos y un estilo azul para las conexiones.
        - Los puntos de referencia de la mano izquierda se dibujan
        con un estilo rosa para los puntos y un estilo púrpura para las conexiones.
        - Los puntos de referencia de la mano derecha se dibujan
        con un estilo naranja para los puntos y un estilo rosa para las conexiones.
    """
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10),
                               thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121),
                               thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121),
                               thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76),
                               thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250),
                               thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66),
                               thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230),
                               thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    """
    Extrae los puntos clave de las landmarks faciales,
    de pose y de las manos desde el objeto de resultados.
    Args:
        results (object): Un objeto de resultados del modelo
        que contiene las landmarks faciales, de pose y de manos.
    Returns:
        numpy.ndarray: Un arreglo unidimensional que contiene las coordenadas de los
        puntos clave extraídos.
                       - Pose: 33 puntos con 4 valores cada uno (x, y, z, visibilidad).
                       - Face: 468 puntos con 3 valores cada uno (x, y, z).
                       - Left Hand: 21 puntos con 3 valores cada uno (x, y, z).
                       - Right Hand: 21 puntos con 3 valores cada uno (x, y, z).
                       Si no se encuentran landmarks en alguna de las categorías,
                       se rellenará con ceros.
    Notes:
        - La función devuelve un arreglo concatenado con los puntos de referencia
        en el siguiente orden:
          [pose, face, left_hand, right_hand].
        - La longitud del arreglo resultante será 33*4 + 468*3 + 21*3 + 21*3,
        que corresponde a los puntos de referencia de las cuatro categorías.
    """
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z]
                for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z]
                for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])
