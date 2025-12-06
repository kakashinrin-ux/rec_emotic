import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------- Parámetros ----------
LOCAL_MODEL = "fer2013_cnn.h5"
EMOJI_FOLDER = "Emojis"
EMOTIONS = ['Enojado', 'Disgustado', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# ---------- 1. Verificar modelo ----------
if not os.path.exists(LOCAL_MODEL):
    print(f"No se encontró el modelo {LOCAL_MODEL}. Coloca el archivo .h5 en la carpeta del proyecto.")
    raise SystemExit

# ---------- 2. Cargar modelo ----------
try:
    model = load_model(LOCAL_MODEL, compile=False)
    print("Modelo cargado correctamente.")
except Exception as e:
    print("Error al cargar el modelo .h5:", e)
    raise SystemExit

# ---------- 3. Detector de rostros ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Cargar emoji ----------
def load_emoji(emotion_name):
    path = os.path.join(EMOJI_FOLDER, emotion_name.lower() + ".png")
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None

# ---------- 4. Captura de video ----------
cap = cv2.VideoCapture(0)

# Tamaño del emoji pequeño
EMOJI_SIZE = 300

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]  # tamaño real de la cámara

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Panel del emoji (más pequeño)
    emoji_panel = np.zeros((EMOJI_SIZE, EMOJI_SIZE, 3), dtype=np.uint8)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]

        # Dibujar rectángulo y emoción
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cargar emoji
        emoji = load_emoji(emotion)
        if emoji is not None:
            emoji = cv2.resize(emoji, (EMOJI_SIZE, EMOJI_SIZE))

            if emoji.shape[2] == 4:  # BGRA → BGR
                emoji = cv2.cvtColor(emoji, cv2.COLOR_BGRA2BGR)

            emoji_panel = emoji

    # ---------- Crear panel del emoji con padding para igualar altura ----------
    # Se rellenan bordes arriba y abajo sin deformar: cámara queda igual.
    pad_top = (frame_height - EMOJI_SIZE) // 2
    pad_bottom = frame_height - (EMOJI_SIZE + pad_top)

    emoji_padded = cv2.copyMakeBorder(
        emoji_panel,
        pad_top,
        pad_bottom,
        0, 0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # ---------- Concatenar sin deformar la cámara ----------
    nFrame = cv2.hconcat([frame, emoji_padded])

    cv2.imshow("Emociones", nFrame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
