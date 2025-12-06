import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------- Parámetros ----------
LOCAL_MODEL = "fer2013_cnn.h5"  # Tu modelo preentrenado
EMOJI_FOLDER = "Emojis"         # Carpeta con emojis
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

# ---------- 3. Inicializar detector de rostros (Haar Cascade) ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Función para cargar emoji ----------
def load_emoji(emotion_name):
    path = os.path.join(EMOJI_FOLDER, emotion_name.lower() + ".png")
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None

# ---------- 4. Captura de video ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]

        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostrar emoji si existe
        emoji = load_emoji(emotion)
        if emoji is not None:
            emoji = cv2.resize(emoji, (w, h))
            if emoji.shape[2] == 4:  # canal alfa
                alpha = emoji[:, :, 3] / 255.0
                for c in range(3):
                    frame[y:y+h, x:x+w, c] = (alpha * emoji[:, :, c] +
                                              (1-alpha) * frame[y:y+h, x:x+w, c])
            else:
                frame[y:y+h, x:x+w] = emoji

    cv2.imshow("Emociones", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()

