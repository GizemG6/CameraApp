import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Eğitilmiş modeli yükleme
model = load_model("model.h5")

# Kamera bağlantısını başlatma
cap = cv2.VideoCapture(2) 

while True:
    # Kameradan bir frame al
    ret, frame = cap.read()

    # Frame'i boyutlandırma ve ön işleme yapma
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizasyon

    # Tahmin yapma
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_label = "Class_Label"  # Sınıf etiketlerini kendi veri setinize göre güncelleyin
    confidence = predictions[0, class_index]

    # Tahmin sonuçlarını ekrana yazdırma
    cv2.putText(frame, f"Class: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Frame'i gösterme
    cv2.imshow('Camera Stream', frame)

    # Çıkış için 'q' tuşuna basma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera bağlantısını serbest bırakma
cap.release()
cv2.destroyAllWindows()
