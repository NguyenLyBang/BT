import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === 1. Tạo mô hình CNN với BatchNormalization ===
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === 2. Lưu và tải mô hình ===
def save_model(model, path="model.h5"):
    model.save(path)

def load_trained_model(path="model.h5"):
    return load_model(path)

# === 3. Huấn luyện mô hình ===
def train_model(train_path, val_path, model_save_path="model.h5"):
    # Data Augmentation
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ).flow_from_directory(
        train_path, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical'
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_path, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical'
    )

    model = build_model()
    history = model.fit(train_gen, validation_data=val_gen, epochs=30)
    save_model(model, model_save_path)
    return model, history

# === 4. Đánh giá mô hình ===
def evaluate_model(model, test_path):
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_path, target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical', shuffle=False
    )

    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen), axis=1)

    print("/nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys(), zero_division=1))

    print("/nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Vẽ biểu đồ Confusion Matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

# === 5. Dự đoán cảm xúc từ ảnh ===
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def get_emotion_label(index):
    return emotion_dict.get(index, "Unknown")

# === 6. Tiền xử lý ảnh ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        return face
    return None

# === 7. Camera và nhận diện cảm xúc ===
def start_camera(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Lật ngang
        face = detect_and_preprocess(frame)
        if face is not None:
            face = face.reshape(1, 48, 48, 1) / 255.0
            prediction = model.predict(face)
            emotion = np.argmax(prediction)
            emotion_label = get_emotion_label(emotion)
            cv2.putText(frame, f"Emotion: {emotion_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# === 8. Chạy chương trình chính ===
if __name__ == "__main__":
    # Đường dẫn tới dữ liệu
    train_path = "D:/NLB/XLA & TGMT/BTL/NDCX/fer2013plus/train"
    val_path = "D:/NLB/XLA & TGMT/BTL/NDCX/fer2013plus/val"
    test_path = "D:/NLB/XLA & TGMT/BTL/NDCX/fer2013plus/test"

    # Huấn luyện hoặc tải mô hình
    train_new_model = False  # True nếu cần huấn luyện lại

    if train_new_model:
        model, history = train_model(train_path, val_path)
    else:
        model = load_trained_model("model.h5")

    # Đánh giá mô hình
    evaluate_model(model, test_path)

    # Khởi động camera để nhận diện cảm xúc
    start_camera(model)
