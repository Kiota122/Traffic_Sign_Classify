import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.optimizers import Adam


# Hàm tạo bộ dữ liệu
def load_data():
    data = []
    labels = []
    classes = 43
    cur_path = os.getcwd()

    # Truy xuất hình ảnh và nhãn
    for i in range(classes):
        path = os.path.join(cur_path, 'Train', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = cv2.imread(os.path.join(path, a))
                image = cv2.resize(image, (64, 64))
                data.append(image)
                labels.append(i)
            except Exception as e:
                print("Error loading image:", e)

    # Chuyển lists thành mảng numpy
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

# Tải hoặc tạo và lưu dữ liệu
if os.path.exists('train.pickle'):
    print("Loading data from pickle file...")
    with open('train.pickle', 'rb') as f:
        data, labels = pickle.load(f)
else:
    print("Loading data...")
    data, labels = load_data()
    print("Saving data to pickle file...")
    with open('train.pickle', 'wb') as f:
        pickle.dump((data, labels), f)

# Chia data thành bộ train, test và validate
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# In ra size của từng bộ
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
print("Validate set:", X_validation.shape, y_validation.shape)

# Chuyển đổi labels thành dạng one hot
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_validation = to_categorical(y_validation, 43)

def build_model(input_shape=(64,64,3), filter_size = (3,3), pool_size = (2, 2), output_size = 43):
    model = Sequential([
        Conv2D(16, filter_size, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(16, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=pool_size),
        Dropout(0.2),
        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    model.summary()
    return model

# Build model với kích thước đầu vào 64x64 và output là 43 classes
model = build_model(input_shape=(64,64,3), output_size=43)

# Khởi tạo điểm lưu model
filepath="weights-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

epochs = 30
batch_sizes = 16

# Xây dựng trình tạo ảnh huấn luyện để tăng cường dữ liệu
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.2,
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
aug_val = ImageDataGenerator(rescale=1./255)

# Huấn luyện mô hình
history = model.fit(
    aug.flow(X_train, y_train, batch_size=batch_sizes),
    epochs=epochs,
    validation_data=aug_val.flow(X_validation, y_validation),
    callbacks=[checkpoint]
)
model.save('my_model.keras')

# Vẽ đồ thị
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()