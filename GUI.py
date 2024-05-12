import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image, ImageFont, ImageDraw
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
# Load model đã được huấn luyện
from keras.models import load_model

model = load_model('weights-22-1.00.keras')

#  Tạo từ điển dán nhãn tất cả các lớp biển báo giao thông.
classes = {0: 'Giới hạn tốc độ (20km/h)',
           1: 'Giới hạn tốc độ (30km/h)',
           2: 'Giới hạn tốc độ (50km/h)',
           3: 'Giới hạn tốc độ (60km/h)',
           4: 'Giới hạn tốc độ (70km/h)',
           5: 'Giới hạn tốc độ (80km/h)',
           6: 'Hết hạn giới hạn tốc độ (80km/h)',
           7: 'Giới hạn tốc độ (100km/h)',
           8: 'Giới hạn tốc độ (120km/h)',
           9: 'Cấm vượt',
           10: 'Cấm vượt phương tiện trên 3.5 tấn',
           11: 'Quyền ưu tiên tại ngã tư',
           12: 'Đường ưu tiên',
           13: 'Nhường đường',
           14: 'Dừng',
           15: 'Không có phương tiện',
           16: 'Phương tiện > 3.5 tấn cấm',
           17: 'Cấm đi vào',
           18: 'Chú ý chung',
           19: 'Đường cong trái nguy hiểm',
           20: 'Đường cong phải nguy hiểm',
           21: 'Đường cong đôi',
           22: 'Đường gập ghềnh',
           23: 'Đường trơn trượt',
           24: 'Đường hẹp bên phải',
           25: 'Công trường',
           26: 'Đèn tín hiệu giao thông',
           27: 'Người đi bộ',
           28: 'Người đi bộ qua đường',
           29: 'Xe đạp qua đường',
           30: 'Cảnh báo sương muối',
           31: 'Động vật hoang dã qua đường',
           32: 'Hết hạn giới hạn tốc độ + vượt',
           33: 'Rẽ phải phía trước',
           34: 'Rẽ trái phía trước',
           35: 'Chỉ đi thẳng',
           36: 'Chỉ đi thẳng hoặc rẽ phải',
           37: 'Chỉ đi thẳng hoặc rẽ trái',
           38: 'Đi bên phải',
           39: 'Đi bên trái',
           40: 'Đường vòng bắt buộc',
           41: 'Hết hạn cấm vượt',
           42: 'Hết hạn cấm vượt phương tiện > 3.5 tấn'}

def classify(file_path):
    # Chuyển đổi kích cỡ
    image = cv2.imread(file_path)
    image = cv2.resize(image, dsize=(64, 64))
    image = image.astype('float') * 1. / 255
    # Chuyển đổi sang numpy array và mở rộng chiều
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    print(image.shape)
    # Dự đoán
    pred_probabilities = model.predict(image)
    pred = np.argmax(pred_probabilities, axis=-1)
    detected_sign = classes[pred[0]]

    print(detected_sign)
    label.configure(foreground='#011638', text=detected_sign)
    text_to_speech(detected_sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = cv2.imread(file_path)
        uploaded = cv2.resize(uploaded, ((top.winfo_width() // 2), (top.winfo_height() // 2)))
        cv2.imwrite("temp.png", uploaded)
        im = Image.open("temp.png")
        im = ImageTk.PhotoImage(im)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

# Mở camera và hiển thị hình ảnh trực tiếp lên màn hình
def open_camera():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera")
            break

        # Chuyển đổi frame thành hình ảnh có kích thước 64x64 để phù hợp với mô hình
        resized_frame = cv2.resize(frame, dsize=(64, 64))
        resized_frame = resized_frame.astype('float') * 1. / 255

        # Chuẩn bị frame để đưa vào mô hình (chuyển đổi sang numpy array và mở rộng chiều)
        input_frame = np.expand_dims(resized_frame, axis=0)

        # Dự đoán lớp của biển báo giao thông
        pred_probabilities = model.predict(input_frame)
        pred = np.argmax(pred_probabilities, axis=-1)
        detected_sign = classes[pred[0]]

        if (np.max(pred_probabilities) >= 0.9) and (np.argmax(pred_probabilities[0]) != 0):
            # Chuyển đổi frame thành hình ảnh RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Sử dụng font Arial với kích thước 40 và màu xanh
            font = ImageFont.truetype("arial.ttf", 40)
            draw = ImageDraw.Draw(pil_image)
            draw.text((50, 50), detected_sign, font=font, fill=(0, 255, 0))

            # Chuyển đổi hình ảnh từ PIL Image thành numpy array và chuyển đổi lại sang định dạng BGR
            frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('Camera Feed', frame_with_text)
        else:
            cv2.imshow('Camera Feed', frame)

        # Nhận diện biển báo giao thông trên hình ảnh từ camera
        print(detected_sign)

        # Thoát khỏi vòng lặp khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên camera và cửa sổ
    cap.release()
    cv2.destroyAllWindows()

def text_to_speech(text, lang='vi'):
    tts = gTTS(text=text, lang=lang)
    tts.save("traffic sign.mp3")

    # Phát âm thanh
    playsound("traffic sign.mp3")

# Tạo GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông ')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

button_frame = Frame(top)
button_frame.pack(side=BOTTOM, pady=50)

upload = Button(button_frame, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=LEFT, padx=10)

camera_button = Button(button_frame, text="Mở camera", command=open_camera, padx=10, pady=5)
camera_button.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
camera_button.pack(side=LEFT, padx=10)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Phân loại biển báo giao thông bằng DeepLearning (CNN)", pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')

heading1 = Label(top, text="Nguyễn Đăng Tạ Khôi MSSV: 2020604232", pady=5, font=('arial', 20, 'bold'))
heading1.configure(background='#ffffff', foreground='#364156')

heading.pack()
heading1.pack()
top.mainloop()