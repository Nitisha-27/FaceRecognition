# Face Recognition Attendance System (Improved Version)
import cv2
import os
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
from tkinter import simpledialog
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import hashlib
import datetime

# === Utility Functions ===
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# === GUI Setup ===
window = tk.Tk()
window.title("Face Recognition Based Attendance System")
window.geometry('800x600')
window.configure(bg='#f0f0f0')
window.resizable(False, False)

# === Paths ===
data_dir = os.getcwd()
training_image_path = os.path.join(data_dir, 'TrainingImage')
student_details_path = os.path.join(data_dir, 'StudentDetails')
training_label_path = os.path.join(data_dir, 'TrainingImageLabel')
attendance_path = os.path.join(data_dir, 'Attendance')

for path in [training_image_path, student_details_path, training_label_path, attendance_path]:
    assure_path_exists(path)

psd_file = os.path.join(training_label_path, 'psd.txt')
if not os.path.isfile(psd_file):
    with open(psd_file, 'w') as f:
        f.write(hash_password("admin"))

# === Components ===
message = tk.Label(window, text="Face Recognition Attendance System", bg="#1f1f2e", fg="white",
                   font=('Helvetica', 20, 'bold'))
message.pack(pady=20)

frame = tk.Frame(window)
frame.pack(pady=10)

lbl = tk.Label(frame, text="Enter ID", width=20, font=('Helvetica', 12))
lbl.grid(row=0, column=0, padx=10)
txt_id = tk.Entry(frame, width=30)
txt_id.grid(row=0, column=1, padx=10)

lbl2 = tk.Label(frame, text="Enter Name", width=20, font=('Helvetica', 12))
lbl2.grid(row=1, column=0, padx=10)
txt_name = tk.Entry(frame, width=30)
txt_name.grid(row=1, column=1, padx=10)

message_display = tk.Label(window, text="", bg="#f0f0f0", fg="green", font=('Helvetica', 12))
message_display.pack(pady=10)

# === Functions ===
def take_images():
    Id = txt_id.get().strip()
    name = txt_name.get().strip()

    if not Id or not name:
        message_display.configure(text='Please enter both ID and Name.', fg='red')
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    sample_num = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_num += 1
            face_img = gray[y:y + h, x:x + w]
            file_path = os.path.join(training_image_path, f"{name}.{Id}.{sample_num}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Capturing Faces', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sample_num >= 60:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(os.path.join(student_details_path, 'StudentDetails.csv'), 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([Id, name])

    message_display.configure(text='Images Saved for ID: ' + Id, fg='green')


def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces, Ids = [], []
    for image_file in os.listdir(training_image_path):
        img_path = os.path.join(training_image_path, image_file)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, 'uint8')
        try:
            id_val = int(image_file.split('.')[1])
        except:
            continue
        faces.append(img_np)
        Ids.append(id_val)

    if not faces:
        message_display.configure(text='No faces found for training.', fg='red')
        return

    recognizer.train(faces, np.array(Ids))
    recognizer.save(os.path.join(training_label_path, 'Trainner.yml'))
    message_display.configure(text='Training Completed Successfully!', fg='green')


def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(os.path.join(training_label_path, 'Trainner.yml'))
    except:
        message_display.configure(text='Please train data first.', fg='red')
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    df = {}
    try:
        with open(os.path.join(student_details_path, 'StudentDetails.csv')) as f:
            reader = csv.reader(f)
            for row in reader:
                df[int(row[0])] = row[1]
    except:
        message_display.configure(text='StudentDetails.csv not found!', fg='red')
        return

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = set()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 60:
                name = df.get(id_, 'Unknown')
                attendance.add((id_, name))
                cv2.putText(img, f"{name} [{conf:.2f}]", (x, y-10), font, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y-10), font, 0.6, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Marking Attendance', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(attendance_path, f"Attendance_{ts}.csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Name', 'Time'])
        for entry in attendance:
            writer.writerow([entry[0], entry[1], datetime.datetime.now().strftime("%H:%M:%S")])

    message_display.configure(text=f"Attendance marked for {len(attendance)} students.", fg='green')


# === Buttons ===
button_frame = tk.Frame(window, bg='#f0f0f0')
button_frame.pack(pady=20)

take_img_btn = ttk.Button(button_frame, text="Take Images", command=take_images)
take_img_btn.grid(row=0, column=0, padx=10)

train_img_btn = ttk.Button(button_frame, text="Train Images", command=train_images)
train_img_btn.grid(row=0, column=1, padx=10)

track_img_btn = ttk.Button(button_frame, text="Track Images", command=mark_attendance)
track_img_btn.grid(row=0, column=2, padx=10)

quit_btn = ttk.Button(button_frame, text="Quit", command=window.destroy)
quit_btn.grid(row=0, column=3, padx=10)

# === Mainloop ===
window.mainloop()
