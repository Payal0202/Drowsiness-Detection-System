import numpy as np
import cv2
from matplotlib import pyplot as plt
import dlib
from imutils import face_utils
import winsound
import tkinter as tk
from PIL import Image, ImageTk
import ctypes


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/usre/Desktop/MT Project/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")


sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
face_frame = None


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0


# Custom function for histogram equalization
def custom_equalize_hist(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf.max()  # Normalize to the maximum intensity value
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    # Reshape the equalized image to the original shape
    equalized_image = equalized_image.reshape(image.shape)
    equalized_image = np.clip(equalized_image, 0, 255).astype(np.uint8)
    return equalized_image


# Custom function for canny edge detection
def canny_edge_detection(image, sigma=0.2):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Compute the median of the grayscale pixel intensities
    v = np.median(blurred)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)

    return edges


def update_frames():
    global sleep, drowsy, active, status, color, face_frame, update_id

    _, frame = cap.read()
    # Sharpen the color image
    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])
    
    sharpened_frame =cv2.filter2D(frame, -1, kernel)

    # Convert the sharpened color image to grayscale
    gray = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2GRAY)
   
    # Apply histogram equalization to the grayscale image
    gray_equalized = custom_equalize_hist(gray)
    # gray_equalized = cv2.equalizeHist(gray)

    edges = canny_edge_detection(gray_equalized)
    
    faces = detector(gray_equalized)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = gray_equalized.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray_equalized, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], 
                              landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], 
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        mouth_aspect_ratio = (compute(landmarks[61], landmarks[67]) + compute(landmarks[62], landmarks[66]) + compute(landmarks[63], landmarks[65])) / (3 * compute(landmarks[60], landmarks[64]))
        mouth_yawned = mouth_aspect_ratio > 0.2

        if left_blink == 0 or right_blink == 0 or mouth_yawned:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 4:
                status = "SLEEPING:("
                color = (255, 0, 0)
                winsound.Beep(1000, 200)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 4:
                status = "DROWSY!!"
                color = (0, 0, 255)
                winsound.Beep(1000, 200)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 4:
                status = "ACTIVE:)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        # cv2.putText(face_frame, "Face Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # cv2.putText(gray, "Gray Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # cv2.putText(sharpened_frame, "Sharpened Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 0), 2)
        # cv2.putText(gray_equalized, "Histogram Equalization Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
       
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if face_frame is not None:
        face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)# Convert the sharpened color image to RGB format
        sharpened_frame_rgb = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

        # Convert the sharpened color image to ImageTk format
        
        frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb).resize((300,300)))
        gray_tk = ImageTk.PhotoImage(image=Image.fromarray(gray_rgb).resize((300,300)))
        face_frame_tk = ImageTk.PhotoImage(image=Image.fromarray(face_frame_rgb).resize((300,300)))
        sharpened_frame_tk = ImageTk.PhotoImage(image=Image.fromarray(sharpened_frame_rgb).resize((300,300))) 
        edges_tk = ImageTk.PhotoImage(image=Image.fromarray(edges_rgb).resize((270,300)))


        label1.config(image=frame_tk)
        label1.image = frame_tk

        label2.config(image=gray_tk)
        label2.image = gray_tk

        label3.config(image=face_frame_tk)
        label3.image = face_frame_tk

        label4.config(image=sharpened_frame_tk)
        label4.image = sharpened_frame_tk

        label5.config(image=edges_tk)
        label5.image = edges_tk
        
        # Add text below the frames box
        label1_text.config(text="Original Frame")
        label2_text.config(text="Gray Frame")
        label3_text.config(text="Histogram Equalization Frame")
        label4_text.config(text="Sharpened Frame")
        label5_text.config(text="Canny Edge Detection Frame")
        

    cv2.imwrite('edges.jpg', edges)
    
    update_id=root.after(10, update_frames)


root = tk.Tk()
root.title("Drowsiness Detection System")

label1 = tk.Label(root, bg="white")
label1.grid(row=0, column=0, padx=5, pady=5)

label2 = tk.Label(root, bg="white")
label2.grid(row=0, column=1, padx=5, pady=5)

label3 = tk.Label(root, bg="white")
label3.grid(row=0, column=2, padx=5, pady=5)

label4 = tk.Label(root, bg="white")
label4.grid(row=0, column=3, padx=5, pady=5)

label5 = tk.Label(root, bg="white")
label5.grid(row=0, column=4, padx=5, pady=5)


# Labels for text below frames box
label1_text = tk.Label(root, text="Original Frame")
label1_text.grid(row=1, column=0, padx=5, pady=5)

label2_text = tk.Label(root, text="Gray Frame")
label2_text.grid(row=1, column=1, padx=5, pady=5)

label3_text = tk.Label(root, text="Face Frame")
label3_text.grid(row=1, column=2, padx=5, pady=5)

label4_text = tk.Label(root, text="Sharpened Frame")
label4_text.grid(row=1, column=3, padx=5, pady=5)

label5_text = tk.Label(root, text="Canny Edge Detection Frame")
label5_text.grid(row=1, column=4, padx=5, pady=5)


# Function to stop detection and release the video capture device
def stop_detection_and_release():
    root.after_cancel(update_id)
    cap.release()
    cv2.destroyAllWindows()

# Global variable to keep track of detection state
detection_running = False
update_id = None


# Function to start detection
def start_detection():
    global detection_running, update_id
    if not detection_running:
        detection_running = True
        update_id = root.after(10, update_frames)


# Function to stop detection
def stop_detection():
    global detection_running, update_id
    if detection_running:
        detection_running = False
        root.after_cancel(update_id)


# Start detection button
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.grid(row=3, column=0, columnspan=6, padx=10, pady=10)


# Stop detection button
stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.grid(row=4, column=0, columnspan=6, padx=10, pady=10)


# Function to handle application exit
def on_closing():
    stop_detection_and_release()
    root.destroy()


# Bind the application exit to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)


root.mainloop()


cv2.destroyAllWindows()


