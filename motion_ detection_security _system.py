import threading
import sounddevice as sd
import cv2
import imutils
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

alarm = False
alarm_mode = False
alarm_counter = 0

def beep_alarm():
    global alarm
    while alarm_mode:
        duration = 5.0  # seconds
        frequency = 440  # Hz
        fs = 44100  # sampling rate

        t = np.arange(int(fs * duration)) / fs
        signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        print("MOTION DETECTED")
        sd.play(signal, fs)
        sd.wait()

        server=smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login('Recipient Email address','Password')
        server.sendmail('Recipient Email Address','Sender Email Address','Motion is detected')
        print('Message sent successfully')
        
        # Call detect_objects function
        _, frame = cap.read()
        detect_objects(frame)

        break

def detect_objects(image):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Your object detection code here

    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            detected_objects.append((label, boxes[i]))
    print('Details of the object is bieng sent')
    send_email(detected_objects, frame)

def send_email(detected_objects, frame):
    sender_email = "**********"
    receiver_email = "**********"
    password = "****************"

    subject = "Detected Object"
    body = "Detected object(s):"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    for label, (x, y, w, h) in detected_objects:
        object_image = frame[y:y + h, x:x + w]
        _, buffer = cv2.imencode(".jpg", object_image)
        image_attachment = MIMEImage(buffer.tobytes())
        image_attachment.add_header("Content-Disposition", f"attachment; filename={label}.jpg")
        message.attach(image_attachment)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

        difference = cv2.absdiff(start_frame, frame_bw)
        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        start_frame = frame_bw

        if threshold.sum() > 300:  # This is the sensitivity of the motion detection, you can change this value
            alarm_counter += 1
            if alarm_counter > 20:
                if not alarm:
                    alarm = True
                    threading.Thread(target=beep_alarm).start()
        else:
            if alarm_counter > 0:
                alarm_counter -= 1
        cv2.imshow("Cam", threshold)
    elif not alarm_mode and alarm:
        black_frame = np.zeros_like(frame)
        cv2.imshow("Cam", black_frame)
    else:
        cv2.imshow("Cam", frame)

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord('t'):
        print("You have activated/deactivated the alarm!")
        alarm_mode = not alarm_mode
        alarm_counter = 0
    elif key_pressed == ord('q'):
        print("Quitting the program!")
        alarm_mode = False
        break

cv2.destroyAllWindows()
