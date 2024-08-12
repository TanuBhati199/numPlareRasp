import numpy as np
import cv2
import easyocr
import csv
from datetime import datetime

def save_plate2csv(plate_text):
    date_time = datetime.now()
    date_str = date_time.strftime("%Y-%m-%d")
    day_str = date_time.strftime("%A")
    time_str = date_time.strftime("%H:%M:%S")
    with open('detected_plate.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([plate_text, date_str, time_str, day_str])

def detect_num_plate(cap):
    reader = easyocr.Reader(['en'])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        plate_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                plate_contour = approx
                break
        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            cropped_plate = frame[y:y+h, x:x+w]
            result = reader.readtext(cropped_plate)
            detected_text = []
            for bbox, text, prob in result:
                detected_text.append(text)
            if detected_text:
                final_text = ''.join(detected_text)
                print(f"Detected plate number: {final_text}")
                save_plate2csv(final_text)
                cv2.imwrite(f"cropped_plate_{final_text}.jpg", cropped_plate)
                cv2.imshow("Cropped Plate", cropped_plate)
                choice = input("Press 1 to capture more images or 0 to exit: ")
                if choice == '0':
                    break
            cv2.drawContours(frame, [plate_contour], -1, (0, 255, 0), 3)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)    
detect_num_plate(cap)

