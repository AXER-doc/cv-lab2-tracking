import cv2
import sys
import numpy as np

# Инициализация видео источника (файл или камера)
if len(sys.argv) > 1:
    cap = cv2.VideoCapture(sys.argv[1])
else:
    cap = cv2.VideoCapture(0)  # Камера по умолчанию

ret, frame = cap.read()
if not ret:
    print("Ошибка чтения видео")
    sys.exit()

# Ручная инициализация bounding box на первом кадре
print("Нажмите кнопку 'a' для автоматического поиска прямоугольника или нарисуйте bounding box мышью")
bbox = cv2.selectROI("region", frame, False) # Первый кадр - выберите объект
cv2.destroyWindow("region")

# Инициализация трекера (CSRT для точности, KCF для скорости)
tracker = cv2.TrackerCSRT_create()  # Альтернатива: cv2.TrackerKCF_create()
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обновление трекера
    success, bbox = tracker.update(frame)

    if success:
        # Рисование рамки и подписи
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("tracking", frame) # Трекинг объекта

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
