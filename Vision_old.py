import cv2

cap = cv2.VideoCapture(0)  # Essayez avec 1, 2, ... si 0 ne fonctionne pas
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Échec de la capture vidéo.")
        break

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Appuyez sur 'q' pour quitter
        break

cap.release()
cv2.destroyAllWindows()
