import cv2
import numpy as np

# Fonction de rappel pour obtenir la position de la souris et afficher la valeur HSV
def show_hsv_values(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Convertir l'image en HSV
        img_hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        
        # Récupérer la valeur HSV du pixel sous la souris
        hsv_value = img_hsv[y, x]
        
        # Afficher la valeur HSV
        hsv_text = f"HSV: {hsv_value}"
        
        # Ajouter du texte sur l'image originale
        img_copy = param.copy()  # Créer une copie pour ne pas altérer l'image originale
        cv2.putText(img_copy, hsv_text, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Afficher l'image avec la valeur HSV
        cv2.imshow("Image with HSV values", img_copy)

# Charger l'image
img = cv2.imread('Photos/tymio_islands_resised.jpg')

# Créer une fenêtre
cv2.imshow("Image with HSV values", img)

# Définir la fonction de rappel pour la souris
cv2.setMouseCallback("Image with HSV values", show_hsv_values, img)

# Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre
cv2.waitKey(0)
cv2.destroyAllWindows()
