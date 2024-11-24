import cv2
import threading
import time
import numpy as np

MIN_AREA = 10
DIST_THRESHOLD = 10

# Fonction pour afficher plusieurs images en parallèle
def show_many_img(img_list, title_list):
    if len(img_list) != len(title_list):
        print("Unequal list size")
        return
    for i in range(len(img_list)):
        cv2.namedWindow(title_list[i], cv2.WINDOW_NORMAL)
        cv2.imshow(title_list[i], img_list[i])

def show_img(img, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)

class VisionModule:
    """Module de gestion de la caméra et analyse d'image"""
    def __init__(self):
        self.cam = None
        self.frame = None

    def initialize_camera(self, cam_port=0):
        """Initialise la caméra"""
        self.cam = cv2.VideoCapture(cam_port)
        if not self.cam.isOpened():
            print(f"Impossible d'ouvrir la caméra sur le port {cam_port}")
            return False
        print(f"Caméra initialisée sur le port {cam_port}")
        return True

    def capture_frame(self):
        """Capture une image de la caméra"""
        if not self.cam or not self.cam.isOpened():
            print("La caméra n'est pas initialisée.")
            return None

        ret, frame = self.cam.read()
        if not ret:
            print("Échec de la capture d'image.")
            return None

        self.frame = frame
        return frame
    
    def distance(self, p1, p2):
        """Calcul de la distance euclidienne entre deux points"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def merge_polygons(self, obstacle_corners, threshold=10):
        """
        Fusionne des polygones dont les coins sont proches.
        obstacle_corners : liste des coins des polygones.
        threshold : distance maximale pour considérer deux points comme proches.
        """
        merged = []
    
        for polygon in obstacle_corners:
            added = False
            for merged_polygon in merged:
                # Vérifiez si le polygone actuel est proche de l'un des polygones fusionnés
                for point in polygon:
                    if any(self.distance(point, merged_point) < threshold 
                           for merged_point in merged_polygon):
                        # Si oui, fusionnez les coins
                        merged_polygon.extend(polygon)
                        added = True
                        break
                if added:
                    break
                
            if not added:
                # Si le polygone n'est pas proche d'un polygone fusionné existant, ajoutez-le tel quel
                merged.append(list(polygon))
    
        # Nettoyez les doublons dans les points fusionnés
        merged_cleaned = []
        for merged_polygon in merged:
            cleaned = []
            for point in merged_polygon:
                if not any(self.distance(point, existing_point) < threshold 
                           for existing_point in cleaned):
                    cleaned.append(point)
            # Appliquer l'enveloppe convexe pour garantir une forme convexe
            if len(cleaned) > 2:  # S'il y a suffisamment de points pour former un polygone
                cleaned = cv2.convexHull(np.array(cleaned, dtype=np.int32)).reshape(-1, 2).tolist()
        
            merged_cleaned.append(cleaned)
    
        return merged_cleaned

    def detect_obstacle_corners(self, img):
        """
        Détecte les bords des obstacles dans l'image, les approxime par des polygones,
        et garde uniquement ceux dont la couleur moyenne est noire (obstacles).
        """
        # 1. Conversion de l'image en HSV pour une analyse des couleurs
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 2. Conversion de l'image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Application d'un flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. Détection des bords avec Canny
        edges = cv2.Canny(blurred, threshold1=150, threshold2=200)

        # 5. Trouver les contours dans l'image avec la méthode findContours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. Liste pour stocker les coins des obstacles
        obstacle_corners = []

        # 7. Image pour visualiser les polygones détectés
        img_with_polygons = img.copy()

        # 8. Plage de teinte pour le noir (en HSV)
        lower_black_hsv = np.array([0, 80, 20])  # Ajustez selon vos besoins
        upper_black_hsv = np.array([179, 150, 120])

        # 9. Parcourir chaque contour trouvé
        approx_img = img.copy()  # Image pour visualiser l'approximation des polygones
        for contour in contours:
            # Approximation du contour par un polygone
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Précision de l'approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Générer une couleur aléatoire pour chaque polygone
            color = np.random.randint(0, 256, 3).tolist()  # Couleur aléatoire (BGR)

            # Dessiner les coins du polygone
            for (x, y) in approx.reshape(-1, 2):  # Dessiner les coins
                cv2.circle(approx_img, (x, y), 5, (0, 0, 255), -1)  # Coins en rouge
            cv2.drawContours(approx_img, [approx], -1, color, 2)  # Dessiner les polygones
            obstacle_corners.append(approx.reshape(-1, 2).tolist())

        # Affichage intermédiaire de l'approximation des polygones
        #show_many_img([approx_img], ["Polygones avant fusion"])

        # 10. Fusionner les polygones proches
        obstacle_corners = self.merge_polygons(obstacle_corners, DIST_THRESHOLD)

        # Affichage après fusion des polygones
        merged_img = img.copy()
        for polygon in obstacle_corners:
            polygon_array = np.array(polygon, dtype=np.int32)
            color = np.random.randint(0, 256, 3).tolist()  # Couleur aléatoire pour chaque polygone
            cv2.drawContours(merged_img, [polygon_array], -1, color, 2)  # Dessiner les polygones fusionnés
            for (x, y) in polygon:
                cv2.circle(merged_img, (x, y), 5, (0, 0, 255), -1)  # Coins en rouge

        #show_many_img([merged_img], ["Polygones après fusion"])

        # 11. Calculer la moyenne des couleurs à l'intérieur des polygones et filtrer par couleur
        final_polygons = []
        MIN_AREA = 1000  # Seuil d'aire minimal pour un polygone

        for polygon in obstacle_corners:
            # Créer un masque pour extraire l'intérieur du polygone
            mask = np.zeros_like(hsv_img[:, :, 0])
            cv2.drawContours(mask, [np.array(polygon, dtype=np.int32)], -1, (255), thickness=cv2.FILLED)

            # Calcul de l'aire du polygone
            polygon_array = np.array(polygon, dtype=np.int32)
            area = cv2.contourArea(polygon_array)

            # Si l'aire est inférieure au seuil, on ignore ce polygone
            if area < MIN_AREA:
                continue  # Passer au prochain polygone si l'aire est trop petite

            # Calcul de la moyenne des couleurs à l'intérieur du polygone en HSV
            mean_color_hsv = cv2.mean(hsv_img, mask=mask)

            # Filtrer en fonction de la couleur noire
            if ((lower_black_hsv[0] <= mean_color_hsv[0] <= upper_black_hsv[0]) and 
                (lower_black_hsv[1] <= mean_color_hsv[1] <= upper_black_hsv[1]) and 
                (lower_black_hsv[2] <= mean_color_hsv[2] <= upper_black_hsv[2])):
                final_polygons.append(polygon)

        # 12. Affichage final avec les polygones filtrés
        final_img = img.copy()
        for polygon in final_polygons:
            polygon_array = np.array(polygon, dtype=np.int32)
            color = np.random.randint(0, 256, 3).tolist()  # Couleur aléatoire pour chaque polygone
            cv2.drawContours(final_img, [polygon_array], -1, color, 2)
            for (x, y) in polygon:
                cv2.circle(final_img, (x, y), 5, (0, 0, 255), -1)

        # Affichage final
        #show_many_img([final_img], ["Polygones filtrés"])

        return final_polygons, final_img

    def analyze_frame(self, frame):
        """
        Analyse l'image : dans cet exemple, on détecte les contours.
        Vous pouvez remplacer cette fonction par votre propre code d'analyse.
        """
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Application d'un flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Détection des contours avec Canny
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

        # Convertir les bords en image couleur pour affichage
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Retourner l'image avec analyse (ici, les contours)
        return edges_colored

    def release_camera(self):
        """Libère la caméra et ferme les fenêtres OpenCV"""
        if self.cam:
            self.cam.release()
        cv2.destroyAllWindows()


class CameraFeedThread(threading.Thread):
    """Thread pour capturer et afficher un flux vidéo constant"""
    def __init__(self, vision_module):
        super().__init__()
        self.vision_module = vision_module
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            frame = self.vision_module.capture_frame()
            if frame is not None:
                # Utilisation de la méthode d'analyse de VisionModule
                #processed_frame = self.vision_module.analyze_frame(frame)
                # Appeler la méthode pour détecter les coins des obstacles
                obstacle_corners, img_with_polygons = self.vision_module.detect_obstacle_corners(frame)


                # Afficher les deux images en parallèle
                show_many_img([frame, img_with_polygons], ["Original", "Processed_with_polygones"])

                # Quitter si la touche 'q' est pressée
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

    def stop(self):
        """Arrêter le thread et libérer la caméra"""
        self.stop_event.set()
        self.vision_module.release_camera()


def main():
    """Point d'entrée principal"""
    vision = VisionModule()
    if not vision.initialize_camera(cam_port=0):
        print("Erreur : Impossible d'initialiser la caméra.")
        return

    camera_thread = CameraFeedThread(vision)
    camera_thread.start()

    try:
        while True:
            # Vous pouvez exécuter d'autres tâches en parallèle ici
            # Le programme principal continue de tourner sans bloquer l'affichage
            print("Le programme principal fonctionne en arrière-plan...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt du programme demandé.")
    finally:
        camera_thread.stop()
        camera_thread.join()
        print("Programme terminé.")


if __name__ == "__main__":
    main()



##################################################################
##################################################################
""""

class Vision_module():
    
    def modify_image_for_visualization(self, img, obstacle_corners, tymio_position, tymio_radius=40):
        ""
        Modifie l'image en mettant le fond en bleu, les obstacles en noir, et le Tymio en blanc.
        
        :param img: L'image originale.
        :param obstacle_corners: Liste des coins des polygones représentant les obstacles.
        :param tymio_position: Position du Tymio dans l'image (x, y).
        :param tymio_radius: Rayon approximatif du Tymio pour dessiner un cercle autour de lui.
        :return: L'image modifiée.
        ""
        # Vérifier si l'image a été correctement chargée
        if img is None:
            raise ValueError("L'image n'a pas pu être chargée. Veuillez vérifier le chemin du fichier.")
    
        # Créer une image bleue (fond bleu)
        modified_img = np.zeros_like(img)  # Créer une image de la même taille que 'img'
        modified_img[:, :] = (220, 220, 0)  # Bleu en BGR
        
        # Dessiner les obstacles en noir (polygones définis par obstacle_corners)
        for corners in obstacle_corners:
            # Convertir les coins en array numpy et dessiner chaque polygone
            poly_points = np.array(corners, np.int32)
            poly_points = poly_points.reshape((-1, 1, 2))
            cv2.fillPoly(modified_img, [poly_points], (0, 0, 0))  # Noir pour les obstacles
        
        # Dessiner le Tymio en blanc (cercle autour de la position du Tymio)
        #cv2.circle(modified_img, tymio_position, tymio_radius, (255, 255, 255), -1)  # Blanc pour le Tymio
        
        return modified_img

if __name__ == "__main__":
    filename = 'Photos/tymio_islands.jpg'

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # Vérifier si l'image est correctement chargée
    if img is None:
        print("Erreur lors du chargement de l'image")
        exit()

    # Créer une instance de Vision_module
    visio = Vision_module()

    # Appeler la méthode pour détecter les coins des obstacles
    obstacle_corners, img_with_polygons = visio.detect_obstacle_corners(img)
    
    # Optionnellement, afficher l'image finale avec les polygones détectés
    cv2.imshow("Polygones et coins", img_with_polygons)
    cv2.waitKey(0)

     ##### FUN #####
    #modified_img = visio.modify_image_for_visualization(img_with_polygons, obstacle_corners, (210,520)) ##### METTRE LES BON COORDONNEE DU TYMIO #####
    #cv2.imshow("Modified Image", modified_img)
    #cv2.waitKey(0)

    cv2.destroyAllWindows()

"""