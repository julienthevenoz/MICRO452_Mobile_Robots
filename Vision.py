import cv2
import threading
import time
import numpy as np


class VisionModule:
    """Module de gestion de la caméra"""
    def __init__(self):
        self.cam = None
        self.frame = None

    def initialize_camera(self, cam_port=0):
        self.cam = cv2.VideoCapture(cam_port)
        if not self.cam.isOpened():
            print(f"Impossible d'ouvrir la caméra sur le port {cam_port}")
            return False
        print(f"Caméra initialisée sur le port {cam_port}")
        return True

    def capture_frame(self):
        if not self.cam or not self.cam.isOpened():
            print("La caméra n'est pas initialisée.")
            return None

        ret, frame = self.cam.read()
        if not ret:
            print("Échec de la capture d'image.")
            return None

        self.frame = frame
        return frame
    
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
                processed_frame = self.vision_module.analyze_frame(frame)

                # Afficher l'image analysée
                cv2.imshow("Camera Feed", processed_frame)

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