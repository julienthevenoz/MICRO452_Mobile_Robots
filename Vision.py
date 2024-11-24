import cv2
import numpy as np

MIN_AREA = 10
DIST_THRESHOLD = 10


class VisionModule:
    def __init__(self):
        self.cam = None

    def initialize_camera(self, cam_port=0):  # Par défaut, port 0
        self.cam = cv2.VideoCapture(cam_port)
        if not self.cam.isOpened():
            raise Exception("Impossible d'ouvrir la caméra.")

    def distance(self, p1, p2):
        """Calcul de la distance euclidienne entre deux points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def merge_polygons(self, obstacle_corners, threshold=10):
        """
        Fusionne des polygones dont les coins sont proches.
        """
        merged = []
        for polygon in obstacle_corners:
            added = False
            for merged_polygon in merged:
                for point in polygon:
                    if any(self.distance(point, merged_point) < threshold for merged_point in merged_polygon):
                        merged_polygon.extend(polygon)
                        added = True
                        break
                if added:
                    break
            if not added:
                merged.append(list(polygon))

        # Nettoyage des doublons
        merged_cleaned = []
        for merged_polygon in merged:
            cleaned = []
            for point in merged_polygon:
                if not any(self.distance(point, existing_point) < threshold for existing_point in cleaned):
                    cleaned.append(point)
            if len(cleaned) > 2:
                cleaned = cv2.convexHull(np.array(cleaned, dtype=np.int32)).reshape(-1, 2).tolist()
            merged_cleaned.append(cleaned)

        return merged_cleaned

    def detect_obstacle_corners(self, img):
        """
        Détecte les obstacles dans l'image.
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=150, threshold2=200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacle_corners = []
        lower_black_hsv = np.array([0, 80, 20])
        upper_black_hsv = np.array([179, 150, 120])

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            obstacle_corners.append(approx.reshape(-1, 2).tolist())

        # Fusionner les polygones proches
        obstacle_corners = self.merge_polygons(obstacle_corners, DIST_THRESHOLD)

        final_polygons = []
        for polygon in obstacle_corners:
            mask = np.zeros_like(hsv_img[:, :, 0])
            cv2.drawContours(mask, [np.array(polygon, dtype=np.int32)], -1, (255), thickness=cv2.FILLED)

            polygon_array = np.array(polygon, dtype=np.int32)
            area = cv2.contourArea(polygon_array)
            if area < MIN_AREA:
                continue

            mean_color_hsv = cv2.mean(hsv_img, mask=mask)
            if ((lower_black_hsv[0] <= mean_color_hsv[0] <= upper_black_hsv[0]) and
                (lower_black_hsv[1] <= mean_color_hsv[1] <= upper_black_hsv[1]) and
                (lower_black_hsv[2] <= mean_color_hsv[2] <= upper_black_hsv[2])):
                final_polygons.append(polygon)

        return final_polygons

    def draw_obstacles(self, img, obstacle_corners):
        """
        Dessine les obstacles détectés sur l'image.
        """
        for polygon in obstacle_corners:
            polygon_array = np.array(polygon, dtype=np.int32)
            color = (0, 255, 0)  # Vert pour les polygones
            cv2.drawContours(img, [polygon_array], -1, color, 2)
            for (x, y) in polygon:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Coins en rouge
        return img


def main():
    vision = VisionModule()
    vision.initialize_camera(0)  # Initialiser la caméra (port 0)

    try:
        while True:
            ret, frame = vision.cam.read()
            if not ret:
                print("Erreur lors de la capture vidéo.")
                break

            obstacle_corners = vision.detect_obstacle_corners(frame)
            frame_with_obstacles = vision.draw_obstacles(frame, obstacle_corners)

            cv2.imshow('Détection d\'obstacles', frame_with_obstacles)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Appuyer sur 'q' pour quitter
                break

    finally:
        vision.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()