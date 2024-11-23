import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans ##### librairie pour le K-mean, le PCA,... #####


##### interet de faire une class Vision_Thymio #####
##### le optimal path du coup il va ici où dans le fichier filtering ou ailleur? #####
##### on fait un nouveau fichier pour le filtering ou il va où? #####

# Camera & Image Processing
#Magic numbers
#NB : opencv uses convention of [0,179], [0,255] and [0,255] for HSV values instead of the common [0,360],[0,100], [0,100]
UPPER_GREEN = np.array([120,255,255], dtype='uint8')    
LOWER_GREEN = np.array([70,0,0], dtype='uint8')

UPPER_BROWN = np.array([60,255,255])
LOWER_BROWN = np.array([0,0,0])


def show_img(img,title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    while True: 
        if cv2.waitKey(1) & 255==ord('q'):  # Wait for a q press or window closed to close the window
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def show_many_img(img_list, title_list):
    if len(img_list) != len(title_list):
        print("Unequal list size")
        return
    for i in range(len(img_list)):
        cv2.namedWindow(title_list[i], cv2.WINDOW_NORMAL)
        cv2.imshow(title_list[i], img_list[i])

    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


class Vision_module():
    def __init__(self, image=None):
        self.img = image
        self.cam = None

    # Camera & Image Processing

    def initialize_camera(self, cam_port=4):  #on my PC, the webcam is in port 4
        self.cam = cv2.VideoCapture(cam_port)
        return

    def capture_image(self):
        result, image = self.cam.read()
        print(result)
        show_img(image, 'camera image')
        return
    
    # Map definition

    def Image_correction(image): # distortions or perspective distortions, definition des bords et coin. Crop les bords de l'image
        return
    
    ##### https://evergreenllc2020.medium.com/building-document-scanner-with-opencv-and-python-2306ee65c3db #####
    
    def extract_edge(self, img): ##### à merge/repmplacer par Image_correction #####
        # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        show_img(img,'supposedly masked image')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        show_img(gray,"gray img")

        #blurred_img = cv2.GaussianBlur(gray, ksize=[5,5],sigmaY=10,sigmaX=10)  
        #show_img(blurred_img,"blurred img")

        cannied_img = cv2.Canny(gray,75,200)
        show_img(cannied_img, "canny edge image")


        # Find contours
        contours, hierarchy = cv2.findContours(cannied_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours   #this is just for testing
        drawing = np.zeros((cannied_img.shape[0], cannied_img.shape[1], 3), dtype=np.uint8)
        import random
        for i in range(len(contours)):            
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        # Show in a window
        show_img(drawing,'Contours')
        return contours

    def get_colour_mask(self, img, lower, upper):
        original = img.copy()
        img = cv2.GaussianBlur(img, (5, 5), 0)  #apply gaussian smoothing
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #transform to Hue(=color)-Saturation-Value(=brightness) format to make color detection easier
        mask = cv2.inRange(img, lower, upper)
        # show_img(mask,"mask")
        return mask
    
    def color_segmentation(self, img):
        """
        Segmente l'image en fonction des couleurs spécifiques : noir, blanc, et bleu (mer).
        """
        # Convertir l'image en HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Définir les plages de couleur pour chaque catégorie
        # Plage de couleur pour le noir
        lower_black = np.array([0, 0, 30])
        upper_black = np.array([179, 120, 80])

        # Plage de couleur pour le blanc
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([179, 50, 250])

        # Plage de couleur pour le bleu (mer)
        lower_blue = np.array([85, 40, 100])
        upper_blue = np.array([110, 210, 200])

        # Créer des masques pour chaque couleur
        black_mask = cv2.inRange(hsv_img, lower_black, upper_black)
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        # Combiner les masques
        combined_mask = cv2.bitwise_or(black_mask, white_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Appliquer le masque à l'image pour isoler les objets
        result_img = cv2.bitwise_and(img, img, mask=combined_mask)

        return result_img

    def kmeans_color_segmentation(self, img, n_clusters=3):
        """
        Segmente l'image en utilisant K-means clustering pour détecter les couleurs principales.
        - Blanc (Thymio)
        - Noir (Obstacles)
        - Bleu (Fond/Mer)
        """
        # Convertir l'image en RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir l'image en RGB

        # Redimensionner l'image en une liste de pixels
        pixels = img_rgb.reshape((-1, 3)).astype('float32')  # Aplatir l'image en un tableau 2D

        # Appliquer K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(pixels)  # Récupérer les labels de chaque pixel
        centers = np.uint8(kmeans.cluster_centers_)  # Récupérer les centres des clusters

        # Définir une fonction pour calculer la luminosité (pour la séparation des couleurs)
        def brightness(color):
            return np.mean(color)

        # Trier les clusters en fonction de leur luminosité
        sorted_indices = np.argsort([brightness(center) for center in centers])
        black_cluster = sorted_indices[0]  # Cluster le plus sombre = Noir
        white_cluster = sorted_indices[-1]  # Cluster le plus lumineux = Blanc
        blue_cluster = sorted_indices[1]  # Cluster intermédiaire = Bleu (mer)

        # Mappage des clusters vers les couleurs spécifiées
        color_map = {
            black_cluster: [0, 0, 0],  # Noir pour les obstacles
            blue_cluster: [255, 0, 0],  # Bleu pour la mer
            white_cluster: [255, 255, 255],  # Blanc pour Thymio
        }

        # Appliquer les couleurs mappées sur l'image
        labels_reshaped = labels.reshape(img.shape[:2])  # Redimensionner les labels à la taille de l'image originale
        result_img = np.zeros_like(img)  # Initialiser l'image résultat

        for cluster, color in color_map.items():
            result_img[labels_reshaped == cluster] = color  # Assigner la couleur aux pixels correspondants

        return result_img, labels



    def find_map_corners(self, contours):
        #we sort our contours by area, in descending order, and we only need to keep first 5
        # of them since we're looking for a very big object (the map)

        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5] 
        found = False
        #loop over the contours
        for i,c in enumerate(cnts):
            perimeter = cv2.arcLength(c,True)  #calculate length of the contour
            approx = cv2.approxPolyDP(c, perimeter*0.02, True) #create a polygon that approximates the curve
            drawing = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
            drawn = cv2.drawContours(drawing,approx,-1, (0,255,0), 3)
            # drawn = cv2.drawContours(drawing, c, -1, (0,0,255), 2)
            print(len(approx))
            show_img(drawn,f"contour nb{i}")
            if len(approx) == 4:
                map_contour = approx   #if the polygon has 4 corners, it's probably the rectangle of our map
                found = True
                break

        if found:
            print("Found !")
            drawn = cv2.drawContours(self.img, map_contour,-1, (0,255,0), 2)
            show_img(drawn,"map contours")
            #map_contour is structed in a stupid way (4x1x2), so we do list comprehension
            #magic to make it into a more logical 4x2 array
            map_contour = np.array([xy for subarray in map_contour for xy in subarray])
            return map_contour
        else:
            print("no 4 corner contour")
            return None

    def order_points(self,pts):
        '''Takes an input vector of points (=size 4x2) representing 
        the 4 corners of a rectangle and returns the same points re-ordered
        as : Top-right, Top-left, Bottom-left, Bottom-right
        '''
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect
    
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually''
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

    def detect_aruco(sel, img):
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)

        pass

    def detect_thymio_position(image):
        return

    def detect_goal_position(image):
        return

    def detect_obstacles(image):    #def les bords des obstacles
        return

    def map_rescaling (self):   #### a definir ou on le met exactement 
        return 

if __name__ == "__main__":
    filename = 'Photos/tymio_islands_resised_no_boats.jpg'

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    visio = Vision_module(img)

    show_img(img, "Original Image")
    
    #mask = visio.get_colour_mask(img, LOWER_GREEN, UPPER_GREEN)
    #masked_img = cv2.bitwise_and(img, img, mask=mask)  # apply color mask (keep only green pixels)

    #Thresholding
    segmented_img = visio.color_segmentation(img)
    # K-means segmentation
    #segmented_img, labels = visio.kmeans_color_segmentation(thresholded_img, n_clusters=3)

    show_img(segmented_img, "Filtered Image")

    # Commented out parts for top-view transformation
    # contours = visio.extract_edge(img)
    # corners = visio.find_map_corners(contours)
    # corners = visio.order_points(corners)
    # top_view_img = visio.four_point_transform(img, corners)
    # show_img(top_view_img, 'topi gang')


    
