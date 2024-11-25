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

def squeeze(iterable,type=None):
        '''Reduces the dimension of list,tuple or array and (optional) changes its datatype.
        Always returns a numpy array of dimension 1 at mininum (no scalar)'''
        #so for example if you a the list of tuples with useless dimensions [ [ [(a,b)] , [(a,b)] ] ]
        #it returns the array [ [a,b] , [a,b] ]
        array = np.array(iterable)
        if type is not None:
            array = array.astype(type)

        #squeeze the arrays out of its useless dimensions
        array = array.squeeze() 
        
        #if array was squeezed too much (into dim0), bring it up to dim1
        if array.shape == ():  
            array = np.array([array])
        
        return array

##### deprecated ?
def show_img_Julien(img, title, wait_ms=1):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    # This allows the window to refresh without blocking the program
    key = cv2.waitKey(wait_ms) & 0xFF
    # Optionally handle 'q' to quit the display loop in the main program
    if key == ord('q'):
        return False
    return True        
#### deprecated ?
def show_img(img, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)

class VisionModule:
    """Module de gestion de la caméra et analyse d'image"""
    def __init__(self, image=None, map_size=(1000,1000)):
        self.cam = None
        self.frame = image  #this should always contain the original, unaltered image
        self.frame_viz = None  #this is where the original frame WITH annotations should be
        self.top_view = None   #this is where the top-view visualization should be
        #dim 0 is which corner (0,1,2,3 = tl,tr,br,bl), dim1 is x or y
        self.map_corners = np.ones((4,2),dtype='int32')*(-1)
        self.map_size = map_size  #arbitrary chosen metric of our map

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

    def detect_aruco(sel, img):
        '''Detect all aruco markers on the input image. Returns the array of the marker
        IDs and the array (nb_markers)x4x2 array of the 4 corners of each marker'''
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        markerCorners, markerIds, _ = detector.detectMarkers(gray)
        if markerIds is not None:
            frame_markers = cv2.aruco.drawDetectedMarkers(img.copy(), markerCorners, markerIds)
            print(f"Detected markers: {markerIds.flatten()}")
        else:
            print("no markers")

        #we want to get rid of the tuple around the array, and the extra dimention it implies
        markerCorners = squeeze(markerCorners)#np.array(markerCorners).squeeze() 
        
        return markerCorners, markerIds
    
    def get_6_markers(self,img):
        '''Find 6 markers and return 3D array of them sorted in the order :
        [TL, TR, BR, BL, Thymio, Goal]. Dim1 is markers, dim2 is corners of the marker
        dim3 is x and y of the corner'''
        markers, ids = self.detect_aruco(img)
        print(f"Detected {len(ids)} markers : {list(squeeze(ids))}")
        if markers.shape == (0,):
            return squeeze(markers), squeeze(ids)
        #verifiy that we have the 6 markers   
        # if not(len(ids)==6):
        #     print(f"Detected {len(ids)} markers instead of 6")
        #if we have multiple markers, we need to order them    
        if not(len(ids)== 1):
            pairs = sorted(list(zip(ids,markers))) #make a list of corresponding [id,marker] pairs, then sort it
            ids, markers = zip(*pairs) #unzip the pairs : ids are now in order 0-5 and the corresponding aruco corner markers are in the same order
   
        return squeeze(markers) , squeeze(ids)
    

    def find_marker_center_and_orientation(self, marker):
        '''Returns the center point of four coordinates (must be given
        in TL,TR,BR,BL order) and its orientation as an array of int'''
        #we're gonna calculate the interception of the diagonals to find the center
        tl,tr,br,bl = marker
        alpha = (tr[1] - bl[1]) / (tr[0]-bl[0]) #slope of bl->tr diagonal is : delta_y / delta_x = y2-y4 / x2-x4
        beta = (tl[1]-br[1]) / (tl[0] - br[0]) #slope of br->tl diagonal is y1-y3 / x1 - x3
        #if you find the slope and intercept of each diagonal, equate them and isolate x, you get this
        #do the math yourself
        x_center = (tl[1] - tr[1] - beta*tl[0] + alpha*tr[0]) / (alpha - beta)  
        y_center = alpha*x_center + (tr[1] - alpha*tr[0]) # use the equation of the bl->tr diagonal to find y_center

        #now we will use the dot product to find the relative angle between the top side of the marker (tl->tr) and the horizontal
        top_side = np.array([tr[0]-tl[0],tr[1]-tl[1]])
        top_side = top_side / np.linalg.norm(top_side) #normalize the vector
        unit_hvec = np.array([1,0])  #unitary horizontal vector
        theta = np.arccos(np.dot(top_side,unit_hvec)) # v1 dot v2 = ||v1||*||v2||*cos(theta) = cos(theta) if vects are unitary
        return x_center,y_center, theta
    
    def get_map_corners(self,markers, ids):
        '''Gets the 4 corners of the map based on the marker positions'''
    
        #only keep the first 4 ones, corresponding to the corners
        markers = markers[:4]

        #two ugly bug fixes
        #annoying : if only one marker has been detected, squeeze() will reduce it to (2,)
        #but we need (2,1)
        if markers.shape == (2,):
            markers = np.array([markers])
        #if only one corner has been detected, because of squeeze() the shape will be (4,2) instead of (1,4,2)
        if markers.shape == (4,2):
            markers = np.array([markers])

        # corners_of_markers = []
        for i in range(len(ids)):
            corner_id = ids[i]
            #if the current corner was not detected or if it's thymio/goal marker (4 and 5)
            if not(corner_id in ids) or corner_id >= 4:  
                continue
            center = self.find_marker_center_and_orientation(markers[i])
            center = center[:2]  #only need (x,y), not theta
            self.map_corners[corner_id,:] = center

        self.highlight_corners(self.frame.copy(),self.map_corners)
        # print(f"{len(ids)} markers detected, corners in memory are : {self.map_corners}")
        return self.map_corners
    
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually''
        (tl, tr, br, bl) = pts.astype('int32')
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
        M = cv2.getPerspectiveTransform(pts.astype('float32'), dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        self.highlight_corners(warped, dst)
        # return the warped image
        return warped, M

    
    def rescale_points(self,pts):
        '''Takes a list of points given in pixel coordinates of the original image and 
        rescale them according to our map coordinates'''

        x_min = np.min(self.map_corners[:,0]) #find the min on the x-axis
        x_max = np.max(self.map_corners[:,0])
        y_min = np.min(self.map_corners[:,1]) #find the min on the y-ayis
        y_max = np.max(self.map_corners[:,1])
        rescaled_pts = []
        for pt in pts:
            rescaled_x, rescaled_y = pt
            rescaled_x = self.map_size[0]*(rescaled_x - x_min) / (x_max - x_min)  #map scale * (x normalized w/ regard to map pixel dimensions)
            rescaled_y = self.map_size[1]*(rescaled_y - y_min) / (y_max - y_min)
            rescaled_pts.append((rescaled_x, rescaled_y))
        return rescaled_pts
        
    def highlight_corners(self, image, corners, title='highlighted corners', show=True):
        '''Draws a circle around the corners detected in this iteration,
        and a rectangle around the corners remebered/stored in self.map_corners '''
        # img = image.copy()
        corners = squeeze(corners, type='int32') #np.array(corners, dtype='int32').squeeze()
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]  #red, green, blue, yellow
        for i,corner in enumerate(corners):
            highlighted_corners = cv2.circle(self.frame.copy(),corner, 20, colors[i],4) #this is just for debugging
        in_memory = 0
        for stored_corner in self.map_corners:
            if stored_corner[0] == -1:  #don't show corners which haven't been initialized
                continue 
            #for cv2.rectangle, you need to give the tl and br corners of the rectangle to draw
            top_left = stored_corner + np.array([-25, -25])
            bottom_right = stored_corner + np.array([25,25])
            highlighted_corners = cv2.rectangle(highlighted_corners, top_left, bottom_right, (255,0,255)) #purple
            in_memory += 1

        if show: 
            # show_img(highlighted_corners,"Corners : o = detected    [] = remembered")
            self.frame_viz = highlighted_corners

        print(f"{len(corners)} detected corners, {in_memory} corners stored")

        

    #def detect_thymio_pose(self, markers, ids):
    #    '''Returns integer position array [x,y,z=0] and float theta corresponding to Thymio position and orientation '''
    #    thymio_id = 4
    #    if not(thymio_id in ids):
    #        print("Thymio marker not detcted")
    #        return None, None
    #    thymio_marker = markers[np.where(ids == thymio_id)[0][0]] #get the thymio_marker
    #    x,y,theta = self.find_marker_center_and_orientation(thymio_marker)
    #    #we return the position and angle
    #    return np.array([x,y,0], dtype='int32'), theta 
    

    #def detect_goal_position(self, markers):
    #    ''' Return int array [x,y,z=0] corresponding to goal position'''
    #    goal_marker = markers[5]
    #    x,y,_ = self.find_marker_center_and_orientation(goal_marker)
    #    return np.array([x,y,0], dtype='int32')

    def detect_thymio_pose(self,thymio_marker):
        '''Returns integer position array [x,y,z=0] and float theta corresponding to Thymio position and orientation '''
        x,y,theta = self.find_marker_center_and_orientation(thymio_marker)
        #we return the position with 
        return np.array([x,y], dtype='int32'), theta


    def detect_goal_position(self, goal_marker):
        ''' Return int array [x,y,z=0] corresponding to goal position'''
        x,y,_ = self.find_marker_center_and_orientation(goal_marker)
        return np.array([x,y], dtype='int32')
    
    
    def get_2_markers(self, top_view_img):
        """
        Find 2 markers (Thymio and Goal) in the top-view image
        """
        markers, ids = self.detect_aruco(top_view_img)

        # Verify that we have at least the two desired markers
        if not (len(ids) >= 2):
            print(f"Detected {len(ids)} markers instead of at least 2.")
            return markers.squeeze(), ids.squeeze()

        # Identify markers by their IDs: assume Thymio (e.g., ID=4) and Goal (e.g., ID=5)
        thymio_marker = None
        goal_marker = None
        for marker, marker_id in zip(markers, ids):
            if marker_id == 4:  # Replace '4' with the actual ID for Thymio
                thymio_marker = marker
            elif marker_id == 5:  # Replace '5' with the actual ID for Goal
                goal_marker = marker

        if thymio_marker is None:
            print("Unable to detect Thymio.")
            return False
        if goal_marker is None:
            print("Unable to detect the Goal marker.")
            return False
        return thymio_marker, goal_marker

    def julien_main(self, img):
        ''' ThIS SHOULD NOT STAY ! I only put it as an example of how my code is supposed to be used '''
        # visio = VisionModule()
        # visio.initialize_camera(cam_port=4)
        # ### take a photo with Tifaine's code, or open an already existing photo
        # filename = 'Photos/Photo7.jpg'
        # img = cv2.imread(filename, cv2.IMREAD_COLOR)

        #get location of 6 markers (and their ids) in camera frame
        #NB this function, and all the rest, *should* still work if some markers aren't detected
        markers, ids = self.get_6_markers(img)
        #find the coordinates of the 4 map corners (4x2 array).
        corners = self.get_map_corners(markers, ids)
        top_view_img, four_point_matrix = self.four_point_transform(img,corners)
        self.top_view = top_view_img
        return 
        thymio_pose, thymio_theta = self.detect_thymio_pose(markers, ids)
        goal_pos = self.detect_goal_position(markers)
    
        #let's put a green circle on top of the goal
        annotated_OG_img = cv2.circle(img.copy(), goal_pos[:2], 20, (0,255,0), 5)
        #let's draw an arrow from the center of the thymio in the direction where it's looking
        arrow_length = 100 #in pixels
        start = thymio_pose[:2]
        end = (int(start[0] + arrow_length*np.cos(thymio_theta)), int(start[1] + arrow_length*np.sin(thymio_theta)))
        end = start + (arrow_length*np.cos(thymio_pose[2]),arrow_length*np.sin(thymio_pose[2]))
        annotated_OG_img = cv2.arrowedLine(annotated_OG_img,start, end, (255,0,0), 20)
        #let's put a green circle on top of the goal
        annotated_OG_img = cv2.circle(annotated_OG_img, goal_pos[:2], 20, (0,255,0), 5)
        # show_img(annotated_OG_img,'OG image')
        # self.frame_viz = annotated_OG_img

         #use the 4 corner coordinates to do a perspective transform (correct, flatten, crop) on the map corners
        top_view_img, four_point_matrix = self.four_point_transform(img,corners)
        self.top_view = top_view_img
        #draw the circle and arrow on the transformed image too
        #!NB : THIS CODE IS PROBABLY NOT WORKING ! BUT ZHUORAN FIXED IT ?
        new_thymio = four_point_matrix @ thymio_pose
        new_thymio_angle = thymio_theta - (np.pi)/2
        new_goal = (four_point_matrix @ goal_pos).astype('int32')
        annotated_top_view = cv2.circle(top_view_img.copy(), new_goal[:2], 20, (0,255,0), 5)
        start = np.array(new_thymio[:2]).astype(int)
        end = (int(start[0] + arrow_length*np.cos(thymio_theta)), int(start[1] + arrow_length*np.sin(thymio_theta)))
        annotated_top_view = cv2.arrowedLine(annotated_top_view,start, end, (255,0,0), 20) #the angle should be off
        # show_img(annotated_top_view, 'top view visulazation')

        #convert all coordinates to our "imaginary" map coordinates
        #! NB : this only needs the original camera frame coordinates, the 4 points transform and so on
        #! is only there for visualisation
        pts = self.rescale_points([*corners, thymio_pose[:2], goal_pos])
        *r_corners, r_thymio, r_goal = pts
        r_goal = np.array(r_goal).astype(int)
        imaginary_coordinates = cv2.circle(top_view_img.copy(), r_goal, 20, (0,255,0), 5)
        imaginary_coordinates = cv2.circle(imaginary_coordinates, r_thymio, 20, (255,0,0))
        print(0)

        return annotated_OG_img



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

                self.vision_module.julien_main(frame)
                annotated_img = self.vision_module.frame_viz
                top_view = self.vision_module.top_view

                Thymio_marker, goal_marker = self.vision_module.get_2_markers(top_view)
                robot_position, theta = self.vision_module.detect_thymio_pose(Thymio_marker)
                goal_position = self.vision_module.detect_goal_position(goal_marker)

                obstacle_corners, img_with_polygons = self.vision_module.detect_obstacle_corners(top_view)


                # Afficher les deux images en parallèle
                show_many_img([frame, img_with_polygons, annotated_img], ["Original", "Processed_with_polygones", "Highlighting corners"])

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
    if not vision.initialize_camera(cam_port=4):
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