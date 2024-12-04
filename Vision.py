import cv2
import threading
import time
import numpy as np
from scipy.signal import medfilt

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
        '''Reduces the dimension of list, tuple or array and (optional) changes its datatype.
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

class Analysis:
    """Module de gestion de la caméra et analyse d'image"""
    def __init__(self, image=None):
        self.cam = None
        self.frame = image  #this should always contain the original, unaltered image
        self.frame_viz = image  #this is where the original frame WITH annotations should be
        self.top_view = None   #this is where the top-view visualization should be
        #dim 0 is which corner (0,1,2,3 = tl,tr,br,bl), dim1 is x or y
        self.map_corners = np.ones((4,2),dtype='int32')*(-1)
        self.map_size = (600,300)  #arbitrary chosen metric of our map
        self.last_thymio_pose = None
        self.last_goal_pos = None
        self.path = None

    def initialize_camera(self, cam_port=4):
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
    
    def detect_obstacle_corners(self, img):
        """
        Detects obstacle edges in the image, approximates them as polygons, 
        and keeps only those with an average color that resembles black (obstacles).
        Returns three outputs: the polygons, the binary obstacle mask, and the image with visualized polygons.
        """
        # 1. Convert the image to LAB color space for color analysis
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 2. Calculate the histogram of the L component
        hist = cv2.calcHist([lab_img[:, :, 0]], [0], None, [180], [0, 180])  # 180 bins for L in the range [0, 180]
        signal = np.gradient(medfilt(hist.flatten(), 15))  # Median filtering to smooth the signal

        grad_threshold = -15
        highest = 110

        # 3. Search for the threshold in the signal
        crossing_index = -1
        for i in range(highest - 1, 1, -1):  # Start just below 'highest' and move backwards
            if signal[i] < grad_threshold:
                crossing_index = i
                break

        # Find the threshold
        # if crossing_index != -1:
        #     print(f"The highest index where the signal crosses {grad_threshold} is {crossing_index}.")
        # else:
        #     print(f"No crossing found before index {highest}.")

        treshold = crossing_index

        # 4. Create a binary mask based on the threshold
        mask = lab_img[:, :, 0] < treshold
        obstacles = np.uint8(mask * 255)

        # 5. Process connected components
        COUNT = 2000  # Minimum size threshold for components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obstacles, connectivity=8)
        filtered_mask = np.zeros_like(obstacles)

        for i in range(1, num_labels):  # Ignore background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= COUNT:
                filtered_mask[labels == i] = 255

        # 6. Morphological operations to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Structuring element for dilation
        filtered_mask = cv2.dilate(filtered_mask, kernel)
        filtered_mask = cv2.erode(filtered_mask, kernel)

        # 7. Find contours of the obstacles
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 8. List to store the corners of the obstacles
        obstacle_corners = []

        # 9. Approximate the contours as polygons
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximation precision
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Add the corners of the polygons
            obstacle_corners.append(approx.reshape(-1, 2).tolist())

        # 10. Visualize the polygons with colored edges and circles at corners
        img_with_polygons = img.copy()

        # Assign a unique color for each polygon
        for i, polygon in enumerate(obstacle_corners):
            # Random color for each polygon (BGR format)
            color = np.random.randint(0, 256, 3).tolist()

            # Draw the corners as red circles
            for (x, y) in polygon:
                cv2.circle(img_with_polygons, (x, y), 5, (0, 0, 255), -1)  # Red circles at corners

            # Draw the polygon edges
            polygon_array = np.array(polygon, dtype=np.int32)
            cv2.drawContours(img_with_polygons, [polygon_array], -1, color, 2)  # Polygon edges with random color

        # 11. Return the results: obstacle corners, binary mask, and image with visualized polygons
        return obstacle_corners, filtered_mask, img_with_polygons

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
            #print(f"Detected markers: {markerIds.flatten()}")
        else:
            print("no markers")

        #we want to get rid of the tuple around the array, and the extra dimention it implies
        markerCorners = squeeze(markerCorners)#np.array(markerCorners).squeeze() 
        
        return markerCorners, markerIds
    
    def get_6_markers(self,img):
        '''Find 6 markers and return 3D array of them sorted in the order :
        markers = [TL, TR, BR, BL, Thymio, Goal]. Dim1 is markers, dim2 is corners of the marker
        dim3 is x and y of the corner. Also returns ids, which is an array telling which markers
        have been detected. For ex, if ids=[0,2,3,5], it means that markers=[TL, BR,BL, GOAL] 
        If no markers detected, both ids and markers = None
        '''
        markers, ids = self.detect_aruco(img)
        if ids is None:
            print("NO MARKERS DETECTED")
            return None, None
        #print(f"Detected {len(ids)} markers : {list(squeeze(ids))}")
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
        
        try:
            tl,tr,br,bl = marker
        except:
            print("THE FUCKING BUG ")
            tl,tr,br,bl = marker

        
        #find orientation of the marker
        theta = np.arctan2((tr[1]-br[1]), (tr[0]-br[0]))

        if (tr[0]-bl[0]) == 0 or (tl[0] - br[0]) == 0: 
        #! To prevent a divison by 0, if for some unholy reason the diag distance is 0, we don't try to calculate the center 
        #! and always return the top right corner. This is not ideal but prevents a crash
            print("(find_marker_center_and_orientation) Division by 0 : returning tr instead of center")
            return *tl, theta
        
        #we're gonna calculate the interception of the diagonals to find the center
        tl,tr,br,bl = marker
        alpha = (tr[1] - bl[1]) / (tr[0]-bl[0]) #slope of bl->tr diagonal is : delta_y / delta_x = y2-y4 / x2-x4
        beta = (tl[1]-br[1]) / (tl[0] - br[0]) #slope of br->tl diagonal is y1-y3 / x1 - x3
        #if you find the slope and intercept of each diagonal, equate them and isolate x, you get this
        #do the math yourself
        x_center = (tl[1] - tr[1] - beta*tl[0] + alpha*tr[0]) / (alpha - beta)  
        y_center = alpha*x_center + (tr[1] - alpha*tr[0]) # use the equation of the bl->tr diagonal to find y_center

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
        maxWidth, maxHeight = self.map_size
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts.astype('float32'), dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # warped = cv2.putText(warped, f"{maxWidth} x {maxHeight}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
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

        #print(f"{len(corners)} detected corners, {in_memory} corners stored")

    def draw_path_on_image(self, path, past_kalman_estimates=[]):
        """
        Dessine le chemin calculé par Dijkstra sur l'image donnée.
        Draw the dijsktra path in green on the image, and all past positions (path already travelled) as red dots.

        :param image: Image sur laquelle dessiner
        :param path: Liste des points [x, y] représentant le chemin
        :return: Image avec le chemin dessiné
        """
        img_with_path = self.top_view.copy()
        for i in range(len(path) - 1):
            start = tuple(map(int, path[i]))
            end = tuple(map(int, path[i + 1]))
            cv2.line(img_with_path, start, end, (0, 255, 0), 2)  # Vert pour le chemin
        
        #draw red points for points already travelled
        if past_kalman_estimates:
            for position in past_kalman_estimates:
                # try:
                    # if position.any() != [-1,1]:
                cv2.circle(img_with_path, np.array(position,dtype="int32"), 2, (0,0,255), -1)
                # except:
                #     print(type(position), position)
                #     if position.any() != [-1,1]:
                #         pass
        return img_with_path

    def detect_thymio_pose(self,thymio_marker):
        '''Returns [x,y,theta] of the thymio marker center. X and Y would be int, theta float'''
        if thymio_marker is not None:
            x,y,theta = self.find_marker_center_and_orientation(thymio_marker)
            # we return the position and orientation of the thymio
            self.last_thymio_pose = tuple((x,y,theta))
            #return np.array([x,y], dtype='int32'), theta
            return [x,y,theta]
        
        x,y,theta = self.last_thymio_pose
        #return np.array([x,y], dtype='int32'), theta
        return [x,y,theta]

    def detect_goal_position(self, goal_marker):
        ''' Return int array [x,y,z=0] corresponding to goal position'''
        if goal_marker is not None:
            x,y,_ = self.find_marker_center_and_orientation(goal_marker)
            # we return the position of the goal
            self.last_goal_position = tuple((x,y))
            #return np.array([x,y], dtype='int32')
            return [x,y]
        x,y,_ = self.find_marker_center_and_orientation(goal_marker)
        #return np.array([x,y], dtype='int32')
        return [x,y]
    
    def get_2_markers(self, top_view_img):
        """
        Find 2 markers (Thymio and Goal) in the top-view image
        """
        markers, ids = self.detect_aruco(top_view_img)

        #if no markers have been detected
        if ids is None:
            print("(get_2_markers) Thymio and Goal markers not detected")
            return None, None
        else:
            # # Verify that we have at least the two desired markers
            # if not (len(ids) >= 2):
            #     print(f"(get_2_markers) Detected {len(ids)} markers instead of at least 2.")
            #     #return markers.squeeze(), ids.squeeze()
            if not(4 in ids):
                print("get_2_markers) Thymio not detected")
                return None, markers
            if not(5 in ids):
                print("(get_2_markers) Goal not detected")
                return markers, None

        # Identify markers by their IDs: assume Thymio (e.g., ID=4) and Goal (e.g., ID=5)
        thymio_marker = None
        goal_marker = None
        for marker, marker_id in zip(markers, ids):
            if marker_id == 4:  # Replace '4' with the actual ID for Thymio
                thymio_marker = marker
            elif marker_id == 5:  # Replace '5' with the actual ID for Goal
                goal_marker = marker

        # if thymio_marker is None:
        #     print("Unable to detect Thymio.")
        #     #return False
        # if goal_marker is None:
        #     print("Unable to detect the Goal marker.")
            #return False
        return thymio_marker, goal_marker

    def julien_main(self, img):
        ''' ThIS SHOULD NOT STAY ! I only put it as an example of how my code is supposed to be used '''

        #get location of 6 markers (and their ids) in camera frame
        #NB this function, and all the rest, *should* still work if some markers aren't detected
        markers, ids = self.get_6_markers(img)

        if ids is None:   #if no markers have been detected
            if np.any(self.map_corners == -1): #and one of the 4 corners has not been initialized yet (default value [-1,-1])
                self.top_view = img            #just return the original image
                return 
            #if all 4 corners have been detected previously, use their previous value to get the top view
            corners = self.map_corners
            self.top_view, _ = self.four_point_transform(img, corners)
            return

        #find the coordinates of the 4 map corners (4x2 array).
        corners = self.get_map_corners(markers, ids)
        top_view_img, four_point_matrix = self.four_point_transform(img,corners)
        self.top_view = top_view_img
        return 


class CameraFeed(threading.Thread):
    """Thread pour capturer et afficher un flux vidéo constant"""
    def __init__(self, analysis_module):
        super().__init__()
        self.vision_module = analysis_module
        self.stop_event = threading.Event()
        self.robot_pose = []
        self.goal_position = []
        self.obstacle_corners = []
        self.show_which = [1,1,1,1,1,1]
        self.past_kalman_estimates= []

    # def get_thymio_goal_and_obstacles(self):
    #     '''Returns [x,y,theta] of thymio, [x,y] of goal and list of obstacle 
    #     corners = [[corner1,corner2],[corner1,corner2,corner3,corner4],[corner1,corner2,corner3]]
    #     where cornerX is itself a list of two coordinates [x_corner1,y_corner1].
    #     If the corresponding markers are not detected, returns empty lists []'''
    #     return self.robot_pose, self.goal_position, self.obstacle_corners

    def run(self):
        while not self.stop_event.is_set():
            frame = self.vision_module.capture_frame()
            if frame is not None:
                # Utilisation de la méthode d'analyse de Analysis
                #processed_frame = self.vision_module.analyze_frame(frame)
                # Appeler la méthode pour détecter les coins des obstacles

                self.vision_module.julien_main(frame)
                if self.vision_module.frame_viz is None:
                    self.vision_module.frame_viz = frame

                annotated_img = self.vision_module.frame_viz
                top_view = self.vision_module.top_view
                dijkstra_path_view = top_view

                if self.vision_module.path :
                    dijkstra_path_view = self.vision_module.draw_path_on_image(self.vision_module.path, self.past_kalman_estimates
                    )
                else:
                    print("No path")
                #output variables :>
                # - [x,y,theta] of thymio - [x,y] of goal   -list of obstacle corners (list of list ?)
                #if they have not been detected, will return empty list []
                robot_pose = []
                goal_position = []
                obstacle_corners = []


                #! alternative : if not detected, will return last known pose and goal instead
                #! Julien thinks it's not a good idea
                # robot_pose = self.vision_module.last_thymio_pose
                # goal_position = self.vision_module.last_goal_posq

                obstacle_corners, filtered_mask, img_with_polygons = self.vision_module.detect_obstacle_corners(top_view)

                Thymio_marker, goal_marker = self.vision_module.get_2_markers(top_view)
                if Thymio_marker is not None: 
                    robot_pose= self.vision_module.detect_thymio_pose(Thymio_marker)  

                    arrow_length = 100
                    # Calculate the end point of the arrow using the angle theta
                    end_x = int(robot_pose[0] + arrow_length * np.cos(robot_pose[2]))
                    end_y = int(robot_pose[1] + arrow_length * np.sin(robot_pose[2]))
                    top_view = cv2.arrowedLine(top_view.copy(), np.array(robot_pose[:2],dtype='int32'), (end_x, end_y), (0, 0, 255), 5)
                else:
                    print("Thymio not detected")
           
                if goal_marker is not None:
                    goal_position = self.vision_module.detect_goal_position(goal_marker)
                    if Thymio_marker is not None:
                        top_view = cv2.arrowedLine(top_view.copy(), np.array(robot_pose[:2],dtype='int32'), np.array(goal_position, dtype='int32'), (255, 0, 0), 8)
                else:
                    print("Goal not detected")

                #update attributes
                self.obstacle_corners = obstacle_corners
                self.robot_pose = robot_pose
                self.goal_position = goal_position

    
                
                videofeeds_list = [frame, filtered_mask, img_with_polygons, annotated_img, top_view, dijkstra_path_view]
                titles_list = ["Original", "filtered_mask", "Processed_with_polygones", "Highlighting corners", "thymio Oops, baby", "dijkstra_path_view"]
                
                # Afficher les deux images en parallèle
                if len(videofeeds_list) == len(self.show_which):
                    v_l, t_l = [], []
                    for i in range(len(titles_list)):
                        if self.show_which[i]:
                            v_l.append(videofeeds_list[i])
                            t_l.append(titles_list[i])
                    videofeeds_list, titles_list = v_l, t_l
                            
                else:
                    print(f"[Vision.camerafeed.run()] List of videofeeds must be {len(titles_list)}, not {len(self.show_which)}."\
                          "Defaulting to show all")

                show_many_img(videofeeds_list, titles_list)

                # Quitter si la touche 'q' est pressée
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

    def stop(self):
        """Arrêter le thread et libérer la caméra"""
        self.stop_event.set()
        self.vision_module.release_camera()

class Vision():
    """Thread pour capturer et afficher un flux vidéo constant"""
    def __init__(self):
    # Instantiate CameraFeed and Analysis
        self.analysis = Analysis()
        self.camera_feed = CameraFeed(self.analysis)
        # self.stop_event = threading.Event()

    def begin(self, show_which=[1,1,1,1,1,1]):
        if not self.analysis.initialize_camera(cam_port=4):
            print("Erreur : Impossible d'initialiser la caméra.")
            return
        self.camera_feed.show_which = show_which
        self.camera_feed.start()

    def stop(self):
        self.camera_feed.stop()

    def get_thymio_goal_and_obstacles(self):
        '''Returns [x,y,theta] of thymio, [x,y] of goal and list of obstacle 
        corners = [[corner1,corner2],[corner1,corner2,corner3,corner4],[corner1,corner2,corner3]]
        where cornerX is itself a list of two coordinates [x_corner1,y_corner1].
        If the corresponding markers are not detected, returns empty lists []'''
        robot_pose = self.camera_feed.robot_pose
        goal_position = self.camera_feed.goal_position
        obstacle_corners = self.camera_feed.obstacle_corners
        return robot_pose, goal_position, obstacle_corners


def main():
    # """Point d'entrée principal"""
    # vision = Analysis()
    # if not vision.initialize_camera(cam_port=4):
    #     print("Erreur : Impossible d'initialiser la caméra.")
    #     return

    # camera_thread = CameraFeed(vision)
    # camera_thread.start()

    # try:
    #     while True:
    #         # Vous pouvez exécuter d'autres tâches en parallèle ici
    #         # Le programme principal continue de tourner sans bloquer l'affichage
    #         print("Le programme principal fonctionne en arrière-plan...")
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Arrêt du programme demandé.")
    # finally:
    #     camera_thread.stop()
    #     camera_thread.join()
    #     print("Programme terminé.")
    visio = Vision()
    visio.begin()
    #show_many_img([visio.analysis.frame], ["titre"])


if __name__ == "__main__":
    main()



##################################################################
##################################################################
"""

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
"""