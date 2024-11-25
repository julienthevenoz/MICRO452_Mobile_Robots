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
    def __init__(self, image=None, map_size=(1000,1000)):
        self.img = image
        self.cam = None
        #dim 0 is which corner (0,1,2,3 = tl,tr,br,bl), dim1 is x or y
        self.map_corners = np.ones((4,2),dtype='int32')*(-1)
        self.map_size = map_size  #arbitrary chosen metric of our map

    # Camera & Image Processing

    def initialize_camera(self, cam_port=4):  #on my PC, the webcam is in port 4
        self.cam = cv2.VideoCapture(cam_port)
        return

    def capture_image(self):
        result, image = self.cam.read()
        print(result)
        show_img(image, 'camera image')
        return result, image
    
    # Map definition

    def Image_correction(image): # distortions or perspective distortions, definition des bords et coin. Crop les bords de l'image
        return
    
    ##### https://evergreenllc2020.medium.com/building-document-scanner-with-opencv-and-python-2306ee65c3db #####
    
    def extract_edge(self, img): ##### à merge/repmplacer par Image_correction #####
        # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # show_img(img,'supposedly masked image')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # show_img(gray,"gray img")

        blurred_img = cv2.GaussianBlur(gray, ksize=[5,5],sigmaY=10,sigmaX=10)  
        show_img(blurred_img,"blurred img")

        cannied_img = cv2.Canny(blurred_img,75,200)
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
        # show_img(drawing,'Contours')
        return contours

    def get_colour_mask(self, img, lower, upper):
        original = img.copy()
        img = cv2.GaussianBlur(img, (5, 5), 0)  #apply gaussian smoothing
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #transform to Hue(=color)-Saturation-Value(=brightness) format to make color detection easier
        mask = cv2.inRange(img, lower, upper)
        # show_img(mask,"mask")
        return mask
    
    def kmeans_color_segmentation(self, img, n_clusters=3):
        """
        Segments the image into `n_clusters` regions using K-means clustering,
        and maps each region to a specific color.
        """
        # Step 1: Preprocess image
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian blur to reduce noise
        img_rgb = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1, 3)).astype('float32')  # Flatten image to a 2D array

        # Step 2: Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(pixels)
        centers = np.uint8(kmeans.cluster_centers_)  # Cluster centers (colors)
        segmented_pixels = centers[labels]  # Replace each pixel with its cluster center
        segmented_img = segmented_pixels.reshape(img.shape)  # Reshape to original image size

        # Step 3: Map clusters to specific colors
        color_map = {
            0: [0, 0, 0],        # Obstacle - Black
            1: [0, 130, 180],   # Background - Brownish
            2: [0, 220, 230],    # Robot - Cyan
            3: [60, 255, 0],     # Water - Green
        }
        result_img = np.zeros_like(segmented_img)  # Initialize result image
        labels_reshaped = labels.reshape(img.shape[:2])  # Reshape labels to 2D (H x W)

        for i in range(n_clusters):
            mask = (labels_reshaped == i)  # Mask for current cluster
            result_img[mask] = color_map.get(i, [255, 255, 255])  # Assign color based on cluster

        # Step 4: Display the original and segmented images
        show_many_img([img, segmented_img, result_img], ["Original Image", "Segmented (Raw)", "Segmented (Mapped Colors)"])

        return result_img, labels_reshaped


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

    def detect_aruco(sel, img):
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
        markerCorners = np.array(markerCorners).squeeze() 
        
        return markerCorners, markerIds
    
    def get_6_markers(self,img):
        '''Find 6 markers and return 3D array of them sorted in the order :
        [TL, TR, BR, BL, Thymio, Goal]. Dim1 is markers, dim2 is corners of the marker
        dim3 is x and y of the corner'''
        markers, ids = self.detect_aruco(img)
        #verifiy that we have the 6 markers
        if not(len(ids)==6):
            print(f"Detected {len(ids)} markers instead of 6")
            #exit(1)
            return markers.squeeze(), ids.squeeze()
        pairs = sorted(list(zip(ids,markers))) #make a list of corresponding [id,marker] pairs, then sort it
        ids, markers = zip(*pairs) #unzip the pairs : ids are now in order 0-5 and the corresponding aruco corner markers are in the same order
        markers = np.array(markers).squeeze()  #get rid of the useless exterior array
        ids = np.array(ids).squeeze()
        return markers, ids

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

    def get_map_corners(self,markers, ids):
        '''Gets the 4 corners of the map based on the marker positions'''
    
        #only keep the first 4 ones, corresponding to the corners
        markers = markers[:4]
        # corners_of_markers = []
        for i in range(4):
            corner_id = ids[i]
            if corner_id >= 4:  #markers 4 and 5 are not corners
                break
            center = self.find_marker_center_and_orientation(markers[corner_id])[:2]  #we only keep x and y, not theta
            self.map_corners[corner_id,:] = center
            # corners_of_markers.append(center)
            # if corner_id == 0:
            #     self.map_corners[0,0,:] = center
            # elif corner_id == 2:
            #     self.map_corners[0,1,:] = center
            # elif corner_id == 3:
            #     self.map_corners[1,0,:] = center
            # else:
            #     self.map_corners[1,1,:] = center

        # self.map_corners = np.array(corners_of_markers)  #save the pixel coordinates of the map corners
        # self.highlight_corners(self.img,self.map_corners)
        print(f"{len(ids)} markers detected, corners in memory are : {self.map_corners}")
        return self.map_corners
    
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
        # top_side = np.array([tr[0]-br[0],tr[1]-br[1]])
        # top_side = top_side / np.linalg.norm(top_side) #normalize the vector
        # unit_hvec = np.array([1,0])  #unitary horizontal vector
        # theta = np.arccos(np.dot(top_side,unit_hvec)) # v1 dot v2 = ||v1||*||v2||*cos(theta) = cos(theta) if vects are unitary
        theta = np.arctan2((tr[1]-br[1]), (tr[0]-br[0]))
        return x_center,y_center, theta

    def detect_thymio_pose(self,thymio_marker):
        '''Returns integer position array [x,y,z=0] and float theta corresponding to Thymio position and orientation '''
        x,y,theta = self.find_marker_center_and_orientation(thymio_marker)
        #we return the position with 
        return np.array([x,y], dtype='int32'), theta


    def detect_goal_position(self, goal_marker):
        ''' Return int array [x,y,z=0] corresponding to goal position'''
        x,y,_ = self.find_marker_center_and_orientation(goal_marker)
        return np.array([x,y], dtype='int32')

    def detect_obstacles(image):    #def les bords des obstacles
        return

    def map_rescaling (self):   #### a definir ou on le met exactement 
        return 
    
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
    
    def highlight_corners(self, image, corners, title='highlighted corners'):
        img = image.copy()
        corners = np.array(corners, dtype='int32').squeeze()
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]  #red, green, blue, yellow
        for i,corner in enumerate(corners):
            if corner[0] == -1:  #don't show corners which haven't been initialized
                continue 
            highlighted_corners = cv2.circle(img,corner, 20, colors[i],4) #this is just for debugging
        # for i in range(4):
        #     print(corners[i],colors[i])
        #     highlighted_corners = cv2.circle(img,corners[i], 20, colors[i],4) #this is just for debugging
        show_img(highlighted_corners,title)


if __name__ == "__main__":
    filename = 'Photos/Photo_test_aruco.jpg'
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    visio = Vision_module(img)

   
    markers, ids = visio.get_6_markers(img)
    corners = visio.get_map_corners(markers, ids)
    top_view_img, four_point_matrix = visio.four_point_transform(img,corners)

    Thymio_marker, goal_marker = visio.get_2_markers(top_view_img)
    robot_position, theta = visio.detect_thymio_pose(Thymio_marker)
    goal_position = visio.detect_goal_position(goal_marker)


    arrow_length = 500
    # Calculate the end point of the arrow using the angle theta
    end_x = int(robot_position[0] + arrow_length * np.cos(theta))
    end_y = int(robot_position[1] + arrow_length * np.sin(theta))
    top_view_img = cv2.arrowedLine(top_view_img, robot_position, (end_x, end_y), (0, 0, 255), 5)
    top_view_img = cv2.arrowedLine(top_view_img, robot_position, goal_position, (255, 0, 0), 20)

    show_img(top_view_img, 'Opps, baby')
   