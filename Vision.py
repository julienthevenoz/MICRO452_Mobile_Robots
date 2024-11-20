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
        as : Top-left, Top-right, Bottom-right, Bottom-left
        '''
        pts = np.array(pts) #transform in a numpy array
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
        # show_img(gray,'gray')
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        markerCorners, markerIds, _ = detector.detectMarkers(gray)
        if markerIds is not None:
            frame_markers = cv2.aruco.drawDetectedMarkers(img.copy(), markerCorners, markerIds)
            print(f"Detected markers: {markerIds.flatten()}")
            # show_img(frame_markers, 'frame markers')
        else:
            print("no markers")
        return markerCorners, markerIds
    
    def get_6_markers(self,img):
        '''Find 6 markers and return an array of them sorted in the order :
        [TL, TR, BR, BL, Thymio, Goal]'''
        markers, ids = self.detect_aruco(img)
        #verifiy that we have the 6 markers
        if not(len(ids)==6):
            print(f"Detected {len(ids)} markers instead of 6")
            #exit(1)
            return None
        pairs = sorted(list(zip(ids,markers))) #make a list of corresponding [id,marker] pairs, then sort it
        ids, markers = zip(*pairs) #unzip the pairs : ids are now in order 0-5 and the corresponding aruco corner markers are in the same order
        return markers
        
    def get_map_corners(self,markers):
        '''Gets the 4 corners of the map based on the marker positions'''
    
        #only keep the first 4 ones, corresponding to the corners
        markers = markers[:4]
        corners_of_markers = []
        for i in range(4):
            center,_ = self.find_marker_center_and_orientation(markers[i][0])
            marker_centers = cv2.circle(img,center, 10, (0,0,255),2) #this is just for debugging
            corners_of_markers.append(center)
        show_img(marker_centers,"centers of markers")
        return corners_of_markers
    
    def find_marker_center_and_orientation(self, marker):
        '''Returns the center point of four coordinates (must be given
        in TL,TR,BR,BL order)'''
        #we're gonna calculate the interception of the diagonals to find the center
        tl,tr,br,bl = marker
        alpha = (tr[1] - bl[1]) / (tr[0]-bl[0]) #slope of bl->tr diagonal is : delta_y / delta_x = y2-y4 / x2-x4
        beta = (tl[1]-br[1]) / (tl[0] - br[0]) #slope of br->tl diagonal is y1-y3 / x1 - x3
        #if you find the slope and intercept of each diagonal, equate them and isolate x, you get this
        #do the math yourself
        x_center = (tl[1] - tr[1] - beta*tl[0] + alpha*tr[0]) / (alpha - beta)  
        y_center = alpha*x_center + (tr[1] - alpha*tr[0]) # use the equation of the bl->tr diagonal to find y_center

        #now we will use the dot product to find the relative angle between the top side of the marker (tl->tr) and the horizontal
        top_side = np.array([[tr[0]-tl[0]],[tr[1]-tr[1]]])
        top_side = top_side / np.linalg.norm(top_side) #normalize the vector
        unit_hvec = np.array([1,0])  #unitary horizontal vector
        theta = np.arccos(np.dot(top_side,unit_hvec)) # v1 dot v2 = ||v1||*||v2||*cos(theta) = cos(theta) if vects are unitary
        return (int(x_center),int(y_center), theta)
        

    def detect_thymio_pose(self,markers):
        '''Returns (x,y,theta) corresponding to Thymio position and orientation '''
        thymio_marker = markers[4]
        return self.find_marker_center_and_orientation(thymio_marker)
    

    def detect_goal_position(self, image):
        ''' Return (x,y) corresponding to goal position'''
        goal_marker = markers[5]
        x,y,_ = self.find_marker_center_and_orientation(goal_marker)
        return (x,y)

    def detect_obstacles(image):    #def les bords des obstacles
        return

    def map_rescaling (self):   #### a definir ou on le met exactement 
        return 

if __name__ == "__main__":
    filename = 'Photos/Photo6.jpg'
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    visio = Vision_module(img)
    # mask = visio.get_colour_mask(img, LOWER_GREEN, UPPER_GREEN)
    # masked_img = cv2.bitwise_and(img, img, mask=mask)  #apply color mask (keep only green pixels)

    # K-means segmentation
    # corners, ids = visio.detect_aruco(img)
    # segmented_img, labels = visio.kmeans_color_segmentation(img, n_clusters=3)

    # contours = visio.extract_edge(img)
    # corners = visio.find_map_corners(contours)
    #now that we have the 4 corners of the map, we have to order them
    # corners = visio.order_points(corners)
    #now we can apply the transform to get a perpendicular top-view
    markers = visio.get_6_markers(img)
    corners = visio.get_map_corners(markers)
    top_view_img = visio.four_point_transform(img,corners)
    thymio_pose = visio.detect_thymio_pose(markers)
    goal_pos = visio.detect_goal_position(markers)

    #let's draw an arrow from the center of the thymio in the direction where it's looking
    arrow_length = 30 #in pixels
    start = thymio_pose[:2]
    end = start + [arrow_length*np.cos(thymio_pose[2]),arrow_length*np.sin(thymio_pose[2])]  
    annotated_top_view = cv2.arrowedLine(top_view_img,start, end, (255,0,0), 2)
    #let's put a green circle on top of the goal
    annotated_top_view = cv2.circle(annotated_top_view, goal_pos, 20, (0,255,0), 2)
    
    show_img(annotated_top_view,'topi gang')

    
