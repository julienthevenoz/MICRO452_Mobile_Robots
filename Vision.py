import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans ##### librairie pour le K-mean, le PCA,... #####


def show_img(img, title, wait_ms=1):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    # This allows the window to refresh without blocking the program
    key = cv2.waitKey(wait_ms) & 0xFF
    # Optionally handle 'q' to quit the display loop in the main program
    if key == ord('q'):
        return False
    return True


def show_many_img(img_list, title_list):
    if len(img_list) != len(title_list):
        print("Unequal list size")
        return
    for i in range(len(img_list)):
        cv2.namedWindow(title_list[i], cv2.WINDOW_NORMAL)
        cv2.imshow(title_list[i], img_list[i])

    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


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
        print("Result : ",result)
        if result:
            self.img = image
        # print(result)
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
        markerCorners = squeeze(markerCorners)#np.array(markerCorners).squeeze() 
        
        return markerCorners, markerIds
    
    def get_6_markers(self,img):
        '''Find 6 markers and return 3D array of them sorted in the order :
        [TL, TR, BR, BL, Thymio, Goal]. Dim1 is markers, dim2 is corners of the marker
        dim3 is x and y of the corner'''
        markers, ids = self.detect_aruco(img)
        #verifiy that we have the 6 markers
        if markers.shape == (0,):
            print("Detected 0 markers")
            return squeeze(markers), squeeze(ids)
        if not(len(ids)==6):
            print(f"Detected {len(ids)} markers instead of 6")

        #if we have multiple markers, we need to order them    
        if not(len(ids)== 1):
            pairs = sorted(list(zip(ids,markers))) #make a list of corresponding [id,marker] pairs, then sort it
            ids, markers = zip(*pairs) #unzip the pairs : ids are now in order 0-5 and the corresponding aruco corner markers are in the same order
   
        return squeeze(markers) , squeeze(ids)
        
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
            center = center[:2]
            # center = self.find_marker_center_and_orientation(markers[corner_id])[:2]  #we only keep x and y, not theta
            self.map_corners[corner_id,:] = center

        # self.map_corners = np.array(corners_of_markers)  #save the pixel coordinates of the map corners
        self.highlight_corners(self.img.copy(),self.map_corners)
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
        top_side = np.array([tr[0]-tl[0],tr[1]-tl[1]])
        top_side = top_side / np.linalg.norm(top_side) #normalize the vector
        unit_hvec = np.array([1,0])  #unitary horizontal vector
        theta = np.arccos(np.dot(top_side,unit_hvec)) # v1 dot v2 = ||v1||*||v2||*cos(theta) = cos(theta) if vects are unitary
        return x_center,y_center, theta

    def detect_thymio_pose(self,markers):
        '''Returns integer position array [x,y,z=0] and float theta corresponding to Thymio position and orientation '''
        thymio_marker = (markers[4])
        x,y,theta = self.find_marker_center_and_orientation(thymio_marker)
        #we return the position with 
        return np.array([x,y,0], dtype='int32'), theta 
    

    def detect_goal_position(self, markers):
        ''' Return int array [x,y,z=0] corresponding to goal position'''
        goal_marker = markers[5]
        x,y,_ = self.find_marker_center_and_orientation(goal_marker)
        return np.array([x,y,0], dtype='int32')

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
    
    def highlight_corners(self, image, corners, title='highlighted corners', show=True):
        '''Draws a circle around the corners detected in this iteration,
        and a rectangle around the corners remebered/stored in self.map_corners '''
        img = image.copy()
        corners = squeeze(corners, type='int32') #np.array(corners, dtype='int32').squeeze()
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]  #red, green, blue, yellow
        for i,corner in enumerate(corners):
            highlighted_corners = cv2.circle(img,corner, 20, colors[i],4) #this is just for debugging
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
            show_img(highlighted_corners,"Corners : o = detected    [] = remembered")

        print(f"{len(ids)} detected corners, {in_memory} corners stored")



if __name__ == "__main__":
    filename = 'Photos/Photo7.jpg'
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    visio = Vision_module(img)
    visio.initialize_camera(cam_port=5)

    # contours = visio.extract_edge(img)
    # corners = visio.find_map_corners(contours)
    #now that we have the 4 corners of the map, we have to order them
    # corners = visio.order_points(corners)
    #now we can apply the transform to get a perpendicular top-view
    # markers, ids = visio.get_6_markers(img)
    # corners = visio.get_map_corners(markers, ids)
    # top_view_img, four_point_matrix = visio.four_point_transform(img,corners)
    # thymio_pose, thymio_theta = visio.detect_thymio_pose(markers)
    # goal_pos = visio.detect_goal_position(markers)
    # #let's put a green circle on top of the goal
    # original_image = cv2.circle(img, goal_pos[:2], 20, (0,255,0), 5)
    # #let's draw an arrow from the center of the thymio in the direction where it's looking
    # arrow_length = 100 #in pixels
    # start = thymio_pose[:2]
    # end = (int(start[0] + arrow_length*np.cos(thymio_theta)), int(start[1] + arrow_length*np.sin(thymio_theta)))
    # # end = start + (arrow_length*np.cos(thymio_pose[2]),arrow_length*np.sin(thymio_pose[2]))
    # original_image = cv2.arrowedLine(original_image,start, end, (255,0,0), 20)
    # #let's put a green circle on top of the goal
    # original_image = cv2.circle(original_image, goal_pos[:2], 20, (0,255,0), 5)
    # show_img(original_image,'OG image')


    # #use the transform from original image to warped image to transform all the corresponding points and display them
    # #on the map. 
    # new_thymio = four_point_matrix @ thymio_pose
    # # new_thymio_angle = thymio_theta - (np.pi)/2
    # new_goal = (four_point_matrix @ goal_pos).astype('int32')
    # top_view_img = cv2.circle(top_view_img, new_goal[:2], 20, (0,255,0), 5)
    # start = np.array(new_thymio[:2]).astype(int)
    # end = (int(start[0] + arrow_length*np.cos(thymio_theta)), int(start[1] + arrow_length*np.sin(thymio_theta)))
    # top_view_img = cv2.arrowedLine(top_view_img,start, end, (255,0,0), 20) #the angle should be off
    # show_img(top_view_img, 'top view')

    # ###convert everything to map scale
    # pts = visio.rescale_points([*corners, thymio_pose[:2], goal_pos])
    # *r_corners, r_thymio, r_goal = pts
    # r_goal = np.array(r_goal).astype(int)
    # top_view_img = cv2.circle(top_view_img, r_goal, 20, (0,255,0), 5)
    # print(0)


    import time
    i=0
    while True:
        i+=1
        print(i)
        rslt,img = visio.capture_image()
        if rslt:
            visio.img = img
            markers, ids = visio.get_6_markers(img)
            corners = visio.get_map_corners(markers, ids)
        time.sleep(0.1)    

   

    # while True:
    #     print("Capturing image...")
    #     rslt, img = visio.capture_image()
    #     if rslt:
    #         visio.img = img
    #         if not show_img(img, 'Live Feed'):
    #             break  # Exit if 'q' is pressed
    #     time.sleep(0.1)  # Add a small delay to reduce CPU usage
    
    # Cleanup
    visio.cam.release()
    cv2.destroyAllWindows()
