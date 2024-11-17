##### interet de faire une class Vision_Thymio #####
##### le optimal path du coup il va ici où dans le fichier filtering ou ailleur? #####
# Camera & Image Processing

def initialize_camera():
    return

def capture_image():
    return

# Map definition

def Image_correction(image): #distortions or perspective distortions
    return

def detect_map_corners(image):
    return

##### https://evergreenllc2020.medium.com/building-document-scanner-with-opencv-and-python-2306ee65c3db #####

def map_rescaling ():
    return image

# Object detection
##### idée : faire une map de bateau, notre thymio c'est un bateau pirate qui doit atteindre une ile ou un trésors et il doit se déplacer sur l'eau (sol bleu) avec des obstacle en mode gros rocher (noir) #####
##### comment on défini notre image? comme une grille? autrement?...#####
##### est-ce qu'on met différentes couleurs à nos obstacle, sur notre thymio, goal,..#####
def color_filter():
    return image

def detect_thymio_position(image):
    return

def detect_goal_position(image):
    return

def detect_obstacles(image):
    returnimport cv2
import numpy as np
import matplotlib.pyplot as plt 


#Magic numbers
#NB : opencv uses convention of [0,179], [0,255] and [0,255] for HSV values instead of the common [0,360],[0,100], [0,100]
UPPER_GREEN = np.array([120,255,255], dtype='uint8')    
LOWER_GREEN = np.array([70,0,0], dtype='uint8')

UPPER_BROWN = np.array([60,255,255])
LOWER_BROWN = np.array([0,0,0])


def show_img(img,title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)  # Wait for a key press to close the window
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

    def get_colour_mask(self, img, lower, upper):
        original = img.copy()
        img = cv2.GaussianBlur(img, (5, 5), 0)  #apply gaussian smoothing
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #transform to Hue(=color)-Saturation-Value(=brightness) format to make color detection easier
        mask = cv2.inRange(img, lower, upper)
        show_img(mask,"mask")
        return mask



    def extract_edge(self, img):
        # img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        show_img(img,'supposedly masked image')
        blurred_img = cv2.GaussianBlur(img, ksize=[5,5],sigmaY=10,sigmaX=10)  
        show_img(blurred_img,"blurred img")
        cannied_img = cv2.Canny(blurred_img,100,200)
        show_img(cannied_img, "canny edge image")
        return cannied_img




if __name__ == "__main__":
    filename = 'Photos/Photo1.jpg'
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    visio = Vision_module(img)
    mask = visio.get_colour_mask(img, LOWER_GREEN, UPPER_GREEN)
    masked_img = cv2.bitwise_and(img, img, mask=mask)  #apply color mask (keep only green pixels)

    cannied_img = visio.extract_edge(masked_img)
    
    # # Show the images in separate windows
    # cv2.imshow('basic', img)
    # cv2.imshow('blurred', blurred_img)

    # while True:
    #     # Check if both windows are still open
    #     if cv2.getWindowProperty('basic', cv2.WND_PROP_VISIBLE) < 1 and \
    #     cv2.getWindowProperty('blurred', cv2.WND_PROP_VISIBLE) < 1:
    #         break

    #     # Optionally, allow the user to press Esc to close
    #     key = cv2.waitKey(1)
    #     if key == 27:  # 27 = ESC key
    #         break

    # # Cleanup
    # cv2.destroyAllWindows()
