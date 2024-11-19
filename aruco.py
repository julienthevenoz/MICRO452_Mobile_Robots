import cv2
import cv2.aruco as aruco
import numpy as np

def generate_aruco_marker(marker_id, size=200, output_file=None):
    """
    Generate an ArUco marker with the specified ID and save it as an image.

    :param marker_id: The ID of the marker (integer).
    :param size: The size of the marker image in pixels (square).
    :param output_file: Path to save the generated marker image (optional).
    :return: The generated marker image as a NumPy array.
    """
    # Use the default dictionary for 4x4 ArUco markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Generate the marker
    marker = np.zeros((size, size), dtype=np.uint8)
    marker = aruco.generateImageMarker(aruco_dict, marker_id, size)

    # Save the marker to a file if a path is provided
    if output_file:
        cv2.imwrite(output_file, marker)

    return marker

# Generate 6 markers with IDs from 0 to 5
for marker_id in range(6):
    output_path = f"ArucoMarkers/{marker_id}.png"
    generate_aruco_marker(marker_id, size=300, output_file=output_path)
    print(f"Marker {marker_id} saved to {output_path}")
