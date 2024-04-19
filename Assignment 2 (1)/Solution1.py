#A :47.88%
#B :62.75%
#C :48.38%

import cv2
import sys
import numpy as np

def restore_image(input_file, output_file):
    
    #Read the input image
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    
    #Blur the image by taking median in the neighbourhood
    blur_image = cv2.medianBlur(input_image, 3)

    #Create a CLAHE object (Clip Limit and Grid Size can be adjusted)
    clahe = cv2.createCLAHE(clipLimit=0.0005, tileGridSize=(1, 1))

    #Apply CLAHE to the input image
    clahe_image = clahe.apply(blur_image)

    #Blur the image by taking median in the neighbourhood
    blur2_image = cv2.medianBlur(clahe_image, 5)
    
    restore_image = blur2_image

    show_image = np.hstack((input_image,restore_image))

    # Write the output image
    cv2.imwrite(output_file + '.jpg', restore_image)

    # Display input and output images
    cv2.imshow('Input | Restored',show_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python restore_combined.py input_image output_image")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        restore_image(input_image_path, output_image_path)
