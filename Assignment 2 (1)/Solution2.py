#A :50.56%
#B :69.94%
#C :82.30%

import cv2
import sys
import numpy as np

def restore_image(input_file, output_file):

    #Read the input image
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    #Blur the image by taking the median in the neighbourhood   
    blur_image = cv2.medianBlur(input_image,5)
    
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]], dtype=np.float32)

    #Apply convolution with the Composite Laplacian kernel
    laplacian_output = cv2.filter2D(blur_image, -1, laplacian_kernel)

    #Apply subtraction between the laplacian_output and blur_image
    subtract_image = cv2.subtract(blur_image, laplacian_output)

    #Apply addition between the subtract_image and blur_image
    add_image = cv2.add(blur_image,subtract_image)

    restore_image = add_image

    #Display the restore_image
    cv2.imshow('restore_image',restore_image)

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
