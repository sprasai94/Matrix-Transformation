import numpy as np
import cv2
import math as m


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def readImage(image):
    img = cv2.imread(image)
    return img


def saveImage(name, image):
    cv2.imwrite(name, image)


def transformation_rotation(img, rot_matrix, w_new, h_new):
    #Original image height and dwidth
    h_real, w_real = img.shape[:2]
    #original image center and rotated image center
    center_old = Point(w_real/2, h_real/2)
    center_new = Point(w_new / 2, h_new / 2)

    rotated_Image = np.zeros((h_new, w_new, 3), dtype=np.uint8)
    for i in range(w_new):
        for j in range(h_new):
            # finding the pixel coordinate (y,x) using rotation matrix that is going to be rotated to each final position (j,i)
            x = int(((i -center_new.x) * rot_matrix[0,0] -(j - center_new.y) * rot_matrix[0,1]) + center_old.x)
            y = int(((i -center_new.x) * rot_matrix[1,0] + (j - center_new.y) * rot_matrix[1,1]) + center_old.y)

            # filling each pixel (j,i) of new image with pixel (y,x) that will be transformed to position (j,i)
            if 0 < x < w_real and 0 < y < h_real:
                rotated_Image[j,i] = img[y,x]

    return rotated_Image



def transformation_scaling(image, scaling_matrix, width, height):
    transformedimage = np.zeros((height, width, 3), dtype=np.uint8)
    h_real, w_real = image.shape[:2]
    for i in range(width):
        for j in range(height):
            #finding the pixel coordinate (y,x) using scaling matrix that is going to be scaled to each final position (j,i)
            x = int(i * scaling_matrix[0, 0] + j * scaling_matrix[0, 1])
            y = int(i * scaling_matrix[1, 0] + j * scaling_matrix[1, 1] )

            #filling each pixel (j,i) of new image with pixel (y,x) that will be transformed to position (j,i)
            if 0 < x < w_real and 0 < y < h_real:
                pixel = image[y, x]
                transformedimage[j, i] = pixel
    return transformedimage


def rotation_function(img,h,w,angle):
    center = Point(w / 2, h / 2)

    # Forming Rotation matrix
    costheta = m.cos(angle * m.pi / 180)
    sintheta = m.sin(angle * m.pi / 180)
    rot_Matrix = np.matrix([[costheta, sintheta], [sintheta, costheta]])

    # calculate new height and width after rotation
    new_Width = int((h * abs(sintheta)) + (w * abs(costheta)))
    new_Height = int((h * abs(costheta)) + (w * abs(sintheta)))


    finalimage = transformation_rotation(img, rot_Matrix, new_Width, new_Height)
    return finalimage


def translation_function(image, width, height, t):
    translatedImage = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            #Translate the pixel by given translation vector
            x = i + t.x
            y = j + t.y
            #copy the pixel value to the tranlated pixel coordinate
            if 0 < x < width and 0 < y < height:
                pixel = image[j, i]
                translatedImage[y, x] = pixel
    return translatedImage


def scaling_function(image, height, width, scale):

    # Forming scaling matrix
    scaling_Matrix = np.matrix([[(1/scale), 0.0],[0.0, (1/scale)]])
    #print('Scaling Matrix:', scaling_Matrix)

    #Finding new height and width after scaling
    new_Width = int(width*scale)
    new_Height = int(height*scale)


    scaledImage = transformation_scaling(image, scaling_Matrix, new_Width, new_Height)
    return scaledImage



if __name__ == "__main__":
    #Select the transformation setting to True to see transformation
    rotation = True
    translation = False
    scaling = False

    #Read Input image file
    img = readImage("image.png")
    #determine dimension of image
    height, width = img.shape[:2]
    print('The size of image:',height,width)

    if rotation:
        angle = 45
        finalImage = rotation_function(img, height, width,angle)
        saveImage("Output1.png", finalImage)


    if translation:
        transPoint = Point(-10,-10)
        finalImage = translation_function(img,width,height,transPoint)
        saveImage("Output3.png", finalImage)

    if scaling:
        scale = 0.5
        finalImage = scaling_function(img, height, width, scale)
        saveImage("Output2.png",finalImage)

    cv2.imshow('FinalImage:', finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()