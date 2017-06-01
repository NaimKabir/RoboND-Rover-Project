import numpy as np
import cv2

world_size = 200
scalefactor = 10

# Identify pixels above and below the bounds you set
def color_thresh(img, rgb_thresh=(160, 160, 160), rgb_cap=(256, 256, 256), blur_factor=5):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) & (img[:, :, 0] <= rgb_cap[0]) \
                   & (img[:, :, 1] > rgb_thresh[1]) & (img[:, :, 1] <= rgb_cap[1]) \
                   & (img[:, :, 2] > rgb_thresh[2]) & (img[:, :, 2] <= rgb_cap[2])

    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1

    # Return the binary image
    return color_select

#Include a blurring functionality
def blur_binarized_img(img, blur_factor = 5):

    # set a blur to smooth the threshed image
    kernel = np.ones((blur_factor, blur_factor), np.float32) / 25
    blurred_binary = cv2.filter2D(img, -1, kernel) > 0.5

    return blurred_binary

# Set a proximity threshold on a binarized image.
# Will black out anything pixdistance away from a starttuple.
def proximity_thresh(img, start_tuple, pixdistance):
    startx, starty = start_tuple
    xs, ys = img.nonzero()
    blacked_out = np.array([np.sqrt((startx - x) ** 2 + (starty - y) ** 2) <= pixdistance for x, y in zip(xs, ys)]) * 1

    for (x, y), value in zip(zip(xs, ys), blacked_out):
        img[x, y] = value

    return img

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw = (yaw/180)*np.pi
    # Apply a rotation
    xpix_rotated = xpix*np.cos(yaw) - ypix*np.sin(yaw)
    ypix_rotated = xpix*np.sin(yaw) + ypix*np.cos(yaw)
    # Return the result
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform

    # Src and Dst for perspective transform are set in the Rover's state, since its invariable.

    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, Rover.perspective_source, Rover.perspective_destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples

    # Applying my color threshold, gaussian blur, and then a proximity threshold for each class I want to detect
    navigable =  blur_binarized_img(color_thresh(warped, (160, 160, 160), (256,256,256)), 4)
    navigable = proximity_thresh(navigable, start_tuple = (navigable.shape[0], navigable.shape[1]/2), pixdistance=60)

    rocks = color_thresh(warped, (135,115,10), (210,185,45)) #No need to blur over tiny rocks
    rocks = proximity_thresh(rocks, start_tuple = (rocks.shape[0], rocks.shape[1]/2), pixdistance=80)

    # obstacles = color_thresh(warped, (0,0,0), (62,56,52)) #No need to blur over obstacle rocks
    # obstacles = proximity_thresh(obstacles, start_tuple = (obstacles.shape[0], obstacles.shape[1]/2), pixdistance=40)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,1] = rocks*255
    Rover.vision_image[:,:,2] = navigable*255

    # 5) Convert map image pixel values to rover-centric coords
    navigableXrov, navigableYrov = rover_coords(navigable)
    rocksXrov, rocksYrov = rover_coords(rocks)

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles  = to_polar_coords(navigableXrov, navigableYrov)

    # 6) Convert rover-centric pixel values to world coordinates
    navigableX, navigableY = pix_to_world(navigableXrov, navigableYrov, \
                                      Rover.pos[0], Rover.pos[1], \
                                      Rover.yaw, world_size, scalefactor)

    rocksX, rocksY = pix_to_world(rocksXrov, rocksYrov, \
                                          Rover.pos[0], Rover.pos[1], \
                                          Rover.yaw, world_size, scalefactor)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[rocksY, rocksX, 0] = 155
    Rover.worldmap[navigableY, navigableX, 2] = 155



    return Rover