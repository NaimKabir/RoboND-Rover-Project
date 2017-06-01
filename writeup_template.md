## Project: Search and Sample Return

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[fig1]: ./misc/fig1.png
[fig2]: ./misc/fig2.png
[fig3]: ./misc/fig3.png


### Notebook Analysis

#### Running the Notebook, and making modifications

The original notebook offered a scaffolding of useful functions, including warp, a simple color threshold, and some incomplete functions for rotating and scaling pixel coordinates from binary images.

My first order of business was filling out the rotate_pix and translate_pix to get a properly function 'pix_to_world' function.

At the same time, I changed the color-thresholding to allow for an upper RGB bound. This allowed me to threshold for navigable ground as well as sample rocks with separate usages of the function.

Then I got to work adding functionality. First up was a `blur_binarized_img` function that smoothed out the sometimes choppy results from a perspective transform.

It takes in an image and a desired 'blur factor' to smooth out the images.

```python
#Include a blurring functionality
def blur_binarized_img(img, blur_factor = 5):

    # set a blur to smooth the threshed image
    kernel = np.ones((blur_factor, blur_factor), np.float32) / 25
    blurred_binary = cv2.filter2D(img, -1, kernel) > 0.5

    return blurred_binary
```

Here's the difference it can make:

*Unblurred*
![unblurred][fig1]

*Blurred*
![blurred][fig2]

This simple difference seems to make a difference when it comes to navigation and fidelity. Towards the same end I also wrote up a proximity thresholder that can help my rover ignore the stretched out artifacts of the perspective transform.

```python
#Set a proximity threshold on a binarized image. 
#Will black out anything pixdistance away from a starttuple.
def proximity_thresh(img, start_tuple, pixdistance):
    startx, starty = start_tuple
    xs, ys = img.nonzero()
    blacked_out = np.array([np.sqrt((startx-x)**2 + (starty-y)**2) <= pixdistance for x,y in zip(xs,ys)])*1

    for (x,y), value in zip( zip(xs,ys) , blacked_out):
        img[x,y] = value
    
    return img
```

I was also unsatisfied with the policy of giving the rover a direction by taking the mean of available navigable angles. So I created a function that could preferentially find likely navigable spots currently in view and orient towards them.

This function looks at the distribution of angles present in the view, and looks for peaks in that distribution. Then, given some 'desired angle', it orients the rover at the angle peak closest to the choice angle.

I want my rover to be a wall-hugger, so I have it preferentially always turn right. This way you could get full coverage of the map with minimal backtracking.


```python
#I don't know if mean angle is the greatest policy for choosing where to go next.
#This function enacts another policy: given a distribution of angles from the binarized image,
#it will choose the mean angle from the subset closest to a 'choice angle'.
def multimodal_angles(angles, choice_angle, angular_resolution = 10):
    
    #Getting distribution of angles present in snapshot
    hist, edges =  np.histogram(angles, angular_resolution, range = (-1,1))
    binvals = np.array([((edges[x+1]-edges[x])/2) + edges[x] for x in range(len(edges)-1)])
    
    #double-differentiate histogram to check where there are peaks in the center and/or edges
    diff = np.sign(np.diff( np.concatenate([ np.zeros([1]), hist]) ))
    diff = np.diff(  np.concatenate([diff, np.ones([1])*-1])  ) 
    diff_idxs = diff == -2 #logical idx of jumps from positive to negative
    
    #There's still a chance that there are no 'peaks' because there are plateaus in angle freq.
    #In these cases changing angular resolution should give us a tie-breaker...so just going to recursively call
    #with a better angular resolution. Eventually, due to random chance occurences of angles, 
    #a peak must emerge, even if it's garbage.
    if np.sum(diff_idxs) == 0:
        return multimodal_angles(angles, choice_angle, angular_resolution + 1)
    
    #Return the mean angle from a peak bin closest to the choice_angle
    chosenpeak = np.argmin(np.abs(binvals - choice_angle)[diff_idxs])
    chosenbin = np.nonzero(diff_idxs)[0][chosenpeak]
    angle_idxs = np.logical_and(angles >= edges[chosenbin], angles <= edges[chosenbin + 1])
    anglesubset = angles[angle_idxs]
    
    return np.mean(anglesubset)
        
```

So you can see that instead of orienting towards the center of mass of navigable territory, the rover can now orient towards one of many detected centers of navigable mass!

![It chose a peak!][fig3]

#### Processing Images

I used the color thresholding function to take different cuts of the raw image, one for navigable terrain and another for sample rocks.

I used to also take one for obstacles, but didn't find much of a use for them in the project, so ended up ditching that.

For navigable terrain, I also added my gaussian blur to smooth out the sometimes unpredictable warping, before adding a proximity threshold to keep my rover fairly nearsighted. From here I change the binary images to rover centric pixel coordinates, which I could turn into world coordinates using pix_to_world and the rover's known position in the world. I could immediately slap these coordinates on the world map to create an overlay.

The nearsightedness of my rover proved to be a valuable tool in getting map fidelity fairly high, as you can see from the movie.

I repeated the very same process used for navigable terrain for my image binaries that singled out rocks, and slapped them into a separate RGB layer.

Obstacles were ignored because they didn't seem very useful and also for some reason took a very long time to compute. This made making a movie difficult, and later in my robot caused a lot of lag between decision-making and actuating, so I chose to completely ignore 'obstacles' as a seperately calculated layer. Navigable terrain contained enough information about obstacles to completely supplant it.

### Autonomous Navigation and Mapping

#### Perception Step and Decision Step

The perception step involved copying exactly what I had in the notebook over to the Rover's code. This involved warping, putting a color threshold on the warped image to produce 'rock' and 'navigable' binaries. The navigable binary was thrown through a gaussian blur and proximity threshold as described above to create a smooth picture of what terrain was navigable.

I made the proximity threshold on the rock binary a little higher so that the rover could map  rocks even if it could not access its area. However I still kept it bounded with a threshold because the warping of the perspective transform tended to give me spurious rock mappings.


```python
    # Applying my color threshold, gaussian blur, and then a proximity threshold for each class I want to detect
    navigable =  blur_binarized_img(color_thresh(warped, (160, 160, 160), (256,256,256)), 4)
    navigable = proximity_thresh(navigable, start_tuple = (navigable.shape[0], navigable.shape[1]/2), pixdistance=60)

    rocks = color_thresh(warped, (135,115,10), (210,185,45)) #No need to blur over tiny rocks
    rocks = proximity_thresh(rocks, start_tuple = (rocks.shape[0], rocks.shape[1]/2), pixdistance=80)

```

From binaries I followed up with the now familiar rigmarole of extracting rover centric coordinates, which I could then convert to world coordinates using rotation and translation.

I also extracted angles from the navigable terrain and stored this in the rover's state, as well as the world coordinates mentioned above.

I made sure to add the world coordinates to the right layers of the Rover.worldmap state so that the supporting functions could properly count up found rocks and mapping fidelity: Navigable terrain fit in the blue layer, while rocks sat in the green layer.

Again, obstacles were excluded because it seemed their computation caused a strange lag in the Rover and didn't allow it to update behaviors in time with it's decision making. Its turns always seemed a step behind its vision.

```python
 # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = rocks*255
    # Rover.vision_image[:, :, 1] = obstacles * 255
    Rover.vision_image[:,:,2] = navigable*255

    # 5) Convert map image pixel values to rover-centric coords
    navigableXrov, navigableYrov = rover_coords(navigable)
    rocksXrov, rocksYrov = rover_coords(rocks)
    # obstaclesXrov, obstaclesYrov = rover_coords(obstacles)



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
    # Rover.worldmap[obstaclesY, obstaclesX, 0] = 155
    Rover.worldmap[rocksY, rocksX, 1] = 155
    Rover.worldmap[navigableY, navigableX, 2] = 155

```

The decision step got a lot more interesting.

First off, instead of orienting the rover towards the mean angle from the navigable terrain, I used my `multimodal_angle` function, with a parameter specifying that I'd like to turn right at all possible times, thus hugging the right wall. The function would also allow it to choose a particular turn when it reached a fork in the road, rather than plodding towards the center divider.

I made sure that during the 'stop' mode I'd turn in the OPPOSITE direction, preventing me from backtracking when I don't need to.

During testing of autonomy, I also realized that the rover could get stuck on things it couldn't see in 3D space, like an outcropping boulder. I added a new Rover state called 'struggle' which triggers any time the Rover is accelerating but not moving.

The idea of a struggle is that you take a breath, do a tiny turn, and floor it in hopes that you extricate yourself from your poor situation. It seemed to work well for most scenarios I saw while the rover chugged along.

```python
        #Oftentimes a 3-d overhead structure will halt the rover without it seeing the blockage in its 2-space map
        #In these cases it must struggle to escape.
        elif Rover.mode == 'struggle':

            if Rover.problem_yaw is None:
                Rover.problem_yaw = Rover.yaw

            if Rover.vel < 0.01:
                delta_angle = Rover.yaw - Rover.problem_yaw
                delta_angle = (delta_angle + 180) % 360 - 180
                if np.abs(delta_angle) > Rover.min_struggle_offset:
                    Rover.throttle = 1 #Thrust hard to extricate as fast as possible. Good thing chassis integrity isn't a metric!
                    Rover.problem_yaw = None #To set a new problem yaw if we get stuck again.
                else:
                    Rover.throttle = 0
                    Rover.steer = 15
            else:
                Rover.mode = 'forward'
                Rover.problem_yaw = None
                Rover.solution_yaw = None
```

#### Autonomous Launch!

Running with a patriotic resolution of 1776x1000, on Fastest graphics quality, on my only display, with an average FPS fo 45 or so, I managed to get:

* 98% Mapped
* 82% Fidelity
* 5 rocks found
* Within 480 seconds

This performance was replicable for the most part, even when I had hard crashes into an unexpected boulder.

I could probably improve this performance by fine tuning rover states to allow it to accelerate to different ground speeds given different conditions. For now I only have a very coarse state-space of 'forward', 'struggle', and 'stop'. There are many occassions on which the Rover could absolutely floor it that I don't take advantage of currently.

I also waste a good deal of time struggling against boulders. If I could figure out a way to compute an angle to orient away from imminent obstacles in a fast enough way, I could get a much smoother ride through the Martian valley.




