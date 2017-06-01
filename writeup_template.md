## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[fig1]: ./misc/fig1.png
[fig2]: ./misc/fig2.png
[fig3]: ./misc/fig3.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

### Notebook Analysis

## Running the Notebook, and making modifications

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

## Processing Images

I used the color thresholding function to take different cuts of the raw image, one for navigable terrain and another for sample rocks.

I used to also take one for obstacles, but didn't find much of a use for them in the project, so ended up ditching that.

For navigable terrain, I also added my gaussian blur to smooth out the sometimes unpredictable warping, before adding a proximity threshold to keep my rover fairly nearsighted. From here I change the binary images to rover centric pixel coordinates, which I could turn into world coordinates using pix_to_world and the rover's known position in the world. I could immediately slap these coordinates on the world map to create an overlay.

The nearsightedness of my rover proved to be a valuable tool in getting map fidelity fairly high, as you can see from the movie.

I repeated the very same process used for navigable terrain for my image binaries that singled out rocks, and slapped them into a separate RGB layer.

Obstacles were ignored because they didn't seem very useful and also for some reason took a very long time to compute. This made making a movie difficult, and later in my robot caused a lot of lag between decision-making and actuating, so I chose to completely ignore 'obstacles' as a seperately calculated layer. Navigable terrain contained enough information about obstacles to completely supplant it.


### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
Here is an example of how to include an image in your writeup.

![alt text][image1]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
And another! 

![alt text][image2]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]


