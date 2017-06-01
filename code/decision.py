import numpy as np

# I don't know if mean angle is the greatest policy for choosing where to go next.
# This function enacts another policy: given a distribution of angles from the binarized image,
# it will choose the mean angle from the subset closest to a 'choice angle' (in degrees).
#Default of 1 tells the rover to always turn left. Theoretically in our little starfish map,
#constant left turns should get complete coverage with zero backtracking unless traveling up and down a spoke.
def multimodal_angles(angles, choice_angle = 30, angular_resolution=10):
    # Getting distribution of angles present in snapshot
    hist, edges = np.histogram(angles, angular_resolution, range=(-180/np.pi, 180/np.pi))
    binvals = np.array([((edges[x + 1] - edges[x]) / 2) + edges[x] for x in range(len(edges) - 1)])

    # double-differentiate histogram to check where there are peaks in the center and/or edges
    diff = np.sign(np.diff(np.concatenate([np.zeros([1]), hist])))
    diff = np.diff(np.concatenate([diff, np.ones([1]) * -1]))
    diff_idxs = diff == -2  # logical idx of jumps from positive to negative

    # There's still a chance that there are no 'peaks' because there are plateaus in angle freq.
    # In these cases changing angular resolution should give us a tie-breaker...so just going to recursively call
    # with a better angular resolution. Eventually, due to random chance occurences of angles,
    # a peak must emerge, even if it's garbage.
    if np.sum(diff_idxs) == 0:
        return multimodal_angles(angles, choice_angle, angular_resolution + 1)

    # Return the mean angle from a peak bin closest to the choice_angle
    chosenpeak = np.argmin(np.abs(binvals - choice_angle)[diff_idxs])
    chosenbin = np.nonzero(diff_idxs)[0][chosenpeak]
    angle_idxs = np.logical_and(angles >= edges[chosenbin], angles <= edges[chosenbin + 1])
    anglesubset = angles[angle_idxs]

    print(np.mean(anglesubset))

    return np.mean(anglesubset)


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    print(Rover.mode)

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0

                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(multimodal_angles(Rover.nav_angles * 180/np.pi, -180/np.pi, 9), -15, 15)

                if Rover.vel < 0.01 and Rover.throttle == Rover.throttle_set:
                    Rover.mode = 'struggle'


            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        #Oftentimes a 3-d overhead structure will halt the rover without it seeing the blockage in its 2-space map
        #In these cases it must struggle to escape.
        elif Rover.mode == 'struggle':

            if Rover.problem_yaw is None:
                Rover.problem_yaw = Rover.yaw

            if Rover.vel < 0.01:
                delta_angle = Rover.yaw - Rover.problem_yaw
                delta_angle = (delta_angle + 180) % 360 - 180
                if np.abs(delta_angle) > Rover.min_struggle_offset:
                    Rover.throttle = 1 #Thrust hard to extricate as fast as possible
                    Rover.problem_yaw = None #To set a new problem yaw if we get stuck again.
                else:
                    Rover.throttle = 0
                    Rover.steer = 15
            else:
                Rover.mode = 'forward'
                Rover.problem_yaw = None
                Rover.solution_yaw = None


        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        # If we're in stop mode but still moving keep braking
        if Rover.vel > 0.2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        # If we're not moving (vel < 0.2) then do something else
        elif Rover.vel <= 0.2:
            # Now we're stopped and we have vision data to see if there's a path forward
            if len(Rover.nav_angles) < Rover.go_forward:
                Rover.throttle = 0
                # Release the brake to allow turning
                Rover.brake = 0
                # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                Rover.steer = 15  # Could be more clever here about which way to turn
            # If we're stopped but see sufficient navigable terrain in front then go!
            if len(Rover.nav_angles) >= Rover.go_forward:
                # Set throttle back to stored value
                Rover.throttle = Rover.throttle_set
                # Release the brake
                Rover.brake = 0
                # Set steer to mean angle
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                Rover.mode = 'forward'

    return Rover

