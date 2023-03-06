
import math

from glider import *
import random
from datetime import datetime


def initialize_particles(N, mapFunc, height):
    p = []
    for i in range(N):
        x = random.uniform(-250, 250)
        y = random.uniform(-250, 250)
        heading = random.gauss(0, math.pi/4)
        g = glider(x, y, height, heading, mapFunc)
        g.set_noise(10, 0.05, 5) # values based on Particle Filter Lesson Notes code, measurement is based on sigma used in measurement_prob
        p.append(g)
    return p


def gaussian(mu, sigma, x):
    return math.exp(-((mu - x) ** 2) / (sigma ** 2) / 2.0) / math.sqrt(2.0 * math.pi * (sigma ** 2))

def measurement_prob(radar, particle):
    prob = 1.0
    particle_height_above_ground = particle.sense()
    sigma = 10.0 # based on guessing and checking
    prob *= gaussian(radar, sigma, particle_height_above_ground)
    return prob


def weigh_particles(particles, radar):
    w = []
    max_weight = 0
    max_index = 0
    for i in range(len(particles)):
        weight = measurement_prob(radar, particles[i])
        if weight >= max_weight:
            max_weight = weight
            max_index = i
        w.append(weight)
    return w, max_index


def estimate_next_pos(height, radar, mapFunc, OTHER=None):
    """Estimate the next (x,y) position of the glider."""
    if OTHER is None:
        OTHER = {}
        OTHER['particles'] = initialize_particles(30000, mapFunc, height)
        OTHER['N'] = 30000 #initial number of particles
        OTHER["steps"] = 1
    else:
        OTHER['N'] = 1400
        OTHER["steps"] += 1

    N = OTHER['N']
    particles = OTHER['particles']

    weights = []
    max_weight = 0
    max_index = 0
    for i in range(N):
        weight = measurement_prob(radar, particles[i])
        if weight >= max_weight:
            max_weight = weight
            max_index = i
        weights.append(weight)

    # resample particles

    tmp = []
    index = int(random.random() * N)
    beta = 0.0
    mw = weights[max_index]
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        tmp.append(particles[index])
    particles = tmp

    # If GLIDING then next_angle is being tested and gliding is split up among functions.
    # estimate_next_pos will not move particles one timestep ahead if next_angle is.
    if "GLIDING" not in OTHER:
        p = []
        for g in particles:
            p.append(g.glide())
        particles = p

    # calculate X,Y based weighted average of all particles
    xw_sum = 0.0
    yw_sum = 0.0
    w_sum = 0.0
    for i in range(N):
        x = particles[i].x
        y = particles[i].y
        w = weights[i]
        xw = x*w
        xw_sum += xw
        yw = y * w
        yw_sum += yw
        w_sum += w
    y = yw_sum / w_sum
    x = xw_sum / w_sum

    estimate = (x,y)

    points_to_plot = []

    for p in particles:
        points_to_plot.append([p.x, p.y, p.heading])

    # reduce the number of particles drawn in viz otherwise lag
    particlesToDraw = []
    if N == 30000:
        particlesToDraw = [p for i, p in enumerate(points_to_plot) if i % 20 == 0] # this line of code was borrowed from a Piazza post
    else:
        particlesToDraw = points_to_plot

    OTHER['particles'] = particles

    return estimate, OTHER, particlesToDraw



def next_angle(height, radar, mapFunc, OTHER=None):
    estimate, OTHER, points_to_plot = estimate_next_pos(height, radar, mapFunc, OTHER)
    if OTHER["steps"] < 20: 
        return 0.0, OTHER, points_to_plot
    else:
        OTHER['GLIDING'] = True # this value is used to make sure double gliding isn't occuring with the particles

    particles = OTHER["particles"]

    x_sum = 0.0
    y_sum = 0.0
    h_sum = 0.0
    for p in particles:
        x_sum += p.x
        y_sum += p.y
        h_sum += p.heading
    x = x_sum / len(particles)
    y = y_sum / len(particles)
    average_heading = h_sum / len(particles)

    desired_heading = math.atan2(0 - estimate[1], 0 - estimate[0])
    bearing = desired_heading - average_heading
    bearing = angle_trunc(bearing) # keep bearing between -pi and pi
    ###

    b = []
    for a in particles:
        b.append(a.glide(bearing))
    particles = b
    OTHER['particles'] = particles

    return bearing, OTHER, points_to_plot
