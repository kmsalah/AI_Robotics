# Particle Filter Localization - Glider

This is a practice implementation of a particle filter used to localize a robotic glider that does not have access to
GPS sattelites. The glider is released from space over the surface of Mars and receives a distance to ground measurements
from a downwards facing radar and a altitude estimate from a barometric pressure sensor.

The first part of this project determiens the next location of the glider given its atmospheric height and the radar distance to the ground.

The second part of the project is to set the turn angle of the glider so that the glider returns to the center of the map.