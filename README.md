# Predict-Vehicle-Angle

Peking University/Baidu - Autonomous Driving

# Abstract

This dataset contains photos of streets, taken from the roof of a car. We develop an algorithm to estimate the position and orientation of vehicles.

*Note that rotation values are angles expressed in radians, relative to the camera.*

The primary data is images of cars and related pose information. The pose information is formatted as strings, as follows:

model type, yaw, pitch, roll, x, y, z


Let us first try to understand what the 6 degrees of freedom mean.

To completely specify the position of a 3-D object, we need to specify how it is rotated with respect to X/Y/Z axis, in addition to the the position a reference point (say center of the object) in the object.
As illustrated in the figure below, roll/pitch/yaw correspond to rotation of an object around the X/Y/Z axis respectively.
