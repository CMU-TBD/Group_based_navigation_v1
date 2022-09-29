cked file consists of a series of splines that describe the moving behavior of a person in a video.
Comments in the file start with a '-' and end at the end of the line.

The number of splines can be found in the first line of the file.
Then immediately after that, each spline is defined in the following format:

   Number_of_control_points_N
   x y frame_number gaze_direction   \
   x y frame_number gaze_direction    \
   ....                                >>> N control points
   x y frame_number gaze_direction    /
   x y frame_number gaze_direction   /

   Number_of_control_points_K
   x y frame_number gaze_direction   \
   x y frame_number gaze_direction    \
   ....                                >>> K control points
   x y frame_number gaze_direction    /
   x y frame_number gaze_direction   /
   
   ....
   ...
   
x, y: the position of the person in pixel space, where (0, 0) is the center of the frame.
frame_number: the time (frames)at which the position was tracked
gaze_direction: the viewing direction of the person in degrees (0 degrees means the person is looking upwards)

