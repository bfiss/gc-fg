First:
$ make

To use GraphCut, try one of these examples:

1. Using one of the MRF .txt in data/
 $ ./solveMRF data/person.txt
 
 The output is in labelMRF.ppm

2. Using the presentation app:
 $ ./presentation

 To use this app, click with the left mouse button
 to select some spots that should be tracked, and
 then with the right mouse button to perform Graph
 Cut. Change the penalty value in the trackbar to
 adjust the neighborhood labeling difference penalty.
 If a video is being used, press SPACE to pause and
 play. To exit the application, use ESC.

3. Using Rihanna skin detection example:
 $ ./app
 
 The resulting image is in label.ppm

 4. Using the debug.cpp source (the data term can be changed manually):
 ./dbg
 
 Results are printed to the terminal.
