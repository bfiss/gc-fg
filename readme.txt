First:
$ make

To use GraphCut, try one of these examples:

1. Comparing with NPP:
 $ cd nppApp
 $ ./imageSegmentationNPP
# this ran an example application from the NPP library on the data ../../data/person.txt
# we now want to compare our implementation with NPP's results
 $ cd ..
 $ ./solveMRF ../data/person.txt
 $ eog ../data/person_segmentation.pgm &
# this opened NPP's result
 $ eog labelMRF.ppm &
# this opened our result
# to test with another data set, change the name of the current person.txt file and rename one of the other files to person.txt

2. Using one of the MRF .txt in data/
 $ ./solveMRF ../data/flower.txt
 
 The output is in labelMRF.ppm

3. Using Rihanna skin detection example:
 $ ./app
 
 The resulting image is in label.ppm

 4. Using the debug.cpp source (the data term can be changed manually):
 ./dbg
 
 Results are printed to the terminal.

 