# Binary-Image-Segmentation
Binary Image Segmentation using graph mincut and OpenCV

Binary Image segmentation is the process of classifying the pixels of an image into two categories: pixels belonging to the foreground objects of an image and pixels belonging to the background objects of an image. Image segmentation is an important problem in image processing and computer vision with many application ranging from background substraction and removal to object tracking, etc.

The Min Graph-cut problem 
Given a connected graph G(V, E), and two vertices s (source vertex) and t (sink vertex), a cut is a subset of edges E’ that disconnects any path from s to t. A minimum cut E’’ is a cut where the sum of the weights of all its edges is not larger than any other cut E’. The problem of minimum cut can be generalized to the case where more than one source or sink exist. It is easy to observe that any cut of G classifies the vertices in V into two disjoint sets: vertices connected to s and vertices connected to t.  

The min-cut problem and max-flow are dual to each other, so for max flow Dinic Algorithm is used which is faster than ford-fulkerson algorithm.

Specifications 
The program has 3 arguments: an input image, a configuration file that provides the initial set of foreground and background points and an output image. 

