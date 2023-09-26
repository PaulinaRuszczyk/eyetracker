# Description
Software enables everyday use of a computer by mouse moving based on eye movement.
# Requirements
* g++
* CMake Minimum version 3.13
* OpenCV
* OpenCV contrib

# Compile
* Create build directory
```bash
mkdir build
cd build
```
* Build
```bash
 cmake ..
 make 
```
* Run build app
```bash
./TestOpenCV
```
# Usage
During initial configuration, software locates eyes and its position in
different places of focus on the screen based on a video captured by a webcam. 

After this stage is completed cursor begins to move to the places where users eyes
are focused. By blinking user click the right mouse button and by closing eyes for
approximately 5 seconds users exit the program.

The best performance of the software was observed for dark eyes color and for people without glasses.

