cmake_minimum_required(VERSION 3.1.0)
project(TestOpenCV)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_INCLUDE_CURRENT_DIR ON)
# set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)
# /usr/local/include/dlib/
set(dlib_ROOT "/home/paulinka/Documents/dlib-19.24")

#find dlib
# find_package(dlib REQUIRED)
# Find the QtWidgets library
# find_package(Qt5 COMPONENTS Widgets REQUIRED)
# find_package(Qt5Widgets)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Find the QtWidgets library
find_package(Qt5Widgets)


add_executable(TestOpenCV main.cpp CEyeDetection.cpp CEyeDetection.h CKalman.cpp CKalman.h)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED )
target_link_libraries(TestOpenCV Qt5::Widgets ${OpenCV_LIBS}  Threads::Threads ${X11_LIBRARIES} )#dlib::dlib )#Qt5::Widgets Qt5::Qt5Core)
SET(GCC_COVERAGE_LINK_FLAGS    "-lpthread")