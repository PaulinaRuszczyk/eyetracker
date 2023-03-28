//
// Created by paulinka on 3/28/23.
//

#ifndef TESTOPENCV_CKALMAN_H
#define TESTOPENCV_CKALMAN_H

#include <QApplication>
#include <QPushButton>

#include <utility>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <X11/Xlib.h>
#include <thread>
#include <fstream>

#include "CEyeDetection.h"

class CKalman{
public:
    cv::Mat measurement = cv::Mat::zeros(1, 1, CV_32F);
//oko prawe
    cv::Mat_<float>  rightEyePosition;
    cv::KalmanFilter rightEyeFilter;
//oko Lewe
    cv::Mat leftEyePosition;
    cv::KalmanFilter leftEyeFilter;


    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat meas;
    CKalman(int stateSize,int measSize, int contrSize, unsigned int type)
            :kf(stateSize, measSize, contrSize, type),
             state(stateSize, 1, type),
             meas(measSize, 1, type)
    {
    }

    void KalmanPreWhileLoop();
    void KalmanStatePre( CEyeDetection eyeDetectionObject);
    void actualKalman( CEyeDetection eyeDetectionObject, cv::Mat& mainImage) ;

    bool found = false;
    cv::Point center;
    cv::Rect predRect;

};


#endif //TESTOPENCV_CKALMAN_H
