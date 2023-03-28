//
// Created by paulinka on 3/28/23.
//

#ifndef TESTOPENCV_CKALMAN_H
#define TESTOPENCV_CKALMAN_H


#include "CEyeDetection.h"

class CKalman{
public:
    CKalman(int stateSize,int measSize, int contrSize, unsigned int type)
            : m_KalmanFilter(stateSize, measSize, contrSize, type),
              m_mState(stateSize, 1, type),
              m_mMeas(measSize, 1, type),
              m_mMeasurement(cv::Mat::zeros(1, 1, CV_32F))
    {
    }

    void KalmanPreWhileLoop();
    void actualKalman( CEyeDetection eyeDetectionObject, cv::Mat& mainImage) ;

    cv::Rect m_rPredRect;

private:
    void KalmanStatePre( CEyeDetection eyeDetectionObject);

    bool found = false;
    cv::Point m_pCenter;

    cv::KalmanFilter m_KalmanFilter;
    cv::Mat m_mState;
    cv::Mat m_mMeas;
    cv::Mat m_mMeasurement;

};


#endif //TESTOPENCV_CKALMAN_H
