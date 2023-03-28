//
// Created by paulinka on 3/28/23.
//

#include "CKalman.h"
void CKalman::KalmanPreWhileLoop()
{
    cv::setIdentity(m_KalmanFilter.measurementMatrix, cv::Scalar(1));
    cv::setIdentity(m_KalmanFilter.processNoiseCov, cv::Scalar(1e-6) );
    cv::setIdentity(m_KalmanFilter.measurementNoiseCov, cv::Scalar(1e-3) );
    cv::setIdentity(m_KalmanFilter.errorCovPost, cv::Scalar(1));
}
void CKalman::KalmanStatePre( CEyeDetection eyeDetectionObject)
{
    m_KalmanFilter.statePre.at<float>(0) = eyeDetectionObject.m_pEyeCenter.x + 25;
    m_KalmanFilter.statePre.at<float>(1) = eyeDetectionObject.m_pEyeCenter.y + 25;
    m_KalmanFilter.statePre.at<float>(2) = 0;
    m_KalmanFilter.statePre.at<float>(3) = 0;
}
void CKalman::actualKalman( CEyeDetection eyeDetectionObject, cv::Mat& mainImage) {
    static bool czy=false;
    if(!czy)
    {
        KalmanStatePre(eyeDetectionObject);
        czy=true;
    }
    else
        m_mState = m_KalmanFilter.predict();

    m_rPredRect.width = m_mState.at<int>(2);
    m_rPredRect.height = m_mState.at<int>(3);
    m_rPredRect.x = m_mState.at<int>(0) - m_rPredRect.width / 2;
    m_rPredRect.y = m_mState.at<int>(1) - m_rPredRect.height / 2;

    m_pCenter.x = m_mState.at<int>(0);
    m_pCenter.y = m_mState.at<int>(1);

    if(!eyeDetectionObject.m_bIfEyesFound)
    {
        auto ballsBox=cv::Rect(eyeDetectionObject.m_pEyeCenter,
                               cv::Point(eyeDetectionObject.m_pEyeCenter.x+25, eyeDetectionObject.m_pEyeCenter.y+25));
        m_mMeas.at<float>(0) = ballsBox.x + ballsBox.width / 2;
        m_mMeas.at<float>(1) = ballsBox.y + ballsBox.height / 2;
        m_mMeas.at<float>(2) = (float)ballsBox.width;
        m_mMeas.at<float>(3) = (float)ballsBox.height;
        if (!found)
        {
            m_KalmanFilter.errorCovPre.at<float>(0) = 1;
            m_KalmanFilter.errorCovPre.at<float>(7) = 1;
            m_KalmanFilter.errorCovPre.at<float>(14) = 1;
            m_KalmanFilter.errorCovPre.at<float>(21) = 1;

            m_mState.at<float>(0) = m_mMeas.at<float>(0);
            m_mState.at<float>(1) = m_mMeas.at<float>(1);
            m_mState.at<float>(2) = m_mMeas.at<float>(2);
            m_mState.at<float>(3) = m_mMeas.at<float>(3);

            m_KalmanFilter.statePost = m_mState;

            found = true;
        }else
            m_KalmanFilter.correct(m_mMeas);
    }else
    {
        m_mMeas.at<float>(0) = eyeDetectionObject.m_rEyeRect.x + eyeDetectionObject.m_rEyeRect.width / 2;
        m_mMeas.at<float>(1) = eyeDetectionObject.m_rEyeRect.y + eyeDetectionObject.m_rEyeRect.height / 2;
        m_mMeas.at<float>(2) = (float)eyeDetectionObject.m_rEyeRect.width;
        m_mMeas.at<float>(3) = (float)eyeDetectionObject.m_rEyeRect.height;
        if (!found) {
            m_KalmanFilter.errorCovPre.at<float>(0) = 1;
            m_KalmanFilter.errorCovPre.at<float>(7) = 1;
            m_KalmanFilter.errorCovPre.at<float>(14) = 1;
            m_KalmanFilter.errorCovPre.at<float>(21) = 1;

            m_mState.at<float>(0) = m_mMeas.at<float>(0);
            m_mState.at<float>(1) = m_mMeas.at<float>(1);
            m_mState.at<float>(2) = m_mMeas.at<float>(2);
            m_mState.at<float>(3) = m_mMeas.at<float>(3);

            m_KalmanFilter.statePost = m_mState;

            found = true;
        }else
            cv::Mat estimated = m_KalmanFilter.correct(m_mMeas);

    }
}
