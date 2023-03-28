//
// Created by paulinka on 3/28/23.
//

#include "CKalman.h"
void CKalman::KalmanPreWhileLoop()
    {
        cv::setIdentity( kf.measurementMatrix,
                         cv::Scalar(1)
        );
        cv::setIdentity( kf.processNoiseCov,
                         cv::Scalar(1e-6) );
        cv::setIdentity( kf.measurementNoiseCov, cv::Scalar(1e-3) );
        cv::setIdentity( kf.errorCovPost,
                         cv::Scalar(1)
        );

    }
    void CKalman::KalmanStatePre( CEyeDetection eyeDetectionObject)
    {
        kf.statePre.at<float>(0) = eyeDetectionObject.m_pEyeCenter.x+25;
        kf.statePre.at<float>(1) = eyeDetectionObject.m_pEyeCenter.y+25;
        kf.statePre.at<float>(2) = 0;
        kf.statePre.at<float>(3) = 0;
    }
    void CKalman::actualKalman( CEyeDetection eyeDetectionObject, cv::Mat& mainImage) {
        static bool czy=false;
        if(!czy)
        {
            KalmanStatePre(eyeDetectionObject);
            czy=true;
        }
        else
            state = kf.predict();

        predRect.width = state.at<float>(2);
        predRect.height = state.at<float>(3);
        predRect.x = state.at<float>(0) - predRect.width / 2;
        predRect.y = state.at<float>(1) - predRect.height / 2;

        center.x = state.at<float>(0);
        center.y = state.at<float>(1);

        if(!eyeDetectionObject.boolEye)
        {
            cv::Rect ballsBox=cv::Rect(eyeDetectionObject.m_pEyeCenter,
                                       cv::Point(eyeDetectionObject.m_pEyeCenter.x+25, eyeDetectionObject.m_pEyeCenter.y+25));
            meas.at<float>(0) = ballsBox.x + ballsBox.width / 2;
            meas.at<float>(1) = ballsBox.y + ballsBox.height / 2;
            meas.at<float>(2) = (float)ballsBox.width;
            meas.at<float>(3) = (float)ballsBox.height;
            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = meas.at<float>(2);
                state.at<float>(3) = meas.at<float>(3);

                kf.statePost = state;

                found = true;
            }
            else
                kf.correct(meas);
        }
        else
        {
            meas.at<float>(0) = eyeDetectionObject.m_rEyeRect.x + eyeDetectionObject.m_rEyeRect.width / 2;
            meas.at<float>(1) = eyeDetectionObject.m_rEyeRect.y + eyeDetectionObject.m_rEyeRect.height / 2;
            meas.at<float>(2) = (float)eyeDetectionObject.m_rEyeRect.width;
            meas.at<float>(3) = (float)eyeDetectionObject.m_rEyeRect.height;
            if (!found)
            {
                kf.errorCovPre.at<float>(0) = 1;
                kf.errorCovPre.at<float>(7) = 1;
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = meas.at<float>(2);
                state.at<float>(3) = meas.at<float>(3);

                kf.statePost = state;

                found = true;
            }
            else{
                cv::Mat estimated = kf.correct(meas);
            }
        }
    }
