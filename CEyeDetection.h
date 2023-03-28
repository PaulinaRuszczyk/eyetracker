
#ifndef TESTOPENCV_CEYEDETECTION_H
#define TESTOPENCV_CEYEDETECTION_H

#include <QApplication>
#include <QPushButton>

#include <utility>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <thread>
#include <X11/Xlib.h>

class CEyeDetection{
public:
    void        FaceCascade(cv::CascadeClassifier Cascade, cv::Mat grayScaleImage);    //Find and save face position
    void        EyeCascade(cv::CascadeClassifier eyeCascade, cv::Mat grayScaleImage);  //Find and save right eye position
    void        CutEyesArea(cv::Mat grayScaleImage);                                   //Cut the area of found eye
    cv::Point   Segmentation(cv::Mat inputImage);
    void        FindingCircles(cv::Mat& mainImage);                                    //Finding circles on picture of the eye that potentially can be iris

    void        BlinkingDetection(cv::Mat grayScaleImage, Display* display);
    void        TrackFoundEye(cv::Mat& mainFrame);

    void        PositionOnDisplay(cv::Rect rect);

    void        InfoBoard(std::string sInfo) ;
    void        Calibration();

    void        ShowOnDisplay(Display *display);

    std::vector <cv::Mat>   m_mEyeCutPicture;

    cv::Rect    m_rEyeRect;
    cv::Point   m_pEyeCenter;
    int         m_iEyeRadius;

    bool        m_bIfCalibrated;
    bool        m_bIfEyesFound;

    cv::Point   m_pDelta;
    cv::Point   m_pIrisCenterPoint;

private:
    std::vector <cv::Rect>  m_rDetectedFaces;
    std::vector <cv::Rect>  m_rDetectedEyes;
    cv::Point2d             m_pEyePositionOnDisplay;
    cv::Point               m_pDisplaySize=cv::Point(1920, 1080);
    cv::Mat                 m_mEyeMat;
    bool                    m_bIfBlinked;
    cv::Rect                m_rEyeRectPrev;
    cv::Point               m_pEyeCenterPrev;

    std::pair<double, double> FitLine(cv::Mat Image);

};


#endif //TESTOPENCV_CEYEDETECTION_H
