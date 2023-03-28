
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
#include <X11/Xlib.h>
#include <thread>
#include <fstream>

class CEyeDetection{
public:
    std::vector <cv::Rect>  m_rDetectedFaces;
    std::vector <cv::Rect>  m_rDetectedEyes;

    void        faceCascade(cv::CascadeClassifier Cascade, cv::Mat grayScaleImage);    //Find and save face position
    void        eyeCascade(cv::CascadeClassifier eyeCascade, cv::Mat grayScaleImage);  //Find and save right eye position
    void        cutEyesArea( cv::Mat grayScaleImage);                                  //Cut the area of found eye
    cv::Point   segmentation(cv::Mat wejsciowe);                                       //
    void        findingCircles(cv::Mat& mainImage);                                    //Finding circles on picture of the eye that potentially can be iris

    std::vector <cv::Mat>   m_mEyeCutPicture;

    //Do sprawdzenia gdzie jest oko?
    cv::Rect    m_rEyeRect;
    cv::Point   m_pEyeCenter;
    cv::Mat     m_EyeMat;
    cv::Rect    m_rEyeRectPrev;
    int         m_iEyeRadius;
    cv::Point   m_pEyeCenterPrev;
    bool        m_bIfBlinked;

    std::pair<double, double> FitLine(cv::Mat Image);
    void BlinkingDetection(cv::Mat grayScaleImage,Display* display);
    void trackRightEye( cv::Mat& mainFrame);

    void PositionOnDisplay(cv::Rect rect);
    bool skalibrowano;
    std::vector<cv::Point> punkty;
    cv::Point EkranPx=cv::Point(1920,1080);
    cv::Rect MaxEkran;

    int roznicaX, roznicaY, srodekWyliczonyX,srodekWyliczonyY;
    cv::Point2d okoNaEkraniePX;

    std::vector<int> ZebranePunktyX, ZebranePunktyY;
    int h = 50;
    int j=50;
    void StworzenieEkranuZInformacja(std::string sInfo) ;
    void Calibration();

    void PokazanieNaEkranie(Display *display);
};


#endif //TESTOPENCV_CEYEDETECTION_H
