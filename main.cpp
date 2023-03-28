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
#include "CKalman.h"

int main(int argc, char *argv[]) {

    CKalman Kalman(4,2,0,CV_32F);
    Kalman.KalmanPreWhileLoop();
    Display *display = XOpenDisplay(NULL);
    cv::VideoCapture CapturedImage;
    cv::Mat mainImage,grayScaleImage;
    CEyeDetection eyeDetectionObject;

    if(!CapturedImage.open("/dev/video0"))
        throw std::runtime_error("Kamera nie działa");


    cv::CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load("haarcascade_eye.xml.2");
    if(eyeCascade.empty() || faceCascade.empty())
        std::cout << "Brak kaskady" << std::endl;



    while(cv::waitKey(20)!=27) {

        CapturedImage >> mainImage;
        cvtColor(mainImage, grayScaleImage, cv::COLOR_RGB2GRAY);
        resize(mainImage, eyeDetectionObject.temp,cv::Size( mainImage.cols, mainImage.rows), cv::INTER_LINEAR);

        //kom
        if(!eyeDetectionObject.boolEye) {
            try {
                eyeDetectionObject.faceCascade(faceCascade, grayScaleImage, mainImage);
            }catch (...) {
                continue;
            }                               //If face not detected, skip the rest of the loop

            eyeDetectionObject.eyeCascade(eyeCascade, mainImage, grayScaleImage);
            eyeDetectionObject.cutEyesArea(grayScaleImage);
            eyeDetectionObject.StworzenieEkranuZInformacja("Prosze patrzec w kamere laptopa do momentu uslyszenia dzwieku");

            eyeDetectionObject.findingCircles(mainImage);
            eyeDetectionObject.whereAreCircles(mainImage, display);
            eyeDetectionObject.BlinkingDetection(eyeDetectionObject.m_mEyeCutPicture[0],display);


        }
        else {
            eyeDetectionObject.whereAreCircles(mainImage, display);
            eyeDetectionObject.BlinkingDetection(eyeDetectionObject.m_mEyeCutPicture[0],display);
            eyeDetectionObject.trackRightEye(mainImage);

            Kalman.actualKalman(eyeDetectionObject, mainImage);

            if (!eyeDetectionObject.skalibrowano) {
                eyeDetectionObject.StworzenieEkranuZInformacja("Teraz nastapi kalibracja. Prosze wzrokiem sledzic kropke.");
                eyeDetectionObject.Calibration();
            }
            else {
                eyeDetectionObject.PositionOnDisplay(eyeDetectionObject.m_rEyeRect);

                eyeDetectionObject.PositionOnDisplay(Kalman.predRect);
                eyeDetectionObject.PokazanieNaEkranie(display);
            }
        }
    }
    return 0;
}
