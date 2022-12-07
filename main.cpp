
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
    std::vector <cv::Rect>  m_detectedFaces;

    std::vector <cv::Rect>  m_RightEye;
    std::vector <cv::Mat>   m_rightEyeCutPicture;
    int                     m_rightEyeRadius;

    //Do sprawdzenia gdzie jest oko?
    cv::Rect    m_rightEyeRectPrev= cv::Rect (0,0,0,0);
    cv::Rect    m_rightEyeRect;
    cv::Point   m_rightEyeCenterPrev= cv::Point (0,0);
    cv::Point   m_RightEyeCenter;//= cv::Point (0,0);
    cv::Mat     m_rightEyeMat;
    //Zająć sie i uporządkować liczniki
    int licznik=0;
    int licznikUkładuOczu = 0 ;
    bool UkladOczu = false;         //zmienna czy określono położenie oczu
    bool ifBlinked = false;
    bool mrugniecie; 
    bool boolEye; //do wywalenia
    //Gdzie konstruktor XD

    //Funkcja znajdująca i zapisująca położenie twarzy
    void faceCascade(cv::CascadeClassifier Cascade, cv::Mat grayScaleImage, cv::Mat mainImage)
    {
        if(!UkladOczu) {
            Cascade.detectMultiScale(grayScaleImage, m_detectedFaces, 1.1, 15,
                                     0 | cv::CASCADE_SCALE_IMAGE,cv::Size(0, 0));
            if (!m_detectedFaces.empty()) {
                rectangle(mainImage, cv::Point(m_detectedFaces[0].x, m_detectedFaces[0].y),
                          cv::Point(m_detectedFaces[0].x + m_detectedFaces[0].width - 1,
                                    m_detectedFaces[0].y + m_detectedFaces[0].height - 1),
                          cv::Scalar(255, 0, 0), 3, 8, 0);
            }
        }
        //To chyba można usunąć ? nie wime po co to ale chyba miało być że rysuje twarz w tym sasmym miejscu jak już skonczy szukać to bedzie i tak bez tego
        //Ale ogólnie to wytłumaczyć dlaczego tu są te 0, bo to najbardziej prawdopodobne  miejsca wystąpienia twarzy
      
      if(m_detectedFaces.size()!=0)
      {
         rectangle(mainImage, cv::Point(m_detectedFaces[0].x, m_detectedFaces[0].y),
                  cv::Point(m_detectedFaces[0].x + m_detectedFaces[0].width - 1,
                            m_detectedFaces[0].y + m_detectedFaces[0].height - 1),
                  cv::Scalar(255, 0, 0), 3, 8, 0);
      } 
        //do wywalenia oba rectangle
    }
    //Funkcja znajdująca i zapisująca położenie oczu
    void eyeCascade(cv::CascadeClassifier eyeCascade, cv::Mat& mainImage, cv::Mat szare_zdjecie)
    {
        if(!UkladOczu){
        cv::Mat FaceRight, FaceLeft;                            //Zmienne zapisujące prawą  lewą stronę twarzy po to, by oczy nie były szukane poza obszarem twarzy
        m_detectedFaces[0].width=m_detectedFaces[0].width/2;
        m_detectedFaces[0].height=m_detectedFaces[0].height/2;
        FaceRight = szare_zdjecie(m_detectedFaces[0]); // zapisanie prostokąta prawej częsci twarzy
       // FaceRight = szare_zdjecie(cv::Rect(m_detectedFaces[0].width/2,m_detectedFaces[0].height/2)); // zapisanie prostokąta prawej częsci twarzy (?)
        //Może ładniej będzie i bez powrotu na końcu

        m_detectedFaces[0].x=m_detectedFaces[0].x+m_detectedFaces[0].width;
        FaceLeft=szare_zdjecie(m_detectedFaces[0]);   //zapisanie prostokąta lewej części twarzy

        m_detectedFaces[0].x-=m_detectedFaces[0].width;
        m_detectedFaces[0].width=m_detectedFaces[0].width*2;
        m_detectedFaces[0].height=m_detectedFaces[0].height*2;  // powrót do danych całej twarzy


        //Znalezienie i zaznaczenie obszaru prawego oka

        eyeCascade.detectMultiScale(FaceRight, m_RightEye, 1.1,20,0|cv::CASCADE_SCALE_IMAGE, cv::Size(0,0));

        for(size_t i=0; i<m_RightEye.size();i++)
        {
            m_RightEye[i].x+=m_detectedFaces[0].x;
            m_RightEye[i].y+=m_detectedFaces[0].y;
           /* for(size_t j=0; j<m_detectedFaces.size(); j++)
            {
                rectangle(mainImage,cv::Point(m_RightEye[i].x,m_RightEye[i].y),cv::Point(m_RightEye[i].x+m_RightEye[i].width-1,m_RightEye[i].y+m_RightEye[i].height-1), cv::Scalar(255,255,0),3,8,0); //rysowanie prostokąta
            }*/
        }
        // nie robie lewego oka na razie
         /*
        //Znalezienie i zaznaczenie obszaru lewego oka

        eyeCascade.detectMultiScale(FaceLeft, m_LeftEye, 1.1,20,0|cv::CASCADE_SCALE_IMAGE, cv::Size(0,0));
        for(size_t i=0; i<m_RightEye.size();i++)
        {
            m_LeftEye[i].x+=m_detectedFaces[0].x+m_detectedFaces[0].width/2;
            m_LeftEye[i].y+=m_detectedFaces[0].y;
            for(size_t j=0; j<m_detectedFaces.size(); j++)
            {
                rectangle(mainImage,cv::Point(m_LeftEye[i].x,m_LeftEye[i].y),cv::Point(m_LeftEye[i].x+m_LeftEye[i].width-1,m_LeftEye[i].y+m_LeftEye[i].height-1), cv::Scalar(255,255,0),3,8,0); //rysowanie prostokąta
            }
        }*/
         //Jeśli po 10 iteracji położenie oka sie nie zmienia to  zapisuje te położenie i zakłada, że tam te oko jest
        if(m_rightEyeRectPrev.x-m_RightEye[0].x<3 && m_rightEyeRectPrev.x-m_RightEye[0].x>-3 && m_rightEyeRectPrev.y-m_RightEye[0].y<3 && m_rightEyeRectPrev.y-m_RightEye[0].y>-3 )
        {
            licznikUkładuOczu++;
            if(licznikUkładuOczu==10)
                UkladOczu=true;
        }
     //   std::cout<<m_rightEyeRectPrev<<" "<<m_RightEye[0] <<" "<< licznikUkładuOczu<<"\n";
        m_rightEyeRectPrev=m_RightEye[0];
     //   m_leftEyeRectPrev=m_LeftEye[0];
        }
        //rectangle(mainImage,cv::Point(m_LeftEye[0].x,m_LeftEye[0].y),cv::Point(m_LeftEye[0].x+m_LeftEye[0].width-1,m_LeftEye[0].y+m_LeftEye[0].height-1), cv::Scalar(255,255,0),3,8,0);
        //rectangle(mainImage,cv::Point(m_RightEye[0].x,m_RightEye[0].y),cv::Point(m_RightEye[0].x+m_RightEye[0].width-1,m_RightEye[0].y+m_RightEye[0].height-1), cv::Scalar(255,255,0),3,8,0);
   }

    //Funkcja wycinająca obszar znalezionych oczu
   std::pair<double, double> FitLine(cv::Mat Image){
    cv::Mat src=Image;
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_64F);
    //cv::cvtColor(src, src, cv::COLOR_RGBA2GRAY, 0);
    cv::threshold(src, src, 177, 200, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point> >  contours ;//= new cv.MatVector();
    cv::Mat hierarchy;
    cv::Vec4f line;
    cv::findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> cnt = contours[0];
    // You can try more different parameters
    cv::fitLine(cnt, line, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Scalar contoursColor =  cv::Scalar(255, 255, 255);
    cv::Scalar lineColor =  cv::Scalar(255, 0, 0);
    cv::drawContours(dst, contours, 0, contoursColor, 1, 8, hierarchy, 100);
    double vx = line[0];
    double vy = line[1];
    double x = line[2];
    double y = line[3];
    double lefty = std::round((-x * vy / vx) + y);
    double righty = std::round(((src.cols - x) * vy / vx) + y);
    cv::Point point1 =  cv::Point(src.cols - 1, righty);
    cv::Point point2 =  cv::Point(0, lefty);
    cv::line(dst, point1, point2, lineColor, 2, cv::LINE_AA, 0);
    imshow("canvasOutput", dst);
  //  imshow("wyciete", m_rightEyeCutPicture[0]);
    return std::make_pair(point1.y, point2.y);
   }
   
    void cutEyesArea( cv::Mat grayScaleImage) {
        //std::vector<cv::Mat> CutGray;
        cv::Mat eyesKernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        cv::Rect tempEye;
        bool IsThereEyes;
        /*
        if (!m_LeftEye.empty() && !IsThereEyes && !m_RightEye.empty())
            IsThereEyes = true;*/
        if (!IsThereEyes && !m_RightEye.empty())
            IsThereEyes = true;
        if (IsThereEyes) {
            m_rightEyeCutPicture.resize(2);
            tempEye.y = m_RightEye[0].y+m_RightEye[0].height / 4;
            tempEye.height = m_RightEye[0].height / 2;
            tempEye.x=m_RightEye[0].x;
            tempEye.width=m_RightEye[0].width;
            m_rightEyeCutPicture[0] = grayScaleImage(tempEye);

           /* tempEye.y =m_LeftEye[0].y+ m_LeftEye[0].height / 4;
            tempEye.height = m_LeftEye[0].height / 2;
            tempEye.x =m_LeftEye[0].x;
            tempEye.width =m_LeftEye[0].width;
            m_rightEyeCutPicture[1] = grayScaleImage(tempEye);*/

            // Robienie zmian intensywności koloru
            //filter2D(m_rightEyeCutPicture[0], m_rightEyeCutPicture[0], -1, eyesKernel);
            //filter2D(m_rightEyeCutPicture[1], m_rightEyeCutPicture[1], -1, eyesKernel);
            
            threshold(m_rightEyeCutPicture[0], m_rightEyeCutPicture[0], 0, 255, cv::THRESH_OTSU | cv:: THRESH_BINARY_INV);
            //threshold(m_rightEyeCutPicture[1], m_rightEyeCutPicture[1], 0, 255,  cv::THRESH_OTSU  | cv:: THRESH_BINARY_INV);

            imshow("1", m_rightEyeCutPicture[0]);
cv::medianBlur(m_rightEyeCutPicture[0],m_rightEyeCutPicture[0],3);
            cv::Mat kernel =getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
            cv::erode(m_rightEyeCutPicture[0],m_rightEyeCutPicture[0], kernel);
            //resizeImage();
           // cv::erode(m_rightEyeCutPicture[1],m_rightEyeCutPicture[1],kernel);
            imshow("0", m_rightEyeCutPicture[0]);
                // initialize all intensity values to 0
//FitLine(m_rightEyeCutPicture[0]);
////////////////////////////////////////////////////////////////////////////////////////////////
           /*
                // Generate the histogram
    int histogram[256];
    imhist(m_rightEyeCutPicture[0], histogram);
 
    // Calculate the size of image
    int size = m_rightEyeCutPicture[0].rows * m_rightEyeCutPicture[0].cols;

    // Calculate the probability of each intensity
    float PrRk[256];
    for(int i = 0; i < 256; i++)
    {
        PrRk[i] = (double)histogram[i] / size;
    }
 
    // Generate the equalized histogram
    float PsSk[256];
    for(int i = 0; i < 256; i++)
    {
        PsSk[i] = 0;
    }
 
    for(int i = 0; i < 256; i++)
  for(int j=0; j<=i; j++)
          PsSk[i] += PrRk[j];

    int final[256];
    for(int i = 0; i < 256; i++)
        final[i] = cvRound(PsSk[i]*255);

 for(int i = 0; i < 256; i++)
  for(int j=0; j<=255; j++)
          if (final[i]==final[j] && i!=j)
           {
           final[i]+=final[j];
        } 

 int final1[256];
    for(int i = 0; i < 256; i++)
 {
  final1[i]=0;
 }
  
    for(int i = 0; i < 256; i++)
 {
        final1[cvRound(PsSk[i]*255)] =cvRound(PrRk[i]*size);
 }

    for(int i = 0; i < 256; i++)
  for(int j=0; j<256; j++)
          if (final1[i]==final[j] && i!=j)
           {
          final1[i]+=final1[j];
          std::cout<<"final1["<<i<<"]="<<final1[i]<<std::endl;
        }

    // Generate the equlized image
    cv::Mat new_image = m_rightEyeCutPicture[0].clone();

    for(int y = 0; y < m_rightEyeCutPicture[0].rows; y++)
        for(int x = 0; x < m_rightEyeCutPicture[0].cols; x++)
            new_image.at<uchar>(y,x) = cv::saturate_cast<uchar>(final[m_rightEyeCutPicture[0].at<uchar>(y,x)]);
 
   // Display the original Image
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", m_rightEyeCutPicture[0]);
 
    // Display the original Histogram
    histDisplay(histogram, "Original Histogram");

    // Display the equilzed histogram
    histDisplay(final1, "Equilized Histogram");

    // Display equilized image
    cv::namedWindow("Equilized Image");
    cv::imshow("Equilized Image",new_image);
          //  imshow("1", m_rightEyeCutPicture[1]);*/
        }
    }

    cv::Point segmentation(cv::Mat wejsciowe){
     cv::Mat eyesKernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        cv::Rect tempEye;
        bool IsThereEyes;
       cv::Mat src;
        if (!IsThereEyes && !m_RightEye.empty())
            IsThereEyes = true;
        if (IsThereEyes) {
            src.resize(2);
            tempEye.y = m_RightEye[0].y+m_RightEye[0].height / 4;
            tempEye.height = m_RightEye[0].height / 2;
            tempEye.x=m_RightEye[0].x;
            tempEye.width=m_RightEye[0].width;
            src = wejsciowe(tempEye);
int newHeight = m_rightEyeCutPicture[0].rows*2;
        int newWidth = m_rightEyeCutPicture[0].cols*2;
      //  resize(m_rightEyeCutPicture[0], m_rightEyeCutPicture[0], cv::Size( newWidth, newHeight), cv::INTER_LINEAR);
     
     cv::Mat mask;
    inRange(src, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
    src.setTo(cv::Scalar(0, 0, 0), mask);
    // Show output image
    imshow("Black Background Image", src);
    cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1);


        cv::Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    cv::Mat sharp;
    src.convertTo(sharp, CV_32F);
    cv::Mat imgResult = sharp - imgLaplacian;
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    imshow( "New Sharped Image", imgResult );
imshow( "Laplace Filtered Image", imgLaplacian );

    cv::Mat bw;
    cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
   
    cv::Mat kernel2 =getStructuringElement(cv::MORPH_RECT, cv::Size(4,4));
    erode(bw,bw,kernel2);
    imshow("Binary Image", bw);
    cv::Mat dist;
    distanceTransform(bw, dist, cv::DIST_L2, 3);
    normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);    
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    std::vector<std::vector<cv::Point> > contours;
    findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    cv::Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers", markers8u);
    // Perform the watershed algorithm
    watershed(imgResult, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
        std::vector<cv::Vec3b> colors;
        int b = 100;
        int g = 180;
        int r = 80;
    for (size_t i = 0; i < contours.size(); i++)
    {
         b +=20;
         g +=20;
         r +=18;
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
  for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    imshow("bbb", dst);
    // Visualize the final image

   /*  cv::Mat dst1(dst.cols, dst.rows, CV_32FC1);

cv::matchTemplate(dst, kolecko, dst1, cv::TM_CCORR_NORMED);
double minval, maxval;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

        rectangle(wejsciowe, cv::Point(maxloc.x, maxloc.y),
                  cv::Point(maxloc.x + 30- 1, maxloc.y + 30 - 1),
                  cv::Scalar(255, 0, 0), 3, 8, 0);
    imshow("Final Result", dst);*/
cv::Mat temp;
        cvtColor(dst, temp, cv::COLOR_BGR2GRAY);
    medianBlur(temp, temp, 5);
    std::vector<cv::Vec3f> circles;
     cv::HoughCircles(temp, circles, cv::HOUGH_GRADIENT, 1.5,
                             temp.cols, 100, 20, temp.rows / 4,
                             temp.rows / 2);   
                               cv::Vec3i c;cv::Point center ;
    for (size_t i = 0; i < circles.size(); i++) {
                c = circles[i];
                 center = cv::Point(c[0], c[1]);
                // circle center
                circle(wejsciowe, cv::Point(center.x + m_RightEye[0].x, center.y+m_RightEye[0].height/4+ m_RightEye[0].y), 1,
                       cv::Scalar(255, 150, 0), 3, cv::LINE_AA);
                // circle outline
                 int a = c[2];
                circle(wejsciowe, cv::Point(center.x + m_RightEye[0].x, center.y +m_RightEye[0].height/4+ m_RightEye[0].y), a,
                       cv::Scalar(25, 200, 255), 3, cv::LINE_AA);
           }
           for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                bool in=false;
               double a= pointPolygonTest(contours[index-1],center,in);
                
                if(a>0)
                   dst.at<cv::Vec3b>(i,j) = cv::Vec3b((uchar)255, (uchar)255, (uchar)0);
           else
                   dst.at<cv::Vec3b>(i,j) = cv::Vec3b((uchar)0, (uchar)0, (uchar)0);
            
            }
        }
    }
    imshow("aa", dst);
    return center;

}}
    
    void findingCircles(cv::Mat& mainImage) {
        std::vector<cv::Vec3f> rightEyeCircles;
        if (!m_rightEyeCutPicture.empty()) {
            cv::HoughCircles(m_rightEyeCutPicture[0], rightEyeCircles, cv::HOUGH_GRADIENT, 1.5,
                             m_rightEyeCutPicture[0].cols, 100, 20, m_rightEyeCutPicture[0].rows / 4,
                             m_rightEyeCutPicture[0].rows/2 );
            //cv::Point center;
            cv::Vec3i c;
            // rysowanie okręgów

            for (size_t i = 0; i < rightEyeCircles.size(); i++) {
                c = rightEyeCircles[i];
                m_RightEyeCenter = cv::Point(c[0], c[1]);
                // circle center
               //circle(m_rightEyeCutPicture[0], m_RightEyeCenter, 1, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
                circle(temp, cv::Point(m_RightEyeCenter.x/4 + m_RightEye[0].x, m_RightEyeCenter.y/4+m_RightEye[0].height/4+ m_RightEye[0].y), 1,
                       cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
                // circle outline
                 m_rightEyeRadius = c[2];
               // circle(m_rightEyeCutPicture[0], m_RightEyeCenter, m_rightEyeRadius, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                circle(temp, cv::Point(m_RightEyeCenter.x/4 + m_RightEye[0].x, m_RightEyeCenter.y /4+m_RightEye[0].height/4+ m_RightEye[0].y), m_rightEyeRadius/4,
                       cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            }

imshow("A",m_rightEyeCutPicture[0]);
            /*std::vector<cv::Vec3f> leftEyeCircles;
            cv::HoughCircles(m_rightEyeCutPicture[1], leftEyeCircles, cv::HOUGH_GRADIENT, 1.5,
                             m_rightEyeCutPicture[1].cols, 100, 20, m_rightEyeCutPicture[1].rows / 4,
                             m_rightEyeCutPicture[1].rows / 2);*/
/*
            // rysowanie okręgów
            for (size_t i = 0; i < leftEyeCircles.size(); i++) {
                c = leftEyeCircles[i];
                m_LeftEyeCenter = cv::Point(c[0], c[1]);
                // circle center
                circle(m_rightEyeCutPicture[1], m_LeftEyeCenter, 1, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
                circle(mainImage, cv::Point(m_LeftEyeCenter.x + m_LeftEye[0].x, m_LeftEyeCenter.y +m_LeftEye[0].height/4+ m_LeftEye[0].y), 1,
                       cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
                // circle outline
                int radius = c[2];
                circle(m_rightEyeCutPicture[1], m_LeftEyeCenter, radius, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
                circle(mainImage, cv::Point(m_LeftEyeCenter.x + m_LeftEye[0].x, m_LeftEyeCenter.y+m_LeftEye[0].height/4+ m_LeftEye[0].y), radius,
                       cv::Scalar(0, 255, 255), 3, cv::LINE_AA);

            }*/
            /*  if ((center.x - prev[0]) <= 4 && (center.y - prev[1]) <= 3)
                    iterator++;
                else
                    iterator = 0;
                prev = c;
                std::cout << prev[0] << "   " << prev[1] << "\n";
                rect.x = prev[0] - prev[2] / 2;
                rect.y = prev[1] - prev[2] / 2;

                rect.height = prev[2];
                rect.width = prev[2];
                std::cout << rect.x << "   " << rect.y << "\n";
                //cout << iterator << "\n";
                tpl = CutGray[0](rect);*/

        }
    }

    void whereAreCircles(cv::Mat& mainImage, Display* display)
    {

        if(m_RightEyeCenter.x!=0 || m_RightEyeCenter.y!=0) {
            if (m_rightEyeCenterPrev.x - m_RightEyeCenter.x < 2 && m_rightEyeCenterPrev.x - m_RightEyeCenter.x > -2 &&
                m_rightEyeCenterPrev.y - m_RightEyeCenter.y < 2 && m_rightEyeCenterPrev.y - m_RightEyeCenter.y > -2)
                licznik++;
            else
                licznik = 0;
            if (licznik == 5) {
                std::cout << "Znaleziono oko";
               // XBell(display, 1000);
                XFlush(display);
                //cv::destroyWindow("Info"); //kom
                boolEye = true;
                //rect
                m_rightEyeRect.x = m_RightEyeCenter.x - m_rightEyeRadius + m_RightEye[0].x;
                m_rightEyeRect.y = m_RightEyeCenter.y +m_RightEye[0].height / 4- m_rightEyeRadius + m_RightEye[0].y;
                m_rightEyeRect.width = 2 * m_rightEyeRadius;
                m_rightEyeRect.height = 2 * m_rightEyeRadius;
                //mat
                m_rightEyeMat=mainImage(m_rightEyeRect).clone();
                licznik=0;
                StworzenieEkranuKalibracji();
             //   cv::imshow("Korelacja", m_rightEyeMat);
            }
            //std::cout << licznik << "\n";
            m_rightEyeCenterPrev = m_RightEyeCenter;
        }
    }

    bool BlinkingDetection(cv::Mat grayScaleImage,Display* display)
    {
        std::pair<double, double> line= FitLine(grayScaleImage);
        static int i = 0;
        std::cout<<i<<"\n";
        double l =std::abs(line.first/line.second) ;
        //std::cout <<"\n"<<ifBlinked<<"\n";
        if(l <1.5 && l>0.5){
        if (i == 20 ){
                std::cout << "\n\nMRUGNIECIE\n\n"<<line.first<<"/"<<line.second<<"="<<l;
                ifBlinked = true;
                mrugniecie = true;
                XBell(display, 1000);
                XFlush(display);
                i=0;
                return true;
            }
            else{
                mrugniecie=false;
                i++;
                return false;

            }
        }
        else {
                i=0;
            mrugniecie=false;
            ifBlinked = false;
         return false;
        }

    }
    
    void cutEyes( cv::Mat grayScaleImage) {
        //std::vector<cv::Mat> CutGray;
        cv::Mat eyesKernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        cv::Rect tempEye;
        bool IsThereEyes;
        if (!IsThereEyes && !m_RightEye.empty())
            IsThereEyes = true;
        if (IsThereEyes) {
            m_rightEyeCutPicture.resize(2);
            tempEye.y = m_RightEye[0].y+m_RightEye[0].height / 4;
            tempEye.height = m_RightEye[0].height / 2;
            tempEye.x=m_RightEye[0].x;
            tempEye.width=m_RightEye[0].width;
            m_rightEyeCutPicture[0] = grayScaleImage(tempEye);
        }
    }
    
    //wywalic te wszyskie po opisaniu
    int srednia=0;
    int i=0;
    bool m_kalibracjaMrugania=false;
    cv::Mat temp;
 
    double findingEyeCenter(cv::Mat grayScaleImage)
    {
        int y1=0, y2=0;
        int x1=0, x2=0;
        cv::Mat dst;
        cutEyes(grayScaleImage);
       // threshold(m_rightEyeCutPicture[0], dst, 0, 255, cv::THRESH_OTSU | cv:: THRESH_BINARY_INV);
        cv::Canny(m_rightEyeCutPicture[0],dst,150,255,3);
        for(int i=0; i<dst.rows; i++)
            for(int j=0; j<dst.cols; j++)
                if(dst.at<int>(i, j)>0) {
                    if(y1>i || y1==0)
                        y1=i;
                    else {
                        if (y2 < i)
                            y2 = i;
                    }
                    if(x1>j || x1==0)
                        x1=j;
                    else {
                         if (x2 < j)
                            x2 = j;
                    }
                }
        return  (double)(y2-y1)/(x2-x1);
        //zrobić prawo lewo i podzielic
    }

    void trackRightEye( cv::Mat& mainFrame)
    {
       // cv::Size size(m_rightEyeRect.width * 2, m_rightEyeRect.height * 2);
        cv::Rect window(m_RightEye[0].x+m_RightEye[0].width/4,m_RightEye[0].y,m_RightEye[0].width,m_RightEye[0].height);

        //window &= cv::Rect(0, 0, mainFrame.cols, mainFrame.rows);

        cv::Mat dst(window.width - m_rightEyeMat.rows/4 + 1, window.height - m_rightEyeMat.cols/4 + 1, CV_32FC1);
        cv::matchTemplate(mainFrame(window), m_rightEyeMat, dst, cv::TM_SQDIFF_NORMED);
 cv::imshow("A",m_rightEyeMat);
        double minval, maxval;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);
        if (minval <= 0.2)
        {
            if(!skalibrowano)
           { m_rightEyeRect.x = window.x + minloc.x;
            m_rightEyeRect.y = window.y + minloc.y;}
            else{
               // if((window.x + minloc.x+10>MaxEkran.x && window.x + minloc.x-10<MaxEkran.x+MaxEkran.width) || (window.y + minloc.y+ 5>MaxEkran.y && window.y + minloc.y-5>MaxEkran.y+MaxEkran.height))
            m_rightEyeRect.x = window.x + minloc.x;
            m_rightEyeRect.y = window.y + minloc.y;
            //else
            //std::cout<<"F";
            }
            //mrugniecie=false;
        }
        rectangle(temp, cv::Point(m_rightEyeRect.x, m_rightEyeRect.y),
                  cv::Point(m_rightEyeRect.x + m_rightEyeRect.width - 1, m_rightEyeRect.y + m_rightEyeRect.height - 1),
                  cv::Scalar(255, 0, 0), 3, 8, 0);
        /*else {
            mrugniecie=true;
            if(mrugniecie)
                blinkingDetection(mainFrame);
        }*/
    }

    //MAPOWANIE 
    cv::Point EkranPx=cv::Point(1920,1080);
    double rozmiarEkranuX=33;
    double rozmiarEkranuY=20.3;
    double rozmiarOkaCmX;
    double rozmiarOkaCmY;
    int d=50;
    double okoNaEkranieX;
    double okoNaEkranieY;
    cv::Point2d okoNaEkraniePX;
    bool skalibrowano=false;
    cv::Rect MaxEkran;

    int roznicaX, roznicaY, srodekWyliczonyX,srodekWyliczonyY;
    void przeliczenieNaEkranPx(cv::Rect rect)
    {
        rozmiarOkaCmX=d*(rozmiarEkranuX/(4*d));
        rozmiarOkaCmY=d*(rozmiarEkranuY/(4*d));

        float OkoNaEkranieCmX=2*rozmiarOkaCmX*(srodekWyliczonyX-(rect.x+rect.width/2))/roznicaX;
       /* std::cout<<srodekWyliczonyX<<"srodekWyliczonyX\n";
        std::cout<<roznicaX<<"roznicaX\n";
        std::cout<<OkoNaEkranieCmX<<"OkoNaEkranieCmX\n";
        std::cout<<m_rightEyeRect.x+m_rightEyeRect.width/2<<"m_rightEyeRect.x+m_rightEyeRect.width/2\n";*/

        okoNaEkranieX=2*d*(OkoNaEkranieCmX/d);
       // std::cout<<okoNaEkranieX<<"okoNaEkranieX\n";

        float OkoNaEkranieCmY=2*rozmiarOkaCmY*(srodekWyliczonyY-(rect.y+rect.height/2))/roznicaY;
       // std::cout<<srodekWyliczonyY<<"srodekWyliczonyX\n";
       // std::cout<<OkoNaEkranieCmY<<"OkoNaEkranieCmY\n";

        okoNaEkranieY=2*d*(OkoNaEkranieCmY/d);
       // std::cout<<okoNaEkranieY<<"okoNaEkranieY\n";

        okoNaEkraniePX.x=okoNaEkranieX*EkranPx.x/rozmiarEkranuX+960;
      //  std::cout<<okoNaEkraniePX.x<<"okoNaEkraniePX\n";

        okoNaEkraniePX.y=1080-okoNaEkranieY*EkranPx.y/rozmiarEkranuY-540;
      //  std::cout<<okoNaEkraniePX.y<<"\n";

       // std::cout<<srodekWyliczonyX<<" "<<srodekWyliczonyX-m_rightEyeRect.x+m_rightEyeRect.width/2 <<" "<<m_rightEyeRect.x+m_rightEyeRect.width/2<<" "<<m_rightEyeRect.y+m_rightEyeRect.height/2<<"\n";
      //  std::cout<<okoNaEkraniePX<<"\n";
    }

    void przeliczenieNaEkranKat(cv::Rect rect){
        double tanThetaY, tanThetaX, odlX, odlY, a=58.2;
        //tanThetaY=rozmiarEkranuY/d;
        //odlY=0.017*(roznicaY/2)*d/(rozmiarEkranuY/2-0.017*(roznicaY/2));
        //odlX=0.017*(srodekWyliczonyX+roznicaX/2)*d/(rozmiarEkranuX-0.017*(srodekWyliczonyX+roznicaX/2));
        odlX=d;
        odlY=d;
       //   rozmiarOkaCmX=odlX*(rozmiarEkranuX/(4*d));
        //rozmiarOkaCmY=odlY*(rozmiarEkranuY/(4*d));

        float OkoNaEkranieCmX=1920*(srodekWyliczonyX-(rect.x+rect.width/2))/roznicaX;

      //  okoNaEkranieX=2*d*(OkoNaEkranieCmX/odlX);

        float OkoNaEkranieCmY=1080*(srodekWyliczonyY-(rect.y+rect.height/2))/roznicaY;

       // okoNaEkranieY=2*d*(OkoNaEkranieCmY/odlY);

        okoNaEkraniePX.x=OkoNaEkranieCmX+960;

        okoNaEkraniePX.y=540-OkoNaEkranieCmY;
        std::cout<<okoNaEkraniePX.x<<" ||" <<okoNaEkraniePX.y<<std::endl;

       /* double OkoNaEkranieCmY=(d+odlY)*0.017*(rect.y+rect.height/2-srodekWyliczonyY)/odlY;
        okoNaEkranieY=1080/2+a*(OkoNaEkranieCmY); 
        std::cout<<OkoNaEkranieCmY<<"AAA"<<okoNaEkranieY<<std::endl;
        okoNaEkraniePX.y=1080-okoNaEkranieY*EkranPx.y/rozmiarEkranuY-540;

        //tanThetaX=rozmiarEkranuX/d;
        float OkoNaEkranieCmX=(d+odlX)*0.017*(rect.x)/odlX;
        okoNaEkranieX=a*(OkoNaEkranieCmX); 
       // std::cout<<okoNaEkranieX<<std::endl;
        okoNaEkraniePX.x=okoNaEkranieX*EkranPx.x/rozmiarEkranuX+960;*/
       // std::cout<<odlX<<"||"<<odlY<<std::endl;
    }

    std::vector<cv::Point> punkty;
    void StworzenieEkranuKalibracji() {
        for(int i=0; i<5; i++)
            punkty.push_back(cv::Point(0,0));
    }
    std::vector<int> ZebranePunktyX, ZebranePunktyY;
    int h = 50;
    int j=50;
    void StworzenieEkranuZInformacja(std::string sInfo) {
        cv::Mat Info(EkranPx.y, EkranPx.x, CV_8UC3, cv::Scalar(0, 0, 0));
        namedWindow("Info", cv::WINDOW_NORMAL);
        setWindowProperty("Info", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        putText(Info, sInfo, cv::Point2f(50, 150), cv::FONT_ITALIC, 1, cv::Scalar(0, 0, 255), 2, 8, false);
        imshow("Info", Info);
        std::this_thread::sleep_for(std::chrono::seconds{3});
    }
    void Kalibracja(){
        cv::destroyWindow("Info");
        cv::Mat konfiguracja(EkranPx.y, EkranPx.x, CV_8UC3, cv::Scalar(0, 0, 0));
        namedWindow("Konfiguracja", cv::WINDOW_NORMAL);
        setWindowProperty("Konfiguracja", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
       
        circle(konfiguracja, cv::Point(h, j), 50, cv::Scalar(250, 0, 125), cv::FILLED);
                ZebranePunktyX.push_back(m_rightEyeRect.x + m_rightEyeRect.width / 2);
                ZebranePunktyY.push_back(m_rightEyeRect.y + m_rightEyeRect.height / 2);
        if(!skalibrowano)
            imshow("Konfiguracja", konfiguracja);

        circle(konfiguracja, cv::Point(h, j), 50, cv::Scalar(0, 0, 0), cv::FILLED);
        if(h<1850 &&j==50)
            h+=25;
        else if(h==1850 && j<1000)
            j+=25;
        else if(j==1000 && h>50)
            h-=25;
        else if(h==50 && j>50)
            j-=25;
        if(h==50 && j==50)
            skalibrowano=true;
        if(skalibrowano) {

            circle(konfiguracja, cv::Point(50, 50), 50, cv::Scalar(0, 0, 0), cv::FILLED);
         
            roznicaX = *std::max_element(ZebranePunktyX.begin(), ZebranePunktyX.end()) -
                       *std::min_element(ZebranePunktyX.begin(), ZebranePunktyX.end());
            roznicaY = *std::max_element(ZebranePunktyY.begin(), ZebranePunktyY.end()) -
                       *std::min_element(ZebranePunktyY.begin(), ZebranePunktyY.end());
            srodekWyliczonyX=*std::max_element(ZebranePunktyX.begin(), ZebranePunktyX.end())-roznicaX/2;
            srodekWyliczonyY=*std::max_element(ZebranePunktyY.begin(), ZebranePunktyY.end())-roznicaY/2;
            MaxEkran.x=*std::min_element(ZebranePunktyX.begin(), ZebranePunktyX.end());
            MaxEkran.y=*std::min_element(ZebranePunktyY.begin(), ZebranePunktyY.end());
            MaxEkran.width=*std::max_element(ZebranePunktyY.begin(), ZebranePunktyY.end())-*std::min_element(ZebranePunktyY.begin(), ZebranePunktyY.end());
            MaxEkran.height=*std::max_element(ZebranePunktyY.begin(), ZebranePunktyY.end())-*std::min_element(ZebranePunktyY.begin(), ZebranePunktyY.end());
            cv::destroyWindow("Konfiguracja");
        }
    }

    void PokazanieNaEkranie(Display *display)
    {
        /*cv::Mat koniec(EkranPx.y, EkranPx.x, CV_8UC3, cv::Scalar(0, 0, 0));
        namedWindow("koniec", cv::WINDOW_NORMAL);
        setWindowProperty("koniec", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

        circle(koniec, okoNaEkraniePX, 50, cv::Scalar(250, 0, 125), cv::FILLED);
        imshow("koniec", koniec);
*/
        //obsługa myszy
        Window root_window;
    root_window = XRootWindow(display, 0);
    XSelectInput(display, root_window, KeyReleaseMask);
        XWarpPointer(display, None, root_window, 0,0,1920,1080,okoNaEkraniePX.x, okoNaEkraniePX.y);
       // std::cout<<okoNaEkraniePX.x<<"||"<<okoNaEkraniePX.y<<"\n";
   XSync(display, False); 
   
    if(mrugniecie==True)
    {
         
        XEvent event;
  memset (&event, 0, sizeof (event));
  event.xbutton.button = Button1;
  event.xbutton.same_screen = True;
  event.xbutton.subwindow = DefaultRootWindow (display);
  while (event.xbutton.subwindow)
    {
      event.xbutton.window = event.xbutton.subwindow;
      XQueryPointer (display, event.xbutton.window,
		     &event.xbutton.root, &event.xbutton.subwindow,
		     &event.xbutton.x_root, &event.xbutton.y_root,
		     &event.xbutton.x, &event.xbutton.y,
		     &event.xbutton.state);
    }
  // Press
  event.type = ButtonPress;
  if (XSendEvent (display, PointerWindow, True, ButtonPressMask, &event) == 0)
    fprintf (stderr, "Error to send the event!\n");
  XFlush (display);
     XSync(display, False); 
  // Release
  event.type = ButtonRelease;
  if (XSendEvent (display, PointerWindow, True, ButtonReleaseMask, &event) == 0)
    fprintf (stderr, "Error to send the event!\n");
  XFlush (display);
     XSync(display, False); 
    }
    
      //  circle(koniec, okoNaEkraniePX, 50, cv::Scalar(0, 0, 0), cv::FILLED);
    }
};

class CKalman{
public:
    int m_iRightCenterX, m_iRightCenterY,m_iLeftCenterX, m_iLeftCenterY;
    std::vector<cv::Point> eyev,kalmanv;
    cv::Mat measurement = cv::Mat::zeros(1, 1, CV_32F);
//oko prawe
    cv::Mat_<float>  rightEyePosition;
    //cv::Point  measurePointR;
    cv::Point  calculatePointR;
    cv::Point actualPointR;
    cv::KalmanFilter rightEyeFilter;
//oko Lewe
    cv::Mat leftEyePosition;
    cv::Point  measurePointL;
    cv::Point  calculatePointL;
    //cv::Point actualPointL = measurePointL;
    cv::KalmanFilter leftEyeFilter;


cv::KalmanFilter kf;
    cv::Mat state;  // [x,y,v_x,v_y,w,h]
     cv::Mat meas;
    CKalman(int stateSize,int measSize, int contrSize, unsigned int type) 
    :kf(stateSize, measSize, contrSize, type),
    state(stateSize, 1, type),
    meas(measSize, 1, type)
    {
/*        m_iRightCenterX=0;
        m_iRightCenterY=0;
        m_iLeftCenterX=0;
        m_iLeftCenterY=0;
        leftEyePosition(2,1);
        rightEyePosition(2,1);
        //measurePointR = cv::Point( m_iRightCenterX, m_iRightCenterY);
        //calculatePointR = measurePointR;
       // actualPointR = measurePointR;
        //rightEyeFilter(4,2,0);
        measurePointL = cv::Point( m_iLeftCenterX, m_iLeftCenterY);
        calculatePointL = measurePointL;
        //leftEyeFilter(4,2,0);*/


    }
    void KalmanPreWhileLoop()
    {

    int stateSize = 4;
    int measSize = 2;
    int contrSize = 0;
    unsigned int type = CV_32F;
    cv::setIdentity( kf.measurementMatrix,
cv::Scalar(1)
);
cv::setIdentity( kf.processNoiseCov,
cv::Scalar(1e-6) );
cv::setIdentity( kf.measurementNoiseCov, cv::Scalar(1e-3) );
cv::setIdentity( kf.errorCovPost,
cv::Scalar(1)
);
 /*
    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
     cv::Mat meas(measSize, 1, type);
     cv::setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

 kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;
kf.statePre.setTo(0);
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

       rightEyePosition(1,2,0);
        rightEyeFilter.init(4,2, 0);
        rightEyeFilter.transitionMatrix = (cv::Mat_<float>(4,4) << 1, 0, 1, 0,  0,1,0,1, 0,0,1,0, 0,0,0,1);

        //Inicjalizacja wektora stanu z fazy predykcji

        rightEyeFilter.statePre.at<float>(0) = Right.x;        //Położenie x
        rightEyeFilter.statePre.at<float>(1) = Right.y;        //Położenie y
        rightEyeFilter.statePre.at<float>(2) = 0;                  //Prędkosc x
        rightEyeFilter.statePre.at<float>(3) = 0;                  //Prędkosc y
        setIdentity (rightEyeFilter.measurementMatrix);
        setIdentity (rightEyeFilter.processNoiseCov, cv::Scalar::all(1e-4));
        setIdentity(rightEyeFilter.measurementNoiseCov, cv::Scalar::all(1e-1));
        setIdentity(rightEyeFilter.errorCovPost, cv::Scalar::all(.1));
        randn(rightEyeFilter.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));
/*
        leftEyeFilter.transitionMatrix = (cv::Mat_<float>(4,4) << 1, 0, 1, 0,  0,1,0,1, 0,0,1,0, 0,0,0,1);
        //Inicjalizacja wektora stanu z fazy predykcji
        leftEyeFilter.statePre.at<float>(0) = m_iLeftCenterX;        //Położenie x
        leftEyeFilter.statePre.at<float>(1) = m_iLeftCenterY;        //Położenie y
        leftEyeFilter.statePre.at<float>(2) = 0;        //Prędkosc x
        leftEyeFilter.statePre.at<float>(3) = 0;        //Prędkosc y
        setIdentity (leftEyeFilter.measurementMatrix);
        setIdentity (leftEyeFilter.processNoiseCov, cv::Scalar::all(1e-4));
        setIdentity(leftEyeFilter.measurementNoiseCov, cv::Scalar::all(1e-1));
        setIdentity(leftEyeFilter.errorCovPost, cv::Scalar::all(.1));

*/
    }

    void KalmanStatePre( CEyeDetection eyeDetectionObject)
    {
        kf.statePre.at<float>(0) = eyeDetectionObject.m_RightEyeCenter.x+25;
        kf.statePre.at<float>(1) = eyeDetectionObject.m_RightEyeCenter.y+25;
        kf.statePre.at<float>(2) = 0;
        kf.statePre.at<float>(3) = 0;
    }
bool found = false;
    double ticks = 0;
    int z=0;

            cv::Point center;

            cv::Rect predRect;
    void actualKalman( CEyeDetection eyeDetectionObject, cv::Mat& mainImage) {


             static bool czy=false; 
            if(!czy)
            {
                KalmanStatePre(eyeDetectionObject);
                czy=true;
            }
            else
            state = kf.predict();
            std::cout << "State post:" << std::endl << state << std::endl;
 
            predRect.width = state.at<float>(2);
            predRect.height = state.at<float>(3);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;
 
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(eyeDetectionObject.temp, center/*cv::Point(center.x + eyeDetectionObject.m_RightEye[0].x, center.y +eyeDetectionObject.m_RightEye[0].height/4+ eyeDetectionObject.m_RightEye[0].y)*/, 2, CV_RGB(255,0,0), -1);
 
            cv::rectangle(eyeDetectionObject.temp, predRect/*cv::Rect(predRect.x + eyeDetectionObject.m_RightEye[0].x, predRect.y +eyeDetectionObject.m_RightEye[0].height/4+ eyeDetectionObject.m_RightEye[0].y,predRect.height, predRect.width)*/, CV_RGB(255,0,0), 2);

            if(!eyeDetectionObject.boolEye)
            {
            cv::Rect ballsBox=cv::Rect(eyeDetectionObject.m_RightEyeCenter,
            cv::Point(eyeDetectionObject.m_RightEyeCenter.x+25, eyeDetectionObject.m_RightEyeCenter.y+25));
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
               // kf.errorCovPre.at<float>(28) = 1e-5; // px
               // kf.errorCovPre.at<float>(35) = 1e-5; // px
 
                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
               // state.at<float>(2) = 0;
               /// state.at<float>(3) = 0;
                state.at<float>(2) = meas.at<float>(2);
                state.at<float>(3) = meas.at<float>(3);
                // <<<< Initialization
 
                kf.statePost = state;
                
                found = true;
            }
            else
                kf.correct(meas); 
            }
            else 
            {
            meas.at<float>(0) = eyeDetectionObject.m_rightEyeRect.x + eyeDetectionObject.m_rightEyeRect.width / 2;
            meas.at<float>(1) = eyeDetectionObject.m_rightEyeRect.y + eyeDetectionObject.m_rightEyeRect.height / 2;
            meas.at<float>(2) = (float)eyeDetectionObject.m_rightEyeRect.width;
            meas.at<float>(3) = (float)eyeDetectionObject.m_rightEyeRect.height;
            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
               // kf.errorCovPre.at<float>(28) = 1; // px
               // kf.errorCovPre.at<float>(35) = 1; // px
 
                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                //state.at<float>(2) = 0;
                //state.at<float>(3) = 0;
                state.at<float>(2) = meas.at<float>(2);
                state.at<float>(3) = meas.at<float>(3);
                // <<<< Initialization
 
                kf.statePost = state;
                
                found = true;
            }
            else{
                cv::Mat estimated = kf.correct(meas); 
                 //   center= cv::Point(estimated.at<float>(0),estimated.at<float>(1));
             } }

        /*
        std::cout<<"A";
        cv::Mat prediction = rightEyeFilter.predict();
        cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

        rightEyePosition.push_back( Right.x);
        rightEyePosition.push_back(Right.y) ;
        cv::Mat estimated = rightEyeFilter.correct(rightEyePosition);
        rightEyePosition.release();
         /*  cv::Point statePt(estimated.at<float>(0), estimated.at<float>(1));
           cv::Point measPt(rightEyePosition(0), rightEyePosition(1));
    /*
              eyev.push_back(measPt);
              kalmanv.push_back(statePt);
              for (int i = 0; i < eyev.size() - 1; i++)
                  line(img, eyev[i], eyev[i + 1], cv::Scalar(255, 255, 0), 1);

              for (int i = 0; i < kalmanv.size() - 1; i++)
                  line(img, kalmanv[i], kalmanv[i + 1], cv::Scalar(0, 155, 255), 1);

              /*  rightEyePosition(0)=m_iRightCenterX;
              rightEyePosition(1)=m_iRightCenterY;

      //        cv::Mat prediction = rightEyeFilter.predict();
              cv::Mat estimated = rightEyeFilter.correct(rightEyePosition);
              cv::Point estPt(estimated.at<float>(0),estimated.at<float>(1));
              cv::Point measurePoint(rightEyePosition(0),rightEyePosition(1));
              m_iRightCenterX =actualPointR.x;
              m_iRightCenterY =actualPointR.y;
              actualPointR =estPt;
      
        leftEyeFilter.predict();
        measurePointL=m_iLeftCenterX ;
        measurePointL(1)=m_iLeftCenterY;
        cv::Mat estimatedl = leftEyeFilter.correct(measurePointL);
        cv::Point estPtl(estimatedl.at<float>(0),estimatedl.at<float>(1));
        m_iLeftCenterX =actualPointL.x;
        m_iLeftCenterY =actualPointL.y;
        actualPointL =estPtl;*/
    }
};

int main(int argc, char *argv[]) {
    // >>>> Kalman Filter
    /*int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
 
    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
 
    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
     cv::Mat meas(measSize, 1, type);
     cv::setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

 kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));*/
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


/*
    QApplication app(argc, argv);
    QWidget window;
    window.resize(320, 240);
    window.show();
    window.setWindowTitle(
        QApplication::translate("toplevel", "Top-level widget"));
     app.exec();
*/
    while(cv::waitKey(20)!=27) {

        CapturedImage >> mainImage;
        cvtColor(mainImage, grayScaleImage, cv::COLOR_RGB2GRAY);
        resize(mainImage, eyeDetectionObject.temp,cv::Size( mainImage.cols, mainImage.rows), cv::INTER_LINEAR);
       
        //kom
       if(!eyeDetectionObject.boolEye) {
            eyeDetectionObject.faceCascade(faceCascade, grayScaleImage, mainImage);
            eyeDetectionObject.eyeCascade(eyeCascade, mainImage, grayScaleImage);
            eyeDetectionObject.cutEyesArea(grayScaleImage);
            //eyeDetectionObject.resizeImage();
            // eyeDetectionObject.tresholdMedianValue();
            //cv::Point temp = eyeDetectionObject.m_RightEyeCenter;
            //KOM
            eyeDetectionObject.StworzenieEkranuZInformacja("Prosze patrzec w kamere laptopa do momentu uslyszenia dzwieku");

            eyeDetectionObject.findingCircles(mainImage);
            //Kalman.actualKalman(eyeDetectionObject, mainImage);
            eyeDetectionObject.whereAreCircles(mainImage, display);
            eyeDetectionObject.BlinkingDetection(eyeDetectionObject.m_rightEyeCutPicture[0],display);
            

       } 
        else {
            eyeDetectionObject.whereAreCircles(mainImage, display);
           eyeDetectionObject.BlinkingDetection(eyeDetectionObject.m_rightEyeCutPicture[0],display);
            eyeDetectionObject.trackRightEye(mainImage);
            //cv::imshow("Korelacja", eyeDetectionObject.m_rightEyeMat);

            Kalman.actualKalman(eyeDetectionObject, mainImage);

            //eyeDetectionObject.findingStraightLine(grayScaleImage);
           //kom
           if (!eyeDetectionObject.skalibrowano) {
                eyeDetectionObject.StworzenieEkranuZInformacja("Teraz nastapi kalibracja. Prosze wzrokiem sledzic kropke.");
                eyeDetectionObject.Kalibracja();
            }
            else {           
                  eyeDetectionObject.przeliczenieNaEkranKat(eyeDetectionObject.m_rightEyeRect);

                // eyeDetectionObject.przeliczenieNaEkranPx(eyeDetectionObject.m_rightEyeRect);

                eyeDetectionObject.przeliczenieNaEkranKat(Kalman.predRect);
                //eyeDetectionObject.przeliczenieNaEkranPx(Kalman.predRect);
                eyeDetectionObject.PokazanieNaEkranie(display);

            }
        }

       imshow("mainImage", mainImage);
       imshow("temp", eyeDetectionObject.temp);
    }
    return 0;
}
