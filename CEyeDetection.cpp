#include "CEyeDetection.h"

void CEyeDetection::faceCascade(cv::CascadeClassifier Cascade, cv::Mat grayScaleImage)
{
    Cascade.detectMultiScale(grayScaleImage, m_rDetectedFaces, 1.1, 15,
                             0 | cv::CASCADE_SCALE_IMAGE, cv::Size(0, 0));
    if( m_rDetectedFaces.empty())
        throw std::logic_error("Couldn't find any face");

}

void CEyeDetection::eyeCascade(cv::CascadeClassifier eyeCascade, cv::Mat grayScaleImage)
{
    static auto start = std::chrono::high_resolution_clock::now();
    cv::Mat FaceRight;

    //Using first object in vector because it is most likely detected face
    FaceRight = grayScaleImage(cv::Rect(m_rDetectedFaces[0].x, m_rDetectedFaces[0].y,
                                        m_rDetectedFaces[0].width / 2, m_rDetectedFaces[0].height / 2)); // saving upper right part of the face

    eyeCascade.detectMultiScale(FaceRight, m_rDetectedEyes, 1.1,20,
                                0|cv::CASCADE_SCALE_IMAGE, cv::Size(0,0));
    for(size_t i=0; i<m_rDetectedEyes.size();i++)
    {
        m_rDetectedEyes[i].x+=m_rDetectedFaces[0].x;
        m_rDetectedEyes[i].y+=m_rDetectedFaces[0].y;
    }
    if( m_rDetectedEyes.empty())
        throw std::logic_error("Couldn't find any eye");

    //If detected eye wont change their position for 10s it is saved
    if(m_rEyeRectPrev.x-m_rDetectedEyes[0].x<3 && m_rEyeRectPrev.x-m_rDetectedEyes[0].x>-3 && m_rEyeRectPrev.y-m_rDetectedEyes[0].y<3 && m_rEyeRectPrev.y-m_rDetectedEyes[0].y>-3 )
    {
        if(duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()- start).count() < 10000000)
            throw std::logic_error("Couldn't find any eye");
    }
    else
        start = std::chrono::high_resolution_clock::now();
    m_rEyeRectPrev=m_rDetectedEyes[0];  //Using first object in vector because it is most likely detected the right eye
}

void CEyeDetection::cutEyesArea( cv::Mat grayScaleImage) {
    cv::Mat eyesKernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    cv::Rect tempEye;

    if (!m_rDetectedEyes.empty()) {
        m_mEyeCutPicture.resize(2);
        tempEye.y = m_rDetectedEyes[0].y+m_rDetectedEyes[0].height / 4;
        tempEye.height = m_rDetectedEyes[0].height / 2;
        tempEye.x=m_rDetectedEyes[0].x;
        tempEye.width=m_rDetectedEyes[0].width;
        m_mEyeCutPicture[0] = grayScaleImage(tempEye);

        threshold(m_mEyeCutPicture[0], m_mEyeCutPicture[0], 0, 255, cv::THRESH_OTSU | cv:: THRESH_BINARY_INV);

        cv::medianBlur(m_mEyeCutPicture[0],m_mEyeCutPicture[0],3);
        cv::Mat kernel =getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
        cv::erode(m_mEyeCutPicture[0],m_mEyeCutPicture[0], kernel);
    }
}

cv::Point CEyeDetection::segmentation(cv::Mat wejsciowe){
    cv::Mat eyesKernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    cv::Rect tempEye;
    cv::Mat src;
    if (!m_rDetectedEyes.empty()) {
        src.resize(2);
        tempEye.y = m_rDetectedEyes[0].y+m_rDetectedEyes[0].height / 4;
        tempEye.height = m_rDetectedEyes[0].height / 2;
        tempEye.x=m_rDetectedEyes[0].x;
        tempEye.width=m_rDetectedEyes[0].width;
        src = wejsciowe(tempEye);

        cv::Mat mask;
        inRange(src, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
        src.setTo(cv::Scalar(0, 0, 0), mask);
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

        cv::Mat bw;
        cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
        threshold(bw, bw, 40, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat kernel2 =getStructuringElement(cv::MORPH_RECT, cv::Size(4,4));
        erode(bw,bw,kernel2);
        cv::Mat dist;
        distanceTransform(bw, dist, cv::DIST_L2, 3);
        normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
        threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
        cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
        dilate(dist, dist, kernel1);
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
        cv::Mat temp;
        cvtColor(dst, temp, cv::COLOR_BGR2GRAY);
        medianBlur(temp, temp, 5);
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(temp, circles, cv::HOUGH_GRADIENT, 1.5,temp.cols, 100, 20, temp.rows / 4,temp.rows / 2);
        cv::Vec3i c;cv::Point center ;
        for (size_t i = 0; i < circles.size(); i++) {
            c = circles[i];
            center = cv::Point(c[0], c[1]);
            // circle center
            circle(wejsciowe, cv::Point(center.x + m_rDetectedEyes[0].x, center.y+m_rDetectedEyes[0].height/4+ m_rDetectedEyes[0].y), 1,
                   cv::Scalar(255, 150, 0), 3, cv::LINE_AA);
            // circle outline
            int a = c[2];
            circle(wejsciowe, cv::Point(center.x + m_rDetectedEyes[0].x, center.y +m_rDetectedEyes[0].height/4+ m_rDetectedEyes[0].y), a,
                   cv::Scalar(25, 200, 255), 3, cv::LINE_AA);
        }

        for (int i = 0; i < markers.rows; i++)
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

        return center;
    }
}

void CEyeDetection::findingCircles(cv::Mat& mainImage) {
    std::vector<cv::Vec3f> IrisCicles;
    if (!m_mEyeCutPicture.empty()) {
        cv::HoughCircles(m_mEyeCutPicture[0], IrisCicles, cv::HOUGH_GRADIENT, 1.5,
                         m_mEyeCutPicture[0].cols, 100, 20, m_mEyeCutPicture[0].rows / 4,
                         m_mEyeCutPicture[0].rows/2 );

        m_pEyeCenter = cv::Point(IrisCicles[0][0], IrisCicles[0][1]);
        m_iEyeRadius = IrisCicles[0][2];
        static auto start = std::chrono::high_resolution_clock::now();

        if(m_pEyeCenter.x!=0 || m_pEyeCenter.y!=0) {
            if (!(m_pEyeCenterPrev.x - m_pEyeCenter.x < 2 && m_pEyeCenterPrev.x - m_pEyeCenter.x > -2 &&
                    m_pEyeCenterPrev.y - m_pEyeCenter.y < 2 && m_pEyeCenterPrev.y - m_pEyeCenter.y > -2))
                start = std::chrono::high_resolution_clock::now();
            if (duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()- start).count() >5000000) {
                m_rEyeRect.x = m_pEyeCenter.x - m_iEyeRadius + m_rDetectedEyes[0].x;
                m_rEyeRect.y = m_pEyeCenter.y +m_rDetectedEyes[0].height / 4- m_iEyeRadius + m_rDetectedEyes[0].y;
                m_rEyeRect.width = 2 * m_iEyeRadius;
                m_rEyeRect.height = 2 * m_iEyeRadius;
                m_EyeMat=mainImage(m_rEyeRect).clone();
            }
            m_pEyeCenterPrev = m_pEyeCenter;
        }
    }
}

std::pair<double, double> CEyeDetection::FitLine(cv::Mat Image){
    cv::Mat src=Image;
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_64F);
    cv::threshold(src, src, 177, 200, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point> >  contours;
    cv::Mat hierarchy;
    cv::Vec4f line;
    cv::findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    if( !contours.empty()){
        std::vector<cv::Point> cnt = contours[0];
        cv::fitLine(cnt, line, cv::DIST_L2, 0, 0.1, 0.1);
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
        return std::make_pair(point1.y, point2.y);
    }
    else
        return std::make_pair(0,0);
}

void CEyeDetection::BlinkingDetection(cv::Mat grayScaleImage,Display* display)
{
    std::pair<double, double> line= FitLine(grayScaleImage);
    static int i = 0;
    std::cout<<i<<"\n";
    double l =std::abs(line.first/line.second) ;

    if(l <1.5 && l>0.5){
        if (i == 20 ){
            m_bIfBlinked = true;
            XBell(display, 1000);
            XFlush(display);
            i=0;
        }
        else{
            m_bIfBlinked=false;
            i++;
        }
    }
    else {
        i=0;
        m_bIfBlinked=false;
    }
}

void CEyeDetection::trackRightEye( cv::Mat& mainFrame) {
    cv::Rect window(m_rDetectedEyes[0].x + m_rDetectedEyes[0].width / 4, m_rDetectedEyes[0].y,
                    m_rDetectedEyes[0].width, m_rDetectedEyes[0].height);

    cv::Mat dst(window.width - m_EyeMat.rows / 4 + 1, window.height - m_EyeMat.cols / 4 + 1, CV_32FC1);
    cv::matchTemplate(mainFrame(window), m_EyeMat, dst, cv::TM_SQDIFF_NORMED);
    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);
    if (minval <= 0.2) {
        m_rEyeRect.x = window.x + minloc.x;
        m_rEyeRect.y = window.y + minloc.y;
    }
}

void CEyeDetection::PositionOnDisplay(cv::Rect rect){

    float OkoNaEkranieCmX=1920*(srodekWyliczonyX-(rect.x+rect.width/2))/roznicaX;
    float OkoNaEkranieCmY=1080*(srodekWyliczonyY-(rect.y+rect.height/2))/roznicaY;

    okoNaEkraniePX.x=OkoNaEkranieCmX+960;
    okoNaEkraniePX.y=540-OkoNaEkranieCmY;
}

void CEyeDetection::StworzenieEkranuZInformacja(std::string sInfo) {
    cv::Mat Info(EkranPx.y, EkranPx.x, CV_8UC3, cv::Scalar(0, 0, 0));
    namedWindow("Info", cv::WINDOW_NORMAL);
    setWindowProperty("Info", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    putText(Info, sInfo, cv::Point2f(50, 150), cv::FONT_ITALIC, 1, cv::Scalar(0, 0, 255), 2, 8, false);
    imshow("Info", Info);
    std::this_thread::sleep_for(std::chrono::seconds{3});
}

void CEyeDetection::Calibration(){
    cv::destroyWindow("Info");
    cv::Mat konfiguracja(EkranPx.y, EkranPx.x, CV_8UC3, cv::Scalar(0, 0, 0));
    namedWindow("Konfiguracja", cv::WINDOW_NORMAL);
    setWindowProperty("Konfiguracja", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    circle(konfiguracja, cv::Point(h, j), 50, cv::Scalar(250, 0, 125), cv::FILLED);
    ZebranePunktyX.push_back(m_rEyeRect.x + m_rEyeRect.width / 2);
    ZebranePunktyY.push_back(m_rEyeRect.y + m_rEyeRect.height / 2);
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

void CEyeDetection::PokazanieNaEkranie(Display *display)
{
    Window root_window;
    root_window = XRootWindow(display, 0);
    XSelectInput(display, root_window, KeyReleaseMask);
    XWarpPointer(display, None, root_window, 0,0,1920,1080,okoNaEkraniePX.x, okoNaEkraniePX.y);
    XSync(display, False);

    if(m_bIfBlinked==True)
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

}
