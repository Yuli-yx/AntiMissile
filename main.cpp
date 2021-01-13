#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include "MindVision.hpp"
#include <string>

#define _CAMERA_ON
////uncomment this sentence to debug through videos instead of cameras

using namespace cv;
using namespace std;

cv::Mat K[2], C[2];
cv::Mat R, T;

Mat RO[2],P[2];
Mat Q;
Size imageSize = Size(1024,1080);
Mat mapLx, mapLy,mapRx,mapRy;


Mat frame1;
Mat frame2;

int thresh = 155;
//set exptime
int LCamExpTime = 1500;
int RCamExpTime = 1200;
int maxExpTime = 40000;




Point2f points[2];

////preset the parameters of the camera
////length of the base line
double b = 180;
//normalized focus of the camera
double t=1.4158925644379458e+03;





void setExpTime(int,void*);

void overallDectection(Mat& frame1, Mat& frame2);




void Distort_Transform(Mat& left, Mat& right);
double getDepth(Point2f lpt,Point2f rpt);

Point2f PointsDection(Mat& frame, Point2f lastPoint);

void on_MouseHandle(int event, int x, int y, int flags, void* param);

int main() {
#ifdef _CAMERA_ON
    MindVision mv0("AntiMissile-left", EXAMPLE_PATH"/config/left.Config");
    MindVision mv1("AntiMissile-right", EXAMPLE_PATH"/config/right.Config");
    assert(mv0.open());
    assert(mv1.open());
    // 相机设为软触发模式
    MINDVISION_CHECK_API_THROW(CameraSetTriggerMode(mv0.getCameraHandle(), 1));
    MINDVISION_CHECK_API_THROW(CameraSetTriggerMode(mv1.getCameraHandle(), 1));
    CameraHandle h1,h2;
    h1 = mv0.getCameraHandle();
    h2 = mv1.getCameraHandle();





    //// 清空相机缓冲区
    try {
        cv::Mat src;
        mv0.read(src, nullptr, 100);
        mv1.read(src, nullptr, 100);
    } catch (...) {}



#endif

#ifndef _CAMERA_ON
    VideoCapture capL, capR;
    capL.open("/home/willzhuyx/antiMissile/videos/test1/0.avi");
    capR.open("/home/willzhuyx/antiMissile/videos/test1/1.avi");

    if (!capL.isOpened() )
    {
        printf("can't open video");
    }
#endif

    FileStorage fs("../config/param3.yml",FileStorage::READ);

    if(!fs.isOpened())
    {
        cout << "Failed to open yaml file!";
        return 0;
    }

    fs["K_0"] >> K[0];
    fs["K_1"] >> K[1];
    fs["C_0"] >> C[0];
    fs["C_1"] >> C[1];
    fs["R"] >> R;
    fs["T"] >> T;


    fs.release();

    ////calculate the transform matrix.
    stereoRectify(K[0],C[0],K[1],C[1],imageSize,R,T,RO[0],RO[1],P[0],P[1],Q,CALIB_ZERO_DISPARITY,1);
    initUndistortRectifyMap(K[0], C[0], RO[0], P[0], imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(K[1], C[1], RO[1], P[1], imageSize, CV_32FC1, mapRx, mapRy);

    VideoWriter writer("1.avi",VideoWriter::fourcc('M','J','P','G'),10,imageSize);

#ifdef _CAMERA_ON
    mv0.trigger();
    mv1.trigger();
    mv0.read(frame1);
    mv1.read(frame2);
#endif

//    flip(frame1,frame1,-1);
//    flip(frame2,frame2,-1);

    ////reading the videos
    int q = 1,count = 0;


    while(q != 'q')
    {
#ifndef _CAMERA_ON
        capL.read(frame1);
        capL.read(frame2);
        if (frame1.empty())
            break;
#endif

#ifdef _CAMERA_ON
        mv0.trigger();
        mv1.trigger();


        mv0.read(frame1);
        mv1.read(frame2);
//        flip(frame1,frame1,-1);
//        flip(frame2,frame2,-1);
#endif

        ////distorting the frames
        Distort_Transform(frame1,frame2);

        overallDectection(frame1, frame2);

        double depth = getDepth(points[0],points[1])/1000;
        cout << " depth: " << depth<<endl;

        string dp = to_string(depth) +"m";
        putText(frame1,dp,points[0],FONT_HERSHEY_DUPLEX,1.0,Scalar(150,110,0),1);
        imshow("depth",frame1);
        writer << frame1;

//        imshow("frame1",frame1);
//        imshow("frame2", frame2);

        //// current exp time is acceptable. If needed, using the following trackbar tool to change exp time.
        CameraSetExposureTime(h1,1500);
        CameraSetExposureTime(h2,1500);
//        createTrackbar("exposureTime","frame",&LCamExpTime,maxExpTime,setExpTime,&h1);
//        setExpTime(LCamExpTime,&h1);
//        createTrackbar("exposureTime","frame2",&RCamExpTime,maxExpTime,setExpTime,&h2);
//        setExpTime(RCamExpTime,&h2);
        q = waitKey(10);

        ////press p to pause while dectecting
        if (q == 'p') waitKey(0);

    }



    return 0;
}



void overallDectection(Mat& frame1, Mat& frame2)
{

    ////assign the initial position

    for (int i = 0; i < 2; i++)
    {
        points[i] = Point2f(-1,-1);
    }

    blur(frame1,frame1,Size(3,3));
    blur(frame2,frame2,Size(3,3));

    ////step1: creating mask to do brightness-based detection
    Mat mask1,mask2,masked1,masked2;
    cvtColor(frame1,mask1,COLOR_BGR2GRAY);
    cvtColor(frame2,mask2, COLOR_BGR2GRAY);
    threshold(mask1, mask1,thresh, 255,THRESH_BINARY);
    threshold(mask2, mask2,thresh, 255, THRESH_BINARY);


    int core1 = 5;
    Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(core1,core1), Point(-1,-1));
    morphologyEx(mask1, mask1,MORPH_OPEN,kernel1);
    morphologyEx(mask2, mask2,MORPH_OPEN,kernel1);

    bitwise_and(frame1,frame1,masked1,mask1);
    bitwise_and(frame2,frame2,masked2,mask2);

    ////step2: using hsv color space to detect the object
    Mat hsv1, hsv2;
    cvtColor(masked1,hsv1,COLOR_BGR2HSV);
    cvtColor(masked2,hsv2,COLOR_BGR2HSV);

    inRange(hsv1,Scalar(80,160,250),Scalar(100,220,255),mask1);
    inRange(hsv2,Scalar(80,160,250),Scalar(100,220,255),mask2);

    ////using the following mouse callback function to determine the H S V value of 'hsvMasked1'
//    namedWindow("HSV");
//    Mat hsvMasked;
//    cvtColor(masked1,hsvMasked,COLOR_BGR2HSV);
//    setMouseCallback("HSV",on_MouseHandle,(void*)&hsvMasked);
//    imshow("HSV",hsvMasked);

    int core2 = 29;
    Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(core2,core2), Point(-1,-1));
    Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(-1,-1));
    morphologyEx(mask1,mask1,MORPH_ERODE,kernel3);
    morphologyEx(mask1,mask1,MORPH_DILATE,kernel2);

    Mat kernel4 = getStructuringElement(MORPH_ELLIPSE, Size(1,1), Point(-1,-1));
    Mat kernel5 = getStructuringElement(MORPH_ELLIPSE, Size(30,30), Point(-1,-1));
    morphologyEx(mask2,mask2,MORPH_ERODE,kernel4);
    morphologyEx(mask2,mask2,MORPH_DILATE,kernel5);


    points[0] = PointsDection(mask1, points[0]);
    points[1] = PointsDection(mask2,points[1]);

//    string str1 = to_string(points[0].x) +" "+ to_string(points[0].y);
//    putText(frame1,str1,points[0],FONT_HERSHEY_DUPLEX,1.0,Scalar(150,110,0),1);
//    string str2 = to_string(points[1].x) +" "+ to_string(points[1].y);
//    putText(frame2,str2,points[1],FONT_HERSHEY_DUPLEX,1.0,Scalar(150,110,0),1);
//    cout << points[0].x << " " << points[0].y<<endl;


    int p = waitKey(40);
    if (p == 'p') waitKey(0);

}

void setExpTime(int pos,void* usrdata)
{
    CameraHandle h = *(CameraHandle*)(usrdata);
    CameraSetExposureTime(h,pos);
}

void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    Mat& image = *(Mat*)param;

    switch(event)
    {
        case EVENT_LBUTTONDOWN:
        {
            cout << static_cast<int>(image.at<Vec3b>(y, x)[0]) << ",";
            cout << static_cast<int>(image.at<Vec3b>(y, x)[1]) << ",";
            cout << static_cast<int>(image.at<Vec3b>(y, x)[2]) << endl;
        }
        break;
    }
}

Point2f PointsDection(Mat& frame, Point2f lastPoint) {



    Canny(frame,frame,50,150);


    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(frame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    RotatedRect box;
    int count = 0;
    Point2f validatePoints[contours.size()];
    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() >= 5) {
            box = fitEllipse(contours[i]);
        } else {
            continue;
        }
        int a = MAX(box.size.width, box.size.height);
        if (a < 5 ) continue;
        double ratio = box.size.width / box.size.height;
        if (ratio > 3 || ratio < 0.33) continue;
        validatePoints[count] = box.center;
        count++;

//      drawContours(frame_C,contours,i,Scalar(0,255,0),3,8);
//        ellipse(frame1, box, Scalar(50, 50, 255), 2.3, 8);

    }

 //   imshow("frame",frame1);
    if (count == 1)
    {
        return validatePoints[0];
    }
    else if (count == 0)
    {
        return Point2f(-1,-1);
    }
    else if (count > 1)
    {
        if (lastPoint == Point2f(-1,-1)) return validatePoints[0];
        else
        {
            double distance = 1000;
            Point2f minDistPoint;
            for (int i = 0;i < count;i++)
            {
                double dis = (validatePoints[i].y - lastPoint.y) * (validatePoints[i].y - lastPoint.y) + (validatePoints[i].x-lastPoint.x)*(validatePoints[i].x-lastPoint.x);
                if (dis < distance)
                {
                    distance = dis;
                    minDistPoint = validatePoints[i];
                }
            }
            return minDistPoint;
        }

    }

}


void Distort_Transform(Mat& left, Mat& right)
{
    Mat dst1, dst2;
    undistort(frame1,dst1,K[0],C[0]);
    undistort(frame2,dst2,K[1],C[1]);
    frame1 = dst1;
    frame2 = dst2;
    remap(left,left,mapLx,mapLy,INTER_LINEAR);
    remap(right,right,mapRx,mapRy,INTER_LINEAR);
}

double getDepth(Point2f lpt,Point2f rpt)
{
    return abs(t * b / (rpt.x-lpt.x));
}

