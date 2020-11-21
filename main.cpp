#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;


Mat frame1;
Mat frame2;
int lthresh = 235;
int rthresh = 255;

void on_Trackbar(int, void*);
int main() {
//    the y file can be opened but the data inside could not be read due to unknown reason,
//    FileStorage fs("param.yml",FileStorage::READ);
//    Mat K[2],C[2],R,T;
//    if(!fs.isOpened())
//    {
//        cout << "Failed to open param.yml!";
//        return 0;
//    }
//
//    fs["K_0"] >> K[0];
//    fs["K_1"] >> K[1];
//    fs["C_0"] >> C[0];
//    fs["C_1"] >> C[1];
//    fs["R"] >> R;
//    fs["T"] >> T;
//    cout << K[1];
//    fs.release();


    VideoCapture capL, capR;
    capL.open("/home/willzhuyx/antiMissile/0.avi");
    capR.open("/home/willzhuyx/antiMissile/1.avi");

    if (!capL.isOpened() )
    {
        printf("can't open video");
    }

    namedWindow("output",WINDOW_AUTOSIZE);

    capL.read(frame1);
    VideoWriter writer("track.avi",VideoWriter::fourcc('M','J','P','G'),20,Size(frame1.cols,frame1.rows));

    while(capR.read(frame1))
    {
        Mat frame_C = frame1.clone();
        blur(frame1,frame1,Size(3,3));
        cvtColor(frame1,frame1,COLOR_BGR2GRAY);

        namedWindow("output",WINDOW_NORMAL);

        createTrackbar("left thresh", "output",&lthresh,255, on_Trackbar, &frame1 );
        createTrackbar("right thresh", "output",&rthresh,255, on_Trackbar, &frame1 );
        on_Trackbar(0,0);

        Canny(frame1,frame1,50,150);

 //       Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(-1,-1));
//        morphologyEx(frame1, frame1,MORPH_OPEN,kernel);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(frame1, contours, hierarchy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat blank = Mat::zeros(frame1.size(),CV_8UC3);
//
        for (int i = 0;i < contours.size();i++)
        {
            RotatedRect box;

            if (contours[i].size()>= 5)
            {
                box = fitEllipse(contours[i]);

            } else
            {
                continue;
            }

            if (MAX(box.size.width,box.size.height) > 60 ) continue;

//            drawContours(frame_C,contours,i,Scalar(0,255,0),3,8);


            ellipse(frame_C,box,Scalar(50,50,255),3,8);
        }

        imshow("output",frame_C);
        writer << frame_C;
        waitKey(50);

    }



    return 0;
}

void on_Trackbar(int, void* )
{
//   inRange(frame1,Scalar(lthresh,0,0),Scalar(rthresh,255,255),frame1);
    threshold(frame1,frame1,lthresh,rthresh,THRESH_BINARY);
//    imshow("output",frame1);
}
