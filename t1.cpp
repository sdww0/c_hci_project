#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
 #include <bits/stdc++.h>

using namespace cv;
using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
 
int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
    // vector <string> trackerTypes(types, std::end(types));
 
    // Create a tracker
    string trackerType = trackerTypes[4];
 
    Ptr<Tracker> tracker;
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
    }
    #endif
    // Read video
    VideoCapture video(0);
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }



     
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
    int count = 30;
    while(video.read(frame)){
        imshow("pre",frame);
        count--;
        if(count==0){
            break;
        }
    }
    ok = video.read(frame);
    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);
     
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);
 
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);
    Mat ftrace = Mat::zeros(frame.size(), CV_8UC3);
    ftrace.setTo(0);
    tracker->init(frame, bbox);
    vector<Point> points;
    Point start;
    bool isInit = false;
     count = 50;

    while(video.read(frame))
    {     
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
        Point point = Point(bbox.x+bbox.width/2,bbox.y+bbox.height/2);
        points.push_back(point);
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            


            circle(ftrace,point,1,Scalar(255,255,255),-1);
            if(!isInit){
                isInit = true;
                start = Point(point);
            }
        }
        // Display frame.
        imshow("Tracking", frame);


        
        imshow("point",ftrace);
        // Exit if ESC pressed.
        int k = waitKey(1);
        
        if(k == 27)
        {
            break;
        }
        if(count!=0){
            count--;
        }
        if(count==0&&abs(start.x-point.x)<=10&&abs(start.y-point.y)<=10){
            count = 50;
            ftrace.setTo(0);
            vector<Point> triangle;
            minEnclosingTriangle(points,triangle);
            
            Point2f center;
            float radius;
            minEnclosingCircle(points,center,radius);
            RotatedRect rect = minAreaRect(points);

            vector<double> distances;
            for(int i = 0;i<triangle.size();i++){
                for(int n = i+1;n<triangle.size();n++){
                    distances.push_back(sqrt((triangle[i].x-triangle[n].x)*(triangle[i].x-triangle[n].x)+(triangle[i].y-triangle[n].y)*(triangle[i].y-triangle[n].y)));
                }
            }
            double total = 0.0;
            for(int i = 0;i<distances.size();i++){
                total+=distances[i];
            }
            total/=2.0;
            double triangleArea = sqrt(total);
            for(int i = 0;i<distances.size();i++){
                triangleArea *=sqrt(total-distances[i]);
            }
            double circleArea = M_PI*radius*radius;
            double rectArea = rect.boundingRect2f().area();
            if(triangleArea<=circleArea&&triangleArea<=rectArea){
                cout<<"this is :triangle!"<<endl;
                for(int i = 0;i<triangle.size();i++){
                    for(int n = i+1;n<triangle.size();n++){
                        line(ftrace,triangle[i],triangle[n],Scalar(255,0,0),4);
                    }
                }
            }else if(circleArea<=triangleArea&&circleArea<=rectArea){
                cout<<"this is : circle!"<<endl;
                circle(ftrace,center,radius,Scalar(0,0,255),4);
            }else {
                cout<<"this is : Rectangle"<<endl;
                Point2f* vertices = new cv::Point2f[4];
                rect.points(vertices);
                for (int j = 0; j < 4; j++)
                {
                    line(ftrace, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0),4);
                }
            }
            points.clear();
            start = Point(point);
        }
 
    }
}