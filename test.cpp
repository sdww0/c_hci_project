#include <opencv2/opencv.hpp>
#include <vector>
 
using namespace cv;
using namespace std;



int main(){
	VideoCapture cap(0);
    CascadeClassifier cascade = CascadeClassifier("cascade.xml");
    cascade.load("/media/file/windows-Ubuntu/swap/openCV/learn3/cascade.xml");
    int flag = 0;
    int timeF = 10;
    Mat frame;
    while(1){
        flag++;
        cap>>frame;
        Mat frame_gray;
        cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
        vector<Rect> target;
        cascade.detectMultiScale(frame_gray,target,1.15,3,0,Size(250,250),Size(500,500));
        for(size_t i = 0;i<target.size();i++){
            rectangle(frame,target[i],Scalar(255,0,0),2,8,0);
        }
        imshow("frame",frame);
        waitKey(100);
    }




}

