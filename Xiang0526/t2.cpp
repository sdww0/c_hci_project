#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bits/stdc++.h>
#include <algorithm>
#include <vector>

#define M_PI 3.1415926

using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

double distance(Point p1, Point p2)
{
    return (double)norm(p1 - p2);
}

int main(int argc, char** argv)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN" };
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
    if (!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
    int count = 30;
    while (video.read(frame)) {
        imshow("pre", frame);
        count--;
        if (count == 0) {
            break;
        }
    }
    ok = video.read(frame);
    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);

    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);

    // Display bounding box.
    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
    imshow("Tracking", frame);
    Mat ftrace = Mat::zeros(frame.size(), CV_8UC3);
    ftrace.setTo(0);
    tracker->init(frame, bbox);
    vector<Point> points;
    Point start;
    bool isInit = false;
    count = 50;

    while (video.read(frame))
    {
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
        Point point = Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        points.push_back(point);
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

            circle(ftrace, point, 1, Scalar(255, 255, 255), -1);
            if (!isInit) {
                isInit = true;
                start = Point(point);
            }
        }
        // Display frame.
        imshow("Tracking", frame);

        imshow("point", ftrace);
        // Exit if ESC pressed.
        int k = waitKey(1);

        if (k == 27)
        {
            break;
        }
        if (count != 0) {
            count--;
        }
        if (count == 0 && abs(start.x - point.x) <= 30 && abs(start.y - point.y) <= 30) {
            count = 50;
            ftrace.setTo(0);

            //Polygon
            vector<vector<Point>> points_poly(1);
            approxPolyDP(points, points_poly[0], 5, true);

            vector<double> poly_edge;
            for (int i = 0; i < points_poly[0].size(); i++)
            {
                if (i < points_poly[0].size() - 1)
                {
                    poly_edge.push_back(distance(points_poly[0][i], points_poly[0][i+1]));
                }
                else
                {
                    poly_edge.push_back(distance(points_poly[0][i], points_poly[0][0]));
                }
            }

            double max_edge = *max_element(poly_edge.begin(), poly_edge.end());
            int erase_count = 0;
            for (int i = 0; i < poly_edge.size();)
            {
                cout << "Edge " << i << " portion: " << poly_edge[i] / max_edge << endl;
                if (poly_edge[i] / max_edge < 0.3)
                {
                    points_poly[0].erase(points_poly[0].begin() + i);
                    poly_edge.erase(poly_edge.begin() + i);
                    ++erase_count;
                }
                else
                {
                    ++i;
                }
            }
            cout << erase_count << " edges are erased." << endl;

            //Modify polygon to ellipse
            if (points_poly[0].size() > 8)
            {
                RotatedRect box = fitEllipse(points);
                Point2f* vertices = new cv::Point2f[4];
                box.points(vertices);
                //modify ellipse to circle
                double edge_dist[2];
                edge_dist[0] = sqrt((double)(pow(vertices[0].x - vertices[1].x, 2) + pow(vertices[0].y - vertices[1].y, 2)));
                edge_dist[1] = sqrt((double)(pow(vertices[1].x - vertices[2].x, 2) + pow(vertices[1].y - vertices[2].y, 2)));
                double edge_ratio = edge_dist[0] / edge_dist[1];
                cout << "Edge ratio: " << edge_ratio << endl;
                if (edge_ratio > 0.7 && edge_ratio < 1.3)
                {
                    cout << "This is : Circle!" << endl;
                    Point2f center;
                    float radius;
                    minEnclosingCircle(points, center, radius);
                    circle(ftrace, center, radius, Scalar(0, 0, 255), 4);
                }
                //Do not modify ellipse
                else
                {
                    cout << "This is : Ellipse!" << endl;
                    ellipse(ftrace, box, Scalar(255, 0, 0), 3, LINE_AA);
                }
            }
            //Modify polygon to rectangle
            else if (points_poly[0].size() == 4)
            {
                bool paralg_flag = false;

                double delta_x[2], delta_y[2], angle[2];
                delta_x[0] = (double)(points_poly[0][1].x - points_poly[0][0].x);
                delta_x[1] = (double)(points_poly[0][2].x - points_poly[0][3].x);
                delta_y[0] = (double)(points_poly[0][1].y - points_poly[0][0].y);
                delta_y[1] = (double)(points_poly[0][2].y - points_poly[0][3].y);
                
                for (int i = 0; i < 2; i++)
                {
                    angle[i] = atan2(delta_y[i], delta_x[i]) * 180.0 / M_PI;
                }

                double delta_portion;
                double xmean = (abs(delta_x[0]) + abs(delta_x[1])) * 0.5;
                double ymean = (abs(delta_y[0]) + abs(delta_y[1])) * 0.5;
                if (xmean > ymean)
                {
                    delta_portion = abs(delta_x[0] - delta_x[1]) / max(abs(delta_x[0]), abs(delta_x[1]));
                }
                else
                {
                    delta_portion = abs(delta_y[0] - delta_y[1]) / max(abs(delta_y[0]), abs(delta_y[1]));
                }

                cout << "Delta angle: " << abs(angle[0] - angle[1]) << " Delta portion: " << delta_portion;

                if (abs(angle[0] - angle[1]) < 15 && delta_portion < 0.2)
                {
                    paralg_flag = true;
                }

                double diag[2];
                diag[0] = sqrt((double)(pow(points_poly[0][0].x - points_poly[0][2].x, 2) + pow(points_poly[0][0].y - points_poly[0][2].y, 2)));
                diag[1] = sqrt((double)(pow(points_poly[0][1].x - points_poly[0][3].x, 2) + pow(points_poly[0][1].y - points_poly[0][3].y, 2)));
                double ratio_diag = abs(diag[0] / diag[1]);
                cout << " Ratio diag: " << ratio_diag << endl;

                if (paralg_flag && ratio_diag > 0.6 && ratio_diag < 1.4)
                {
                    RotatedRect rect = minAreaRect(points);
                    cout << "This is : Rectangle!" << endl;
                    Point2f* vertices = new cv::Point2f[4];
                    rect.points(vertices);
                    for (int j = 0; j < 4; j++)
                    {
                        line(ftrace, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 4);
                    }
                }
                //Do not modify polygon
                else
                {
                    cout << "This is polygon with " << points_poly[0].size() << " edges!" << endl;
                    drawContours(ftrace, points_poly, 0, Scalar(0, 255, 255), 2, 8);
                }
            }
            //Do not modify polygon
            else
            {
                cout << "This is polygon with " << points_poly[0].size() << " edges!" << endl;
                drawContours(ftrace, points_poly, 0, Scalar(0, 255, 255), 2, 8);
            }

            points.clear();
            start = Point(point);
        }
    }
}

