#include "Constants.h" 
#include "video_targets_tracing.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <iostream>


using namespace std;
using namespace cv;
using namespace dnn;


int main() {

	Net net = readNetFromTensorflow(Constants::PB_MODEL, Constants::PB_TXT); //read pre-trained models for Tensorflow
	net.setPreferableBackend(DNN_BACKEND_OPENCV); //ask network to use specific computation backend where it supported, default is:DNN_BACKEND_OPENCV
	net.setPreferableTarget(DNN_TARGET_CPU); //ask network to make computations on specific target device

	VideoCapture cap;
	cap.open(Constants::VIDEO_PATH);

	VideoWriter outFile;
	outFile.open(Constants::OUTPUT_VIDEO, VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(CAP_PROP_FPS), 
		Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true);

	auto out_names = net.getUnconnectedOutLayersNames(); //returns names of layers with unconnected outputs

	// generate colors to paint bounding box for detected objects
	RNG rng(0);
	vector<Scalar> colors;
	for (int i = 0; i < 20; i++)
	{
		colors.push_back(Scalar(rng.uniform(0, 255),
			rng.uniform(0, 255),
			rng.uniform(0, 255)));
	}
	
	Mat frame;
	vector<Rect2d> target;
	Ptr<MultiTracker> multi_tracker = MultiTracker::create();

	for(size_t i = 1; ; i++)
	{
		cap >> frame; //get one frame from the input video

		// wait for cancelling operating
		if ( waitKey(30)== 27 || frame.empty())
			break;

		// process every 10 frames to improve perfonmance
		if (i % 10 == 0)
		{
			target = trackWithModel(net, frame, out_names, Constants::CONFIDENCE_THRESHOLD, Constants::NMS_THRESHOLD, colors);
			if (target.size() == 0)
			{
				i = 0;
				imshow("result", frame);
				outFile << frame; //save to output video file
				continue;
			}

			for (int j = 0; j < target.size(); j++)
				multi_tracker->add(TrackerKCF::create(), frame, Rect2d(target[j]));
		}
		else
		{
			multi_tracker->update(frame);
			for (int j = 0; j < multi_tracker->getObjects().size(); j++)
				rectangle(frame, multi_tracker->getObjects()[j], colors[j]);
		}

		imshow("result", frame);

		outFile << frame;

	}

	return 0;
}


vector<Rect2d> trackWithModel(Net& net, Mat& frame, vector<String> targetname, const float confidence_threshold, const float nms_threshold, vector<Scalar> color)
{
	Mat blobImage = blobFromImage(frame, 1.0, Size(frame.cols, frame.rows), Scalar(0, 0, 0), true, false);
	net.setInput(blobImage);

	vector<Mat> outs;
	net.forward(outs, targetname);

	Mat detection = outs[0];
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	vector<float> confidences;
	vector<Rect2d> boxes;
	vector<Rect2d> draw;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold)
		{
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1)); // object index number
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols; // topleft x coordinate
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows; // topleft y coordinate
			float br_x = detectionMat.at<float>(i, 5) * frame.cols; // bottomright x coordinate
			float br_y = detectionMat.at<float>(i, 6) * frame.rows; // bottomright y coordinate
			Rect2d object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y)); // object rectangle

			confidences.push_back(confidence);
			boxes.push_back(object_box);
		}
	}
	vector<int> indices;
	NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect2d box = boxes[idx];
		rectangle(frame, box, color[i], 2, 8, 0);
		//putText(frame, format("conf %.2f", confidences[idx]),
		//	Point(box.x - 10, box.y - 5), FONT_HERSHEY_SIMPLEX,
		//	0.4, color[i], 1, 8);
		draw.push_back(box);
	}
	
	return draw;
}
