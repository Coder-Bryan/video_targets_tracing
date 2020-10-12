#pragma once
#include <string>

namespace Constants {
	
const static std::string PB_MODEL = "E:\\AI\\CV\\Projects\\video_targets_tracing\\Resources\\frozen_inference_graph_mobilenet.pb";
const static std::string PB_TXT   = "E:\\AI\\CV\\Projects\\video_targets_tracing\\Resources\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt";

const static std::string VIDEO_PATH = "E:\\AI\\CV\\Projects\\video_targets_tracing\\Resources\\video1.avi";
const static std::string OUTPUT_VIDEO = "E:\\AI\\CV\\Projects\\video_targets_tracing\\Resources\\video1_output.avi";

const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.5;
}