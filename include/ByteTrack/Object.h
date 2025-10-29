#pragma once

#include <opencv2/opencv.hpp>

namespace byte_track
{
struct Object
{
    cv::Rect2f rect;
    int label = 0;
    float prob = 0.f;

	Object(const cv::Rect2f& _rect, const int& _label, const float& _prob)
		: rect(_rect), label(_label), prob(_prob)
	{
	}
};
}