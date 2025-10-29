#include "ByteTrack/BYTETracker.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>

#include <cstddef>


#define RUN_TESTS 1

#if RUN_TESTS
#include "gtest/gtest.h"
#else
#include <opencv2/opencv.hpp>
#endif

namespace
{
    constexpr double EPS = 1e-2;

    const std::string D_RESULTS_FILE = "detection_results.json";
    const std::string T_RESULTS_FILE = "tracking_results.json";
    const std::string VIDEO_FILE = "palace.mp4";

    // key: track_id, value: rect of tracking object
    using BYTETrackerOut = std::map<size_t, cv::Rect2f>;

    template <typename T>
    T get_data(const boost::property_tree::ptree &pt, const std::string &key)
    {
        T ret;
        if (boost::optional<T> data = pt.get_optional<T>(key))
        {
            ret = data.get();
        }
        else
        {
            throw std::runtime_error("Could not read the data from ptree: [key: " + key + "]");
        }
        return ret;
    }

    std::map<size_t, std::vector<byte_track::Object>> get_inputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, std::vector<byte_track::Object>> inputs_ref;
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto prob = get_data<float>(result, "prob");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(inputs_ref)::iterator itr = inputs_ref.find(frame_id);
            if (itr != inputs_ref.end())
            {
                itr->second.emplace_back(cv::Rect2f(x, y, width, height), 0, prob);
            }
            else
            {
                std::vector<byte_track::Object> v(1, {cv::Rect2f(x, y, width, height), 0, prob});
                inputs_ref.emplace_hint(inputs_ref.end(), frame_id, v);
            }
        }
        return inputs_ref;
    }

    std::map<size_t, BYTETrackerOut> get_outputs_ref(const boost::property_tree::ptree &pt)
    {
        std::map<size_t, BYTETrackerOut> outputs_ref;
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto track_id = get_data<int>(result, "track_id");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");

            decltype(outputs_ref)::iterator itr = outputs_ref.find(frame_id);
            if (itr != outputs_ref.end())
            {
                itr->second.emplace(track_id, cv::Rect2f(x, y, width, height));
            }
            else
            {
                BYTETrackerOut v{
                    {track_id, cv::Rect2f(x, y, width, height)},
                };
                outputs_ref.emplace_hint(outputs_ref.end(), frame_id, v);
            }
        }
        return outputs_ref;
    }
}

#if RUN_TESTS
TEST(ByteTrack, BYTETracker)
{
    boost::property_tree::ptree pt_d_results;
    boost::property_tree::read_json(D_RESULTS_FILE, pt_d_results);

    boost::property_tree::ptree pt_t_results;
    boost::property_tree::read_json(T_RESULTS_FILE, pt_t_results);

    try
    {
        // Get infomation of reference data
        const auto detection_results_name = get_data<std::string>(pt_d_results, "name");
        const auto tracking_results_name = get_data<std::string>(pt_t_results, "name");
        const auto fps = get_data<int>(pt_d_results, "fps");
        const auto track_buffer = get_data<int>(pt_d_results, "track_buffer");

        if (detection_results_name != tracking_results_name)
        {
            throw std::runtime_error("The name of the tests are different: [detection_results_name: " + detection_results_name + 
                                     ", tracking_results_name: " + tracking_results_name + "]");
        }

        // Get input reference data from D_RESULTS_FILE
        const auto inputs_ref = get_inputs_ref(pt_d_results);

        // Get output reference data from T_RESULTS_FILE
        auto outputs_ref = get_outputs_ref(pt_t_results);

        // Test BYTETracker::update()
        byte_track::BYTETracker tracker(fps, track_buffer);
        for (const auto &[frame_id, objects] : inputs_ref)
        {
            const auto outputs = tracker.update(objects);

            // Verify between the reference data and the output of the BYTETracker impl
            EXPECT_EQ(outputs.size(), outputs_ref[frame_id].size());
            for (const auto &outputs_per_frame : outputs)
            {
                const auto &rect = outputs_per_frame->getRect();
                const auto &track_id = outputs_per_frame->getTrackId();
                const auto &ref = outputs_ref[frame_id][track_id];
                EXPECT_NEAR(ref.x, rect.x, EPS);
                EXPECT_NEAR(ref.y, rect.y, EPS);
                EXPECT_NEAR(ref.width, rect.width, EPS);
                EXPECT_NEAR(ref.height, rect.height, EPS);
            }
        }
    }
    catch (const std::exception &e)
    {

        FAIL() << e.what();
    }
}
#else // #if RUN_TESTS
void ShowTracks()
{
    boost::property_tree::ptree pt_d_results;
    boost::property_tree::read_json(D_RESULTS_FILE, pt_d_results);

    boost::property_tree::ptree pt_t_results;
    boost::property_tree::read_json(T_RESULTS_FILE, pt_t_results);

    try
    {
        // Get infomation of reference data
        const auto detection_results_name = get_data<std::string>(pt_d_results, "name");
        const auto tracking_results_name = get_data<std::string>(pt_t_results, "name");
        const auto fps = get_data<int>(pt_d_results, "fps");
        const auto track_buffer = get_data<int>(pt_d_results, "track_buffer");

        if (detection_results_name != tracking_results_name)
        {
            throw std::runtime_error("The name of the tests are different: [detection_results_name: " + detection_results_name +
                ", tracking_results_name: " + tracking_results_name + "]");
        }

        // Get input reference data from D_RESULTS_FILE
        const auto inputs_ref = get_inputs_ref(pt_d_results);

        // Get output reference data from T_RESULTS_FILE
        auto outputs_ref = get_outputs_ref(pt_t_results);

        cv::VideoCapture cap(VIDEO_FILE);
        if (!cap.isOpened())
        {
            std::cerr << "File " << VIDEO_FILE << " not opened!" << std::endl;
            return;
        }
        cv::Mat frame;

        // Test BYTETracker::update()
        byte_track::BYTETracker tracker(fps, track_buffer);
		for (const auto& [frame_id, objects] : inputs_ref)
		{
			const auto outputs = tracker.update(objects);

			cap >> frame;
            if (frame.empty())
            {
                std::cout << "Frame " << cap.get(cv::CAP_PROP_POS_FRAMES) << " is empty!" << std::endl;
                break;
            }

			for (const auto& outputs_per_frame : outputs)
			{
				const auto& rect = outputs_per_frame->getRect();
				const auto& track_id = outputs_per_frame->getTrackId();
				const auto& score = outputs_per_frame->getScore();

				cv::Rect brect(cvRound(rect.x), cvRound(rect.y), cvRound(rect.width), cvRound(rect.height));

				cv::rectangle(frame, brect, cv::Scalar(255, 0, 255), 1);

				std::string label = std::to_string(track_id) + " " + std::to_string(score);
				int baseLine = 0;
				double fontScale = (frame.cols < 2000) ? 0.5 : 0.7;
				int thickness = 1;
				int lineType = cv::LINE_AA;
				int fontFace = cv::FONT_HERSHEY_TRIPLEX;
				cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);

				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					//brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = frame.cols - 1 - brect.x;
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					//brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = frame.rows - 1 - brect.y;
				}

				auto DrawFilledRect = [](cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
				{
					if (alpha)
					{
						const int alpha_1 = 255 - alpha;
						const int nchans = frame.channels();
						int color[3] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]), cv::saturate_cast<int>(cl[2]) };
						for (int y = std::max(0, rect.y); y < std::min(rect.y + rect.height, frame.rows - 1); ++y)
						{
							uchar* ptr = frame.ptr(y) + nchans * rect.x;
							for (int x = std::max(0, rect.x); x < std::min(rect.x + rect.width, frame.cols - 1); ++x)
							{
								for (int i = 0; i < nchans; ++i)
								{
									ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[i]) / 255);
								}
								ptr += nchans;
							}
						}
					}
					else
					{
						cv::rectangle(frame, rect, cl, cv::FILLED);
					}
				};
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
				cv::putText(frame, label, brect.tl(), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, lineType);
			}
            cv::imshow("frame", frame);
			cv::waitKey(40);
		}
        cv::waitKey(0);
    }
    catch (const std::exception& e)
    {

        std::cerr << e.what();
    }
}
#endif


#if RUN_TESTS
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return (RUN_ALL_TESTS());
}
#else
int main(int argc, char** argv)
{
    ShowTracks();
    return 0;
}
#endif
