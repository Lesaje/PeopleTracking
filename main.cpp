#include <opencv2/opencv.hpp>
#include <iostream>

const std::string videoPath = "/home/kuver/Documents/CPP/test.mp4";

struct Detection {
    cv::Rect rect;
    int lifespan;
};

cv::VideoCapture initializeCapture(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error opening video file: " + path);
    }
    return cap;
}

cv::Ptr<cv::BackgroundSubtractor> initializeBackgroundSubtractor() {
    return cv::createBackgroundSubtractorKNN(500, 400, false);
}

void processFrame(const cv::Mat& frame, cv::Ptr<cv::BackgroundSubtractor>& pBackSub,
                  cv::Mat& fgMask, cv::Mat& filteredMask) {
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

    // Apply strong Gaussian blur to reduce noise
    cv::GaussianBlur(grayFrame, grayFrame, cv::Size(21, 21), 0);

    pBackSub->apply(grayFrame, fgMask);

    // Apply aggressive threshold
    cv::threshold(fgMask, fgMask, 250, 255, cv::THRESH_BINARY);

    // Apply morphological operations to remove small noise and fill gaps
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(fgMask, filteredMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_CLOSE, kernel);

    // Additional smoothing
    cv::GaussianBlur(filteredMask, filteredMask, cv::Size(21, 21), 0);
}

Detection findLargestContour(const cv::Mat& mask, const cv::Size& frameSize) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect largestBoundingRect;
    double largestArea = 0;
    const double minArea = 0.01 * frameSize.area(); // Minimum 1% of frame area
    const double maxArea = 0.5 * frameSize.area();  // Maximum 50% of frame area

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > largestArea && area > minArea && area < maxArea) {
            largestArea = area;
            largestBoundingRect = cv::boundingRect(contour);
        }
    }

    return {largestBoundingRect, largestArea > 0 ? 5 : 0}; // Lifespan of 5 if detected, 0 if not
}

void processVideo(cv::VideoCapture& cap, cv::Ptr<cv::BackgroundSubtractor>& pBackSub) {
    cv::Mat frame, fgMask, filteredMask;
    Detection currentDetection = {{}, 0};
    int cooldownPeriod = 0;
    const int maxCooldown = 30; // Adjust based on your video's FPS

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        processFrame(frame, pBackSub, fgMask, filteredMask);

        Detection newDetection = findLargestContour(filteredMask, frame.size());

        currentDetection = newDetection;

        cv::rectangle(frame, currentDetection.rect, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Original with Bounding Box", frame);
        cv::imshow("Filtered Mask", filteredMask);

        if (cv::waitKey(30) == 'q') break;
    }
}

int main() {
    try {
        cv::VideoCapture cap = initializeCapture(videoPath);
        cv::Ptr<cv::BackgroundSubtractor> pBackSub = initializeBackgroundSubtractor();
        processVideo(cap, pBackSub);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}