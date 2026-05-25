#include "edgecrafter_inference.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void print_usage(const char *argv0) {
    std::cerr << "Usage: " << argv0
              << " <edgecrafter.onnx> <image> <labels.txt> [--segmentation | --pose] [--threshold <value>] [--output "
                 "<path>]\n"
              << "Examples:\n"
              << "  " << argv0 << " ./ecdet_l.onnx ./image.jpg ./data/coco.names\n"
              << "  " << argv0 << " ./ecseg_l.onnx ./image.jpg ./data/coco.names --segmentation\n"
              << "  " << argv0 << " ./ecpose_l.onnx ./image.jpg ./data/coco.names --pose\n";
}

} // namespace

int main(int argc, const char *argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path image_path = argv[2];
    const std::filesystem::path label_path = argv[3];

    Config config;
    config.resolution = 0;
    std::filesystem::path output_path = "output_image.jpg";

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--segmentation") == 0) {
            config.task_type = TaskType::SEGMENTATION;
        } else if (std::strcmp(argv[i], "--pose") == 0) {
            config.task_type = TaskType::POSE;
        } else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            config.threshold = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        EdgeCrafterInference inference(model_path, label_path, config);

        int orig_h = 0;
        int orig_w = 0;
        std::vector<float> input = inference.preprocess_image(image_path, orig_h, orig_w);
        std::vector<Result> results = inference.infer(input, orig_h, orig_w);

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not load image for drawing: " + image_path.string());
        }

        inference.draw_results(image, results);
        if (const auto saved = inference.save_output_image(image, output_path)) {
            std::cout << "Output image saved to: " << saved->string() << "\n";
        }

        std::cout << "Found " << results.size() << " results above threshold " << config.threshold << "\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto &result = results[i];
            std::string label = std::to_string(result.class_id);
            if (result.class_id >= 0 && static_cast<size_t>(result.class_id) < inference.get_labels().size()) {
                label = inference.get_labels()[static_cast<size_t>(result.class_id)];
            }
            std::cout << i << ": class=" << label << " (" << result.class_id << ") score=" << result.score << " box=["
                      << result.box[0] << ", " << result.box[1] << ", " << result.box[2] << ", " << result.box[3]
                      << "]";
            if (!result.mask.empty()) {
                std::cout << " mask_pixels=" << cv::countNonZero(result.mask);
            }
            if (!result.keypoints.empty()) {
                int visible = 0;
                for (const auto &kp : result.keypoints) {
                    if (kp[2] > config.keypoint_threshold) {
                        ++visible;
                    }
                }
                std::cout << " keypoints_visible=" << visible << "/" << result.keypoints.size();
            }
            std::cout << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
