#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

/*
  ==============================================================
  PURPOSE:
    Standardize wafer map images for machine learning and analysis.
  
  WHAT THIS SCRIPT DOES:
    1. Reads grayscale PNGs from "extracted_images/" (values {0,127,254})
    2. Maps pixels → {0,1,2} (background, good, bad)
    3. Applies light cleaning to the defect mask
    4. Pads to square and resizes to 64×64
    5. Creates helper masks:
        - bad_mask: binary mask of defects (0/255)
        - dist_to_center: radial gradient from wafer center (0–255)
        - edge_band: binary mask for outer 10% of wafer (0/255)
    6. Saves results in:
        processed_images/
        processed_masks/bad/
        processed_masks/edgeband/
        processed_aux/dist/
    7. Logs basic stats for traceability
  ============================================================== 
*/

static inline uchar map_to_code(uchar v) {
    if (v < 64)       return 0;
    else if (v < 192) return 1;
    else              return 2;
}

static cv::Mat rebuild_wafer(const cv::Mat& codes01, const cv::Mat& bad_mask) {
    cv::Mat out = cv::Mat::zeros(codes01.size(), CV_8UC1);
    for (int y = 0; y < out.rows; ++y) {
        const uchar* c = codes01.ptr<uchar>(y);
        const uchar* b = bad_mask.ptr<uchar>(y);
        uchar* o = out.ptr<uchar>(y);
        for (int x = 0; x < out.cols; ++x) {
            if (b[x]) o[x] = 2;
            else if (c[x] == 1) o[x] = 1;
            else o[x] = 0;
        }
    }
    return out;
}

static cv::Mat pad_to_square(const cv::Mat& img) {
    int h = img.rows, w = img.cols;
    int m = std::max(h, w);
    cv::Mat sq(m, m, img.type(), cv::Scalar(0));
    int y0 = (m - h) / 2;
    int x0 = (m - w) / 2;
    img.copyTo(sq(cv::Rect(x0, y0, w, h)));
    return sq;
}

static cv::Mat make_dist_map(int size) {
    cv::Mat dist(size, size, CV_8UC1);
    cv::Point2f center(size / 2.0f, size / 2.0f);
    float R = (size - 1) / 2.0f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float d = std::hypot(x - center.x, y - center.y) / R;
            d = std::clamp(d, 0.0f, 1.0f);
            dist.at<uchar>(y, x) = static_cast<uchar>(d * 255);
        }
    }
    return dist;
}

static cv::Mat make_edge_band(int size) {
    cv::Mat edge(size, size, CV_8UC1, cv::Scalar(0));
    cv::Point2f center(size / 2.0f, size / 2.0f);
    float R = (size - 1) / 2.0f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float d = std::hypot(x - center.x, y - center.y) / R;
            if (d >= 0.9f && d <= 1.0f)
                edge.at<uchar>(y, x) = 255;
        }
    }
    return edge;
}

int main() {
    fs::path in_root = "extracted_images";
    fs::path out_root = "processed_images";
    fs::path mask_root = "processed_masks";
    fs::path aux_root = "processed_aux";
    fs::create_directories(out_root);
    fs::create_directories(mask_root / "bad");
    fs::create_directories(mask_root / "edgeband");
    fs::create_directories(aux_root / "dist");
    fs::create_directories("logs");

    std::ofstream logcsv("logs/preprocess_log.csv");
    logcsv << "rel_path,label,w_in,h_in,w_out,h_out,bad_px_in,bad_px_out,ms_total\n";

    const int out_size = 64;
    size_t count = 0;

    for (auto& p : fs::recursive_directory_iterator(in_root)) {
        if (!p.is_regular_file() || p.path().extension() != ".png")
            continue;

        std::string label = p.path().parent_path().filename().string();
        cv::Mat g = cv::imread(p.path().string(), cv::IMREAD_GRAYSCALE);
        if (g.empty()) {
            std::cerr << "WARN: failed to read " << p.path() << "\n";
            continue;
        }

        auto t0 = cv::getTickCount();
        int w_in = g.cols, h_in = g.rows;

        cv::Mat codes(g.size(), CV_8UC1);
        for (int y = 0; y < g.rows; ++y) {
            const uchar* row = g.ptr<uchar>(y);
            uchar* out = codes.ptr<uchar>(y);
            for (int x = 0; x < g.cols; ++x)
                out[x] = map_to_code(row[x]);
        }

        cv::Mat bad_mask = (codes == 2);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
        cv::Mat cleaned;
        cv::morphologyEx(bad_mask, cleaned, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);

        int bad_in = cv::countNonZero(bad_mask);
        int bad_temp = cv::countNonZero(cleaned);
        if (bad_temp < 0.9 * bad_in) {
            cv::morphologyEx(bad_mask, cleaned, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 1);
        }
        cv::Mat bad01; cv::threshold(cleaned, bad01, 127, 1, cv::THRESH_BINARY);

        cv::Mat wafer012 = rebuild_wafer(codes, bad01);

        cv::Mat square = pad_to_square(wafer012);
        cv::Mat resized;
        cv::resize(square, resized, cv::Size(out_size, out_size), 0, 0, cv::INTER_NEAREST);

        cv::Mat resized_bad;
        cv::resize(cleaned, resized_bad, cv::Size(out_size, out_size), 0, 0, cv::INTER_NEAREST);
        cv::Mat dist_map = make_dist_map(out_size);
        cv::Mat edge_band = make_edge_band(out_size);

        cv::Mat pngOut(resized.size(), CV_8UC1);
        for (int y = 0; y < resized.rows; ++y) {
            const uchar* r = resized.ptr<uchar>(y);
            uchar* o = pngOut.ptr<uchar>(y);
            for (int x = 0; x < resized.cols; ++x)
                o[x] = (r[x] == 0 ? 0 : (r[x] == 1 ? 127 : 255));
        }

        fs::path rel = fs::relative(p.path(), in_root);
        fs::path out_img = out_root / rel;
        fs::path bad_path = mask_root / "bad" / rel;
        fs::path dist_path = aux_root / "dist" / rel;
        fs::path edge_path = mask_root / "edgeband" / rel;

        fs::create_directories(out_img.parent_path());
        fs::create_directories(bad_path.parent_path());
        fs::create_directories(dist_path.parent_path());
        fs::create_directories(edge_path.parent_path());

        cv::imwrite(out_img.string(), pngOut);
        cv::imwrite(bad_path.string(), resized_bad);
        cv::imwrite(dist_path.string(), dist_map);
        cv::imwrite(edge_path.string(), edge_band);

        int bad_out = cv::countNonZero(resized_bad);
        double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
        logcsv << rel.generic_string() << "," << label << ","
               << w_in << "," << h_in << ","
               << resized.cols << "," << resized.rows << ","
               << bad_in << "," << bad_out << ","
               << ms << "\n";

        if (++count % 10000 == 0)
            std::cout << "Processed " << count << " images...\n";
    }

    std::cout << "Done. Processed " << count << " images.\n";
    return 0;
}
