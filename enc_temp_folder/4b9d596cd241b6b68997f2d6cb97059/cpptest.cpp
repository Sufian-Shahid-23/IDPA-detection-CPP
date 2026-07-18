// =============================================================================
// IDPA Target Detection — C++ / OpenCV
// =============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>

// =============================================================================
// DATA TYPES
// =============================================================================
struct DetectedObject
{
    std::string type;
    int index;
    double area;
};

using Contour = std::vector<cv::Point>;
using ContourVec = std::vector<Contour>;
using ObjVec = std::vector<DetectedObject>;

enum class ThreshMode
{
    DESKEW,   // C=3, kernel 3x3 — used inside deskew pipeline
    DETECTION // C=2, kernel 2x2 — used for final shape detection in main
};

// =============================================================================
// MATH HELPERS (Global because both detection and deskewing use them)
// =============================================================================
static double ptNorm(cv::Point2f a, cv::Point2f b)
{
    double dx = b.x - a.x, dy = b.y - a.y;
    return std::sqrt(dx * dx + dy * dy);
}

static double vecNorm(double dx, double dy)
{
    return std::sqrt(dx * dx + dy * dy);
}

// Signed area of a polygon (shoelace formula) - sign indicates winding direction
double signedArea(const Contour& c)
{
    double a = 0;
    for (size_t i = 0; i < c.size(); i++)
    {
        cv::Point2f p1 = c[i];
        cv::Point2f p2 = c[(i + 1) % c.size()];
        a += (p1.x * p2.y - p2.x * p1.y);
    }
    return a / 2.0;
}

// ── Savitzky-Golay smoothing on contour coordinates ──────────────────
Contour savitzkyGolayContour(const Contour& cnt,
    int window_size = 9,
    int poly_order = 2)
{
    int n = (int)cnt.size();

    if (window_size % 2 == 0)
        window_size++;
    window_size = std::max(window_size, poly_order + 2);
    if (window_size % 2 == 0)
        window_size++;

    if (n < window_size)
    {
        printf("[SG] contour too small (%d pts) for window %d, skipping\n",
            n, window_size);
        return cnt;
    }

    int half = window_size / 2;
    int m = poly_order + 1;

    std::vector<std::vector<double>> A(window_size,
        std::vector<double>(m, 0.0));
    for (int i = 0; i < window_size; i++)
    {
        double x = i - half;
        double xp = 1.0;
        for (int j = 0; j < m; j++)
        {
            A[i][j] = xp;
            xp *= x;
        }
    }

    std::vector<std::vector<double>> AtA(m, std::vector<double>(m, 0.0));
    for (int j = 0; j < m; j++)
        for (int k = 0; k < m; k++)
            for (int i = 0; i < window_size; i++)
                AtA[j][k] += A[i][j] * A[i][k];

    std::vector<std::vector<double>> aug(m, std::vector<double>(2 * m, 0.0));
    for (int j = 0; j < m; j++)
    {
        for (int k = 0; k < m; k++)
            aug[j][k] = AtA[j][k];
        aug[j][j + m] = 1.0;
    }
    for (int col = 0; col < m; col++)
    {
        int pivot = col;
        for (int row = col + 1; row < m; row++)
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col]))
                pivot = row;
        std::swap(aug[col], aug[pivot]);

        double div = aug[col][col];
        if (std::abs(div) < 1e-12)
        {
            printf("[SG] singular matrix, returning original\n");
            return cnt;
        }
        for (int k = 0; k < 2 * m; k++)
            aug[col][k] /= div;
        for (int row = 0; row < m; row++)
        {
            if (row == col)
                continue;
            double f = aug[row][col];
            for (int k = 0; k < 2 * m; k++)
                aug[row][k] -= f * aug[col][k];
        }
    }

    std::vector<double> sg_coeffs(window_size, 0.0);
    for (int i = 0; i < window_size; i++)
        for (int j = 0; j < m; j++)
            sg_coeffs[i] += aug[0][j + m] * A[i][j];

    double coeff_sum = 0.0;
    for (double c : sg_coeffs)
        coeff_sum += c;
    printf("[SG] coefficient sum=%.6f (should be 1.000000)\n", coeff_sum);
    if (std::abs(coeff_sum - 1.0) > 0.01)
    {
        printf("[SG] ERROR: bad coefficients, returning original contour\n");
        return cnt;
    }

    cv::Rect bb = cv::boundingRect(cnt);
    double max_allowed = std::max(bb.width, bb.height) * 0.5;

    Contour result(n);
    for (int i = 0; i < n; i++)
    {
        double sx = 0.0, sy = 0.0;
        for (int j = 0; j < window_size; j++)
        {
            int idx = (i - half + j + n) % n;
            sx += sg_coeffs[j] * cnt[idx].x;
            sy += sg_coeffs[j] * cnt[idx].y;
        }

        double dx = sx - cnt[i].x;
        double dy = sy - cnt[i].y;
        double move = std::sqrt(dx * dx + dy * dy);
        if (move > max_allowed)
        {
            printf("[SG] point %d moved %.1fpx (max=%.1fpx) — aborting\n",
                i, move, max_allowed);
            return cnt;
        }

        result[i] = cv::Point(cvRound(sx), cvRound(sy));
    }

    printf("[SG] SG filter done, output %d pts\n", (int)result.size());
    return result;
}

Contour savitzkyGolayMultiPass(const Contour& cnt,
    int window_size,
    int poly_order = 2,
    int max_passes = 3)
{
    int window = window_size;
    if (window % 2 == 0)
        window++;
    int cap = std::max(5, (int)cnt.size() / 5);
    if (cap % 2 == 0)
        cap--;
    window = std::min(window, cap);
    if (window % 2 == 0)
        window--;
    window = std::max(window, poly_order + 2);
    if (window % 2 == 0)
        window++;

    printf("[SG] multi-pass: pts=%d window=%d order=%d max_passes=%d\n",
        (int)cnt.size(), window, poly_order, max_passes);

    cv::Rect input_bb = cv::boundingRect(cnt);
    double max_allowed_move = std::max(input_bb.width,
        input_bb.height) *
        0.05;

    Contour result = cnt;
    for (int p = 0; p < max_passes; p++)
    {
        Contour next = savitzkyGolayContour(result, window, poly_order);

        double max_move = 0.0;
        for (int i = 0; i < (int)result.size(); i++)
        {
            double dx = next[i].x - result[i].x;
            double dy = next[i].y - result[i].y;
            max_move = std::max(max_move, std::sqrt(dx * dx + dy * dy));
        }

        printf("[SG]   pass %d done, max_point_move=%.2fpx (limit=%.1fpx)\n",
            p + 1, max_move, max_allowed_move);

        if (max_move > max_allowed_move)
        {
            printf("[SG]   DIVERGENCE detected at pass %d, "
                "reverting to previous result\n",
                p + 1);
            break;
        }

        result = next;

        if (max_move < 0.5)
        {
            printf("[SG]   converged at pass %d\n", p + 1);
            break;
        }
    }
    return result;
}

int adaptiveSGWindow(const Contour& cnt)
{
    double perim = cv::arcLength(cnt, true);
    int pts = (int)cnt.size();
    double density = (pts / perim) * 100.0;

    printf("[SG] density=%.2f pts/100px (pts=%d perim=%.0f)\n",
        density, pts, perim);

    int w;
    if (density > 20.0)
        w = 11;
    else if (density > 15.0)
        w = 11;
    else if (density > 8.0)
        w = 9;
    else if (density > 4.0)
        w = 7;
    else
        w = 5;

    if (w % 2 == 0)
        w++;

    int cap = std::max(5, pts / 5);
    if (cap % 2 == 0)
        cap--;
    w = std::min(w, cap);
    if (w % 2 == 0)
        w--;
    w = std::max(w, 5);

    printf("[SG] adaptive window=%d\n", w);
    return w;
}

// ── 2D Vector Median Filter: True Geometric Defect Repair ──────────────────
auto repairByVectorMedian = [](const Contour& cnt, int window_size) -> Contour
    {
        if (cnt.empty() || window_size <= 1) return cnt;
        if (window_size % 2 == 0) window_size++; // Window size must be odd

        int n = (int)cnt.size();
        int half_w = window_size / 2;
        Contour filtered_cnt(n);

        // 1. Slide the vector window along the 2D path
        for (int i = 0; i < n; i++) {
            std::vector<cv::Point> window(window_size);
            for (int j = 0; j < window_size; j++) {
                int idx = (i - half_w + j + n) % n; // Seamless circular wrap
                window[j] = cnt[idx];
            }

            // 2. Find the spatial median point within the local neighborhood
            int best_idx = 0;
            double min_total_dist = std::numeric_limits<double>::max();

            for (int u = 0; u < window_size; u++) {
                double total_dist = 0.0;
                for (int v = 0; v < window_size; v++) {
                    if (u == v) continue;
                    double dx = window[u].x - window[v].x;
                    double dy = window[u].y - window[v].y;
                    total_dist += std::sqrt(dx * dx + dy * dy);
                }

                if (total_dist < min_total_dist) {
                    min_total_dist = total_dist;
                    best_idx = u;
                }
            }

            // Assign the actual intact structural point found
            filtered_cnt[i] = window[best_idx];
        }

        // 3. Clean up consecutive duplicate points to keep the contour crisp
        Contour clean_repaired;
        clean_repaired.reserve(filtered_cnt.size());
        for (const auto& pt : filtered_cnt) {
            if (clean_repaired.empty() || clean_repaired.back() != pt) {
                clean_repaired.push_back(pt);
            }
        }
        if (clean_repaired.size() > 1 && clean_repaired.front() == clean_repaired.back()) {
            clean_repaired.pop_back();
        }

        return clean_repaired;
    };
// ── Debug draw helper ────────────────────────────────────────────────────
auto debugDrawContour = [](const Contour& contour,
    const std::string& filename,
    cv::Scalar color = { 0, 255, 0 })
    {
        if (contour.empty())
            return;
        cv::Rect bb = cv::boundingRect(contour);
        int pad = 20;
        bb.x -= pad;
        bb.y -= pad;
        bb.width += pad * 2;
        bb.height += pad * 2;
        bb.x = std::max(0, bb.x);
        bb.y = std::max(0, bb.y);

        Contour shifted;
        shifted.reserve(contour.size());
        for (auto& pt : contour)
            shifted.push_back(cv::Point(pt.x - bb.x, pt.y - bb.y));

        cv::Mat canvas(bb.height, bb.width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::polylines(canvas, ContourVec{ shifted }, true, color, 2);
        cv::imwrite(filename, canvas);
        printf("[DBG] saved %s\n", filename.c_str());
    };

class GridDewarper {
private:
    int grid_w;
    int grid_h;
    int img_w;
    int img_h;

public:
    GridDewarper(int gw, int gh, int iw, int ih)
        : grid_w(gw), grid_h(gh), img_w(iw), img_h(ih) {
    }

    // Perpendicular search to map wobbly edges, ignoring bullet holes
    std::vector<cv::Point2f> sampleAndFilterEdge(const cv::Point2f& p1, const cv::Point2f& p2,
        const cv::Mat& edge_map, int num_samples, int search_range)
    {
        std::vector<cv::Point2f> actual_points;
        actual_points.reserve(num_samples);

        cv::Point2f edge_vec = p2 - p1;
        float len = std::sqrt(edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y);
        cv::Point2f dir = edge_vec * (1.0f / len);
        cv::Point2f normal(-dir.y, dir.x); // Normal search direction (perpendicular)

        std::vector<float> raw_offsets(num_samples, 0.0f);

        for (int i = 0; i < num_samples; i++) {
            float t = (float)i / (num_samples - 1);
            cv::Point2f ideal_pt = p1 + dir * (t * len);

            float best_offset = 0.0f;
            double max_val = 0;

            for (int step = -search_range; step <= search_range; step++) {
                cv::Point2f sample_pt = ideal_pt + normal * (float)step;

                if (sample_pt.x >= 0 && sample_pt.x < img_w && sample_pt.y >= 0 && sample_pt.y < img_h) {
                    float val = edge_map.at<uchar>(cv::Point(sample_pt.x, sample_pt.y));
                    if (val > max_val) {
                        max_val = val;
                        best_offset = (float)step;
                    }
                }
            }
            raw_offsets[i] = best_offset;
        }

        // 1D Median Filtering over offsets to wipe out bullet holes
        std::vector<float> filtered_offsets(num_samples);
        int kernel_size = 5;
        int half_k = kernel_size / 2;

        for (int i = 0; i < num_samples; i++) {
            std::vector<float> window;
            for (int k = -half_k; k <= half_k; k++) {
                int idx = std::clamp(i + k, 0, num_samples - 1);
                window.push_back(raw_offsets[idx]);
            }
            std::nth_element(window.begin(), window.begin() + half_k, window.end());
            filtered_offsets[i] = window[half_k];
        }

        for (int i = 0; i < num_samples; i++) {
            float t = (float)i / (num_samples - 1);
            actual_points.push_back(p1 + dir * (t * len) + normal * filtered_offsets[i]);
        }

        return actual_points;
    }

    // Jacobi Relaxation: Creates a smooth rubber-sheet warp across the image
    void generateWarpMap(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& constraints, cv::Mat& map_x, cv::Mat& map_y) {
        cv::Mat dx_grid = cv::Mat::zeros(grid_h, grid_w, CV_32FC1);
        cv::Mat dy_grid = cv::Mat::zeros(grid_h, grid_w, CV_32FC1);
        cv::Mat fixed_mask = cv::Mat::zeros(grid_h, grid_w, CV_8UC1);

        // Pin borders so the edges of our photo don't tear
        for (int x = 0; x < grid_w; x++) {
            fixed_mask.at<uchar>(0, x) = 1;
            fixed_mask.at<uchar>(grid_h - 1, x) = 1;
        }
        for (int y = 0; y < grid_h; y++) {
            fixed_mask.at<uchar>(y, 0) = 1;
            fixed_mask.at<uchar>(y, grid_w - 1) = 1;
        }

        float cell_w = (float)img_w / (grid_w - 1);
        float cell_h = (float)img_h / (grid_h - 1);

        for (const auto& c : constraints) {
            cv::Point2f ideal = c.first;
            cv::Point2f actual = c.second;

            int gx = std::clamp((int)std::round(ideal.x / cell_w), 0, grid_w - 1);
            int gy = std::clamp((int)std::round(ideal.y / cell_h), 0, grid_h - 1);

            dx_grid.at<float>(gy, gx) = actual.x - ideal.x;
            dy_grid.at<float>(gy, gx) = actual.y - ideal.y;
            fixed_mask.at<uchar>(gy, gx) = 1;
        }

        cv::Mat next_dx = dx_grid.clone();
        cv::Mat next_dy = dy_grid.clone();
        int iterations = 120; // Enough iterations for smooth relaxation tension

        for (int iter = 0; iter < iterations; iter++) {
            for (int y = 1; y < grid_h - 1; y++) {
                for (int x = 1; x < grid_w - 1; x++) {
                    if (fixed_mask.at<uchar>(y, x)) continue;

                    next_dx.at<float>(y, x) = (dx_grid.at<float>(y - 1, x) + dx_grid.at<float>(y + 1, x) +
                        dx_grid.at<float>(y, x - 1) + dx_grid.at<float>(y, x + 1)) * 0.25f;

                    next_dy.at<float>(y, x) = (dy_grid.at<float>(y - 1, x) + dy_grid.at<float>(y + 1, x) +
                        dy_grid.at<float>(y, x - 1) + dy_grid.at<float>(y, x + 1)) * 0.25f;
                }
            }
            next_dx.copyTo(dx_grid);
            next_dy.copyTo(dy_grid);
        }

        cv::Mat full_dx, full_dy;
        cv::resize(dx_grid, full_dx, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);
        cv::resize(dy_grid, full_dy, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

        map_x.create(img_h, img_w, CV_32FC1);
        map_y.create(img_h, img_w, CV_32FC1);

        for (int y = 0; y < img_h; y++) {
            for (int x = 0; x < img_w; x++) {
                map_x.at<float>(y, x) = (float)x + full_dx.at<float>(y, x);
                map_y.at<float>(y, x) = (float)y + full_dy.at<float>(y, x);
            }
        }
    }
};

// =============================================================================
// 1. DETECTION FUNCTION
// =============================================================================
std::pair<ContourVec, ObjVec> detection(const cv::Mat& img, ThreshMode mode = ThreshMode::DETECTION)
{
    ContourVec unique_contours;
    ObjVec detected_objects;

    cv::Mat gray, blurred;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // ── DARK CIRCLE PRE-PASS ─────────────────────────────────────────────────
    cv::Mat gray_preblur;
    cv::GaussianBlur(gray, gray_preblur, cv::Size(5, 5), 0);
    cv::Mat dark_circle_mask;
    cv::threshold(gray_preblur, dark_circle_mask, 0, 255,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    ContourVec dark_blob_cnts;
    cv::findContours(dark_circle_mask, dark_blob_cnts,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double IMG_AREA_PRE = (double)img.rows * img.cols;
    ContourVec saved_dark_circles;
    for (const auto& dc : dark_blob_cnts)
    {
        double a = cv::contourArea(dc);
        if (a < IMG_AREA_PRE * 0.005 || a > IMG_AREA_PRE * 0.40)
            continue;
        Contour dh;
        cv::convexHull(dc, dh);
        double ha = cv::contourArea(dh);
        double sol = (ha > 0) ? a / ha : 0;
        if (sol < 0.80)
            continue;
        cv::Rect dbr = cv::boundingRect(dc);
        double dar = (dbr.height > 0) ? (double)dbr.width / dbr.height : 0;
        if (dar < 0.50 || dar > 2.0)
            continue;
        saved_dark_circles.push_back(dc);
    }

    // ── SMART POLARITY INVERSION ──────────────────────────────────────────────
    cv::Rect center_roi(img.cols / 4, img.rows / 4, img.cols / 2, img.rows / 2);
    double center_mean = cv::mean(gray(center_roi))[0];
    if (center_mean < 100)
    {
        cv::bitwise_not(gray, gray);
    }

    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    int block = static_cast<int>(img.cols * 0.02);
    if (block % 2 == 0)
        block++;
    block = std::max(block, 21);

    int adaptive_C = (mode == ThreshMode::DESKEW) ? 3 : 2;
    int kernel_size = (mode == ThreshMode::DESKEW) ? 3 : 2;

    cv::Mat thresh_light, thresh_healed;
    cv::adaptiveThreshold(blurred, thresh_light, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
        block, adaptive_C);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(thresh_light, thresh_healed, cv::MORPH_CLOSE, k);
    double IMG_DIAG = std::sqrt((double)img.rows * img.rows + (double)img.cols * img.cols);

    cv::Mat thresh_healed_outer = thresh_healed.clone();
    if (center_mean < 80.0)
    {
        cv::Mat gray_orig;
        cv::cvtColor(img, gray_orig, cv::COLOR_BGR2GRAY);
        cv::Mat bin_orig;
        cv::threshold(gray_orig, bin_orig, 0, 255,
            cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat filled = bin_orig.clone();
        cv::Mat padded;
        cv::copyMakeBorder(filled, padded, 1, 1, 1, 1,
            cv::BORDER_CONSTANT, 0);
        cv::floodFill(padded, cv::Point(0, 0), 128);
        cv::Mat filled_crop = padded(
            cv::Rect(1, 1, filled.cols, filled.rows));

        cv::Mat outer_mask = cv::Mat::zeros(bin_orig.size(), CV_8UC1);
        for (int y = 0; y < filled_crop.rows; y++)
        {
            for (int x = 0; x < filled_crop.cols; x++)
            {
                uchar fc = filled_crop.at<uchar>(y, x);
                uchar bc = bin_orig.at<uchar>(y, x);
                if (fc != 128 || bc == 255)
                    outer_mask.at<uchar>(y, x) = 255;
            }
        }

        cv::Mat k_close = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(outer_mask, outer_mask,
            cv::MORPH_CLOSE, k_close);
        cv::Mat k_erode = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(outer_mask, outer_mask, k_erode,
            cv::Point(-1, -1), 2);

        thresh_healed_outer = outer_mask;
    }

    // ── COLOR-ZONE MASK ───────────────────────────────────────────────────────
    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);
    cv::Mat blur_b, blur_r, blur_g;
    cv::GaussianBlur(bgr[0], blur_b, cv::Size(5, 5), 0);
    cv::GaussianBlur(bgr[2], blur_r, cv::Size(5, 5), 0);
    cv::GaussianBlur(bgr[1], blur_g, cv::Size(5, 5), 0);

    cv::Mat red_mask, green_mask, blue_mask, color_mask;
    cv::Mat diff_rb, diff_rg, diff_gr, diff_bg, diff_br;
    cv::subtract(blur_r, blur_b, diff_rb);
    cv::subtract(blur_r, blur_g, diff_rg);
    cv::subtract(blur_g, blur_r, diff_gr);
    cv::subtract(blur_b, blur_g, diff_bg);
    cv::subtract(blur_b, blur_r, diff_br);

    cv::Mat red_mask_b, red_mask_g;
    cv::threshold(diff_rb, red_mask_b, 40, 255, cv::THRESH_BINARY);
    cv::threshold(diff_rg, red_mask_g, 40, 255, cv::THRESH_BINARY);
    cv::bitwise_and(red_mask_b, red_mask_g, red_mask);
    cv::threshold(diff_gr, green_mask, 30, 255, cv::THRESH_BINARY);
    cv::Mat blue_mask_g_th, blue_mask_r_th;
    cv::threshold(diff_bg, blue_mask_g_th, 30, 255, cv::THRESH_BINARY);
    cv::threshold(diff_br, blue_mask_r_th, 30, 255, cv::THRESH_BINARY);
    cv::bitwise_and(blue_mask_g_th, blue_mask_r_th, blue_mask);

    cv::Mat r_bright, g_bright, b_bright;
    cv::threshold(blur_r, r_bright, 100, 255, cv::THRESH_BINARY);
    cv::threshold(blur_g, g_bright, 80, 255, cv::THRESH_BINARY);
    cv::threshold(blur_b, b_bright, 100, 255, cv::THRESH_BINARY);
    cv::bitwise_and(red_mask, r_bright, red_mask);
    cv::bitwise_and(green_mask, g_bright, green_mask);
    cv::bitwise_and(blue_mask, b_bright, blue_mask);

    cv::bitwise_or(red_mask, green_mask, color_mask);
    cv::bitwise_or(color_mask, blue_mask, color_mask);

    cv::Mat ke2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, ke2, cv::Point(-1, -1), 2);
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, ke2, cv::Point(-1, -1), 1);

    cv::Mat color_mask_filtered = cv::Mat::zeros(color_mask.size(), color_mask.type());
    {
        ContourVec cm_contours;
        cv::findContours(color_mask, cm_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double img_area = (double)color_mask.rows * color_mask.cols;
        for (const auto& c : cm_contours)
        {
            double a = cv::contourArea(c);
            if (a < img_area * 0.001)
                continue;
            Contour hull;
            cv::convexHull(c, hull);
            double hull_a = cv::contourArea(hull);
            double solidity = (hull_a > 0) ? a / hull_a : 0.0;
            cv::Rect br = cv::boundingRect(c);
            double fill_ratio = (br.area() > 0) ? a / (double)br.area() : 0.0;
            if (solidity > 0.85 && fill_ratio > 0.35)
            {
                cv::drawContours(color_mask_filtered, ContourVec{ c }, 0, 255, -1);
            }
        }
    }

    if (mode == ThreshMode::DETECTION)
    {
        cv::bitwise_or(thresh_healed, color_mask_filtered, thresh_healed);
    }
    if (mode == ThreshMode::DETECTION)
    {
        for (const auto& dc : saved_dark_circles)
        {
            cv::Rect dbr = cv::boundingRect(dc);
            cv::Mat region = color_mask(dbr);
            double covered = cv::countNonZero(region);
            double blob_area = cv::contourArea(dc);
            if (covered / (blob_area + 1) < 0.20)
            {
                cv::drawContours(thresh_healed, ContourVec{ dc }, 0, 255, -1);
            }
        }
    }

    // ── CONTOUR EXTRACTION ────────────────────────────────────────────────────
    double IMG_AREA_MERGE = (double)img.rows * img.cols;
    double LARGE_THRESH = IMG_AREA_MERGE * 0.05; // corrected typo: 0.05 is 5%, not 0.45 (which is 45%)

    ContourVec contours_outer, contours_inner;
    std::vector<cv::Vec4i> hier_outer, hier_inner;
    cv::findContours(thresh_healed_outer, contours_outer, hier_outer,
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(thresh_healed, contours_inner, hier_inner,
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    ContourVec contours;
    std::vector<cv::Vec4i> hierarchy;
    for (size_t i = 0; i < contours_outer.size(); i++)
    {
        if (cv::contourArea(contours_outer[i]) >= LARGE_THRESH)
        {
            contours.push_back(contours_outer[i]);
            hierarchy.push_back(hier_outer[i]);
        }
    }
    for (size_t i = 0; i < contours_inner.size(); i++)
    {
        if (cv::contourArea(contours_inner[i]) < LARGE_THRESH)
        {
            contours.push_back(contours_inner[i]);
            hierarchy.push_back(hier_inner[i]);
        }
    }

    if (mode == ThreshMode::DETECTION)
    {
        ContourVec color_contours;
        cv::findContours(color_mask, color_contours,
            cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto& cc : color_contours)
            contours.push_back(cc);
    }

    // ── SCALE-RELATIVE THRESHOLDS ─────────────────────────────────────────────
    double IMG_AREA = (double)img.rows * img.cols;
    double MIN_AREA = IMG_AREA * 0.002;
    double PERIM_LARGE = IMG_DIAG * 0.5;
    double SPIKE_MAX_LEN = IMG_DIAG * 0.1;
    int TOLERANCE_PX = static_cast<int>(IMG_DIAG * 0.03);

    std::vector<cv::Rect> seen_boxes;

    for (const auto& cnt : contours)
    {
        double area = cv::contourArea(cnt);
        double perimeter = cv::arcLength(cnt, true);
        if (area < MIN_AREA)
            continue;
        if (area > IMG_AREA * 0.85)
            continue;

        double compactness = (area > 0) ? (perimeter * perimeter) / area : 9999.0;
        if (compactness > 500 && area < IMG_AREA * 0.05)
            continue;

        cv::Rect br = cv::boundingRect(cnt);
        double aspect_ratio = (br.height > 0) ? (double)br.width / br.height : 0.0;
        if (aspect_ratio < 0.15 || aspect_ratio > 6.0)
            continue;

        Contour hull_pts;
        cv::convexHull(cnt, hull_pts);
        double hull_area = cv::contourArea(hull_pts);
        double solidity = (hull_area > 0) ? area / hull_area : 0.0;
        if (solidity < 0.45)
            continue;

        bool is_dup = false;
        for (const auto& sb : seen_boxes)
        {
            if (std::abs(br.x - sb.x) < TOLERANCE_PX &&
                std::abs(br.y - sb.y) < TOLERANCE_PX &&
                std::abs(br.width - sb.width) < TOLERANCE_PX &&
                std::abs(br.height - sb.height) < TOLERANCE_PX)
            {
                is_dup = true;
                break;
            }
        }
        if (is_dup)
            continue;
        seen_boxes.push_back(br);

        // ── Circle test ────────────────────────────────────────────────────────
        bool is_circle = false;
        cv::RotatedRect ell_rect;
        double ew = 0, eh = 0;

        if (cnt.size() >= 5)
        {
            ell_rect = cv::fitEllipse(cnt);
            ew = ell_rect.size.width;
            eh = ell_rect.size.height;
            double circle_ratio = (eh > 0) ? ew / eh : 0.0;

            cv::RotatedRect min_r = cv::minAreaRect(cnt);
            double rect_area = min_r.size.width * min_r.size.height;
            double rect_extent = (rect_area > 0) ? area / rect_area : 0.0;

            double test_eps = 0.01 * cv::arcLength(cnt, true);
            Contour test_approx;
            cv::approxPolyDP(cnt, test_approx, test_eps, true);
            int num_verts = static_cast<int>(test_approx.size());

            if (circle_ratio > 0.65 && circle_ratio < 1.50 &&
                rect_extent >= 0.72 && rect_extent <= 0.82 &&
                num_verts > 8 && solidity > 0.95)
                is_circle = true;
        }

        std::string shape_type;
        Contour contour_to_add;

        if (is_circle)
        {
            double cr = (eh > 0) ? ew / eh : 0.0;
            shape_type = (cr > 0.90 && cr < 1.10) ? "circle" : "ellipse";

            std::vector<cv::Point> smooth;
            cv::ellipse2Poly(
                cv::Point(static_cast<int>(ell_rect.center.x),
                    static_cast<int>(ell_rect.center.y)),
                cv::Size(static_cast<int>(ew / 2), static_cast<int>(eh / 2)),
                static_cast<int>(ell_rect.angle),
                0, 360, 4, smooth);
            contour_to_add = smooth;
        }
        else
        {
            perimeter = cv::arcLength(cnt, true);

            if (perimeter > PERIM_LARGE)
            {

                cv::Mat iso = cv::Mat::zeros(thresh_healed.size(), thresh_healed.type());
                cv::drawContours(iso, ContourVec{ cnt }, 0, 255, -1);

                int ks = static_cast<int>(perimeter * 0.003);
                if (ks % 2 == 0)
                    ks++;
                ks = std::max(3, std::min(15, ks));

                cv::Mat ke = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks, ks));
                cv::Mat cleaned;
                cv::morphologyEx(iso, cleaned, cv::MORPH_OPEN, ke);
                cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, ke);

                ContourVec smooth_cnts;
                cv::findContours(cleaned, smooth_cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


                auto& main_s = *std::max_element(smooth_cnts.begin(), smooth_cnts.end(),
                    [](const Contour& a, const Contour& b)
                    { return cv::contourArea(a) < cv::contourArea(b); });

                double cur_area = cv::contourArea(cnt);
                double largest_so_far = 0;
                for (const auto& uc : unique_contours)
                    largest_so_far = std::max(largest_so_far, cv::contourArea(uc));

                bool is_outer_torso = (cur_area >= largest_so_far);
                if (false //mode == ThreshMode::DETECTION)
                    )
                {
                    printf("[SHAPE ID] area=%.0f largest_so_far=%.0f is_outer_torso=%d type_guess=%s\n",
                        cur_area, largest_so_far, (int)is_outer_torso,
                        is_outer_torso ? "TORSO" : "INNER");
                }

                double eps = 0.01 * cv::arcLength(cnt, true);
                Contour approx;
                cv::approxPolyDP(cnt, approx, eps, true);

                if (!is_outer_torso) {
                    if (mode == ThreshMode::DESKEW)
                    {
                        Contour ch;
                        cv::convexHull(cnt, ch);
                        double eps = 0.01 * cv::arcLength(ch, true);
                        cv::approxPolyDP(ch, contour_to_add, eps, true);
                    }
                    else
                    {
                        // ── DETECTION: robust octagon recovery ────────────────────────
                        Contour ch;
                        cv::convexHull(cnt, ch);
                        double hull_perim = cv::arcLength(ch, true);

                        Contour best_approx;
                        int best_diff = 9999;
                        for (double ef = 0.005; ef <= 0.04; ef += 0.002)
                        {
                            Contour trial;
                            cv::approxPolyDP(ch, trial, ef * hull_perim, true);
                            int diff = std::abs((int)trial.size() - 8);
                            if (diff < best_diff)
                            {
                                best_diff = diff;
                                best_approx = trial;
                            }
                            if (best_diff == 0)
                                break;
                        }

                        auto signed_area = [](const Contour& poly) -> double
                            {
                                double a = 0;
                                int n = (int)poly.size();
                                for (int i = 0; i < n; i++)
                                {
                                    const cv::Point& p1 = poly[i];
                                    const cv::Point& p2 = poly[(i + 1) % n];
                                    a += (double)p1.x * p2.y - (double)p2.x * p1.y;
                                }
                                return a * 0.5;
                            };

                        Contour approx = best_approx;
                        bool changed = true;
                        while (changed && (int)approx.size() > 8)
                        {
                            changed = false;
                            int sz = (int)approx.size();
                            double perim_now = cv::arcLength(approx, true);
                            double side_thresh = perim_now * 0.12;
                            double base_area = std::abs(signed_area(approx));
                            int worst_idx = -1;
                            double worst_score = -1.0;

                            for (int i = 0; i < sz; i++)
                            {
                                cv::Point2f pc = approx[i];
                                cv::Point2f pp = approx[(i - 1 + sz) % sz];
                                cv::Point2f pn = approx[(i + 1) % sz];
                                double lp = ptNorm(pc, pp);
                                double ln = ptNorm(pc, pn);
                                if (lp >= side_thresh || ln >= side_thresh)
                                    continue;
                                Contour trial = approx;
                                trial.erase(trial.begin() + i);
                                double trial_area = std::abs(signed_area(trial));
                                if (trial_area < base_area * 0.998)
                                    continue;
                                double score = 1.0 / (lp + ln + 1e-6);
                                if (score > worst_score)
                                {
                                    worst_score = score;
                                    worst_idx = i;
                                }
                            }
                            if (worst_idx >= 0)
                            {
                                approx.erase(approx.begin() + worst_idx);
                                changed = true;
                            }
                        }

                        if ((int)approx.size() < 6)
                        {
                            cv::approxPolyDP(ch, approx, 0.015 * hull_perim, true);
                        }

                        contour_to_add = approx;
                    }
                }
                else {
                    // Standard path: Spike removal for Torso or standard detection
                    cv::Rect xb = cv::boundingRect(main_s);
                    double x_box = xb.x, y_box = xb.y;
                    double w_box = xb.width, h_box = xb.height;

                    int idx = 0;
                    while (idx < (int)approx.size() && (int)approx.size() > 4)
                    {
                        cv::Point2f pc = approx[idx];
                        cv::Point2f pp = approx[(idx - 1 + (int)approx.size()) % (int)approx.size()];
                        cv::Point2f pn = approx[(idx + 1) % (int)approx.size()];

                        double lp = ptNorm(pc, pp);
                        double ln = ptNorm(pc, pn);

                        if (lp < SPIKE_MAX_LEN && ln < SPIKE_MAX_LEN)
                        {
                            double v1x = pp.x - pc.x, v1y = pp.y - pc.y;
                            double v2x = pn.x - pc.x, v2y = pn.y - pc.y;
                            double dot = v1x * v2x + v1y * v2y;
                            double n1 = vecNorm(v1x, v1y);
                            double n2 = vecNorm(v2x, v2y);
                            double cos_a = std::max(-1.0, std::min(1.0, dot / (n1 * n2 + 1e-6)));
                            double angle = std::acos(cos_a) * 180.0 / CV_PI;

                            bool protect = false;
                            if (is_outer_torso)
                            {
                                bool upper = (pc.y < y_box + h_box * 0.40);
                                bool head = (pc.y < y_box + h_box * 0.25 &&
                                    pc.x > x_box + w_box * 0.35 &&
                                    pc.x < x_box + w_box * 0.65);
                                bool shoulder = (pc.y < y_box + h_box * 0.40 &&
                                    (pc.x < x_box + w_box * 0.30 ||
                                        pc.x > x_box + w_box * 0.70));
                                protect = upper || head || shoulder;
                            }

                            if (angle < 115 && !protect)
                            {
                                approx.erase(approx.begin() + idx);
                                continue;
                            }
                        }
                        idx++;
                    }
                    contour_to_add = approx;
                }
                
            }
            else
            {
                Contour ch;
                cv::convexHull(cnt, ch);
                double eps = 0.006 * cv::arcLength(ch, true);
                cv::approxPolyDP(ch, contour_to_add, eps, true);
            }
        }

        if (contour_to_add.empty())
            continue;

        if (!is_circle)
        {
            double fe = 0.02 * cv::arcLength(contour_to_add, true);
            Contour fa;
            cv::approxPolyDP(contour_to_add, fa, fe, true);
            int fv = static_cast<int>(fa.size());

            if (fv == 3)
                continue;
            else if (fv <= 8)
                shape_type = "polygon (" + std::to_string(fv) + " pts)";
            else
                shape_type = "polygon";
        }

        DetectedObject obj;
        obj.type = shape_type;
        obj.index = static_cast<int>(unique_contours.size());
        obj.area = cv::contourArea(contour_to_add);
        unique_contours.push_back(contour_to_add);
        detected_objects.push_back(obj);
    }

    // ── Remove inner zones (Second largest shape and Head rectangle for deskew pass) ──
    if (detected_objects.size() >= 2)
    {
        // 1. Rank all objects by area (largest to smallest)
        std::vector<int> order(detected_objects.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b)
            { return detected_objects[a].area > detected_objects[b].area; });

        std::vector<int> indices_to_erase;

        // 2. Mark the second largest shape for deletion based strictly on AREA.
        // order[0] is the massive outer torso. order[1] is guaranteed to be the main inner zone.
        indices_to_erase.push_back(order[1]);

        // 3. Find the Head rectangle based on VERTICES.
        // We start looking at i = 2 so we don't accidentally check the Torso or the main inner zone.
        for (size_t i = 2; i < order.size(); i++)
        {
            int idx = order[i];
            const std::string& st = detected_objects[idx].type;

            // Only look for the 4-point rectangle
            bool is_rect = (st.find("rectangle") != std::string::npos || st.find("4 pts") != std::string::npos);

            if (is_rect)
            {
                indices_to_erase.push_back(idx);
            }
        }

        // 4. CRITICAL: Sort the marked indices in DESCENDING order before erasing
        std::sort(indices_to_erase.rbegin(), indices_to_erase.rend());

        // 5. Erase them from back to front to prevent vector shifting issues
        for (int idx : indices_to_erase)
        {
            unique_contours.erase(unique_contours.begin() + idx);
            detected_objects.erase(detected_objects.begin() + idx);
        }

        // 6. Re-number the internal index properties for the remaining objects
        for (int i = 0; i < (int)detected_objects.size(); i++)
        {
            detected_objects[i].index = i;
        }
    }

    return { unique_contours, detected_objects };
}
//----------------------------------------------------------------------
// ── REFINEMENT WARP after initial deskew ─────────────────────────────
//----------------------------------------------------------------------

cv::Mat refineWarpAfterDeskew(const cv::Mat& deskewed_img, bool debug = false)
{
    cv::Mat gray;
    cv::cvtColor(deskewed_img, gray, cv::COLOR_BGR2GRAY);
    int W = deskewed_img.cols;
    int H = deskewed_img.rows;

    // ── Step 1: Find the main outer contour ──────────────────────────
    cv::Mat blur, thresh;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(blur, thresh, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 2);

    // Clean small noise
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, k);
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, k);

    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(thresh, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Pick largest contour that isn't the whole image
    double max_area = 0; int max_i = -1;
    for (int i = 0; i < (int)cnts.size(); i++) {
        double a = cv::contourArea(cnts[i]);
        if (a > max_area && a < W * H * 0.90) {
            max_area = a; max_i = i;
        }
    }
    if (max_i < 0 || max_area < W * H * 0.05) {
        if (debug) cv::imwrite("debug_refine_fail_nocontour.png", thresh);
        return deskewed_img;
    }

    const auto& cnt = cnts[max_i];

    // ── Step 2: Straightness check ───────────────────────────────────
    // Split contour into left/right halves and measure vertical deviation.
    // If already straight enough, skip refinement.
    {
        std::vector<cv::Point> left_pts, right_pts;
        for (const auto& p : cnt) {
            if (p.x < W / 2) left_pts.push_back(p);
            else            right_pts.push_back(p);
        }

        auto verticalDeviation = [](const std::vector<cv::Point>& pts) -> double {
            if (pts.size() < 2) return 0;
            cv::Vec4f l;
            cv::fitLine(pts, l, cv::DIST_L2, 0, 0.01, 0.01);
            // Angle from vertical: if perfectly vertical vx=0,vy=1
            // deviation = angle of vx from 0
            return std::abs(std::atan2(std::abs(l[0]), std::abs(l[1]))
                * 180.0 / CV_PI);
            };

        double left_dev = verticalDeviation(left_pts);
        double right_dev = verticalDeviation(right_pts);

        if (debug)
            printf("[refine] vertical deviation: L=%.2f° R=%.2f°\n",
                left_dev, right_dev);

        // Already straight — skip
        if (left_dev < 1.5 && right_dev < 1.5) {
            if (debug) {
                cv::Mat dbg = deskewed_img.clone();
                cv::putText(dbg,
                    "SKIP: straight L=" + std::to_string(left_dev).substr(0, 4)
                    + " R=" + std::to_string(right_dev).substr(0, 4) + "deg",
                    { 10,30 }, cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0,200,0 }, 2);
                cv::imwrite("debug_refine_skip.png", dbg);
            }
            return deskewed_img;
        }
    }

    // ── Step 3: Split contour into 4 bands and fit one line per side ─
    // Use position-based splitting (not angle):
    // LEFT  band: points where x < 30% of width
    // RIGHT band: points where x > 70% of width
    // TOP   band: points where y < 30% of height
    // BOT   band: points where y > 70% of height
    //
    // This is robust because the deskewed torso always has its
    // left/right sides in the left/right thirds of the image.

    std::vector<cv::Point> left_pts, right_pts, top_pts, bot_pts;
    for (const auto& p : cnt) {
        float rx = (float)p.x / W;
        float ry = (float)p.y / H;
        if (rx < 0.30f) left_pts.push_back(p);
        if (rx > 0.70f) right_pts.push_back(p);
        if (ry < 0.30f) top_pts.push_back(p);
        if (ry > 0.70f) bot_pts.push_back(p);
    }

    // Need enough points on each side
    if (left_pts.size() < 5 || right_pts.size() < 5 ||
        top_pts.size() < 5 || bot_pts.size() < 5) {
        if (debug) {
            printf("[refine] not enough band pts: L=%zu R=%zu T=%zu B=%zu\n",
                left_pts.size(), right_pts.size(),
                top_pts.size(), bot_pts.size());
            cv::imwrite("debug_refine_fail_bands.png", thresh);
        }
        return deskewed_img;
    }

    // ── Step 4: Fit lines with outlier rejection ──────────────────────
    auto fitRobust = [](std::vector<cv::Point> pts,
        float inlier_px) -> cv::Vec4f
        {
            cv::Vec4f line;
            cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
            // 2 iterations of outlier rejection
            for (int iter = 0; iter < 2; iter++) {
                float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
                std::vector<cv::Point> inliers;
                for (const auto& p : pts) {
                    float dx = p.x - x0, dy = p.y - y0;
                    float dist = std::abs(dx * vy - dy * vx);
                    if (dist <= inlier_px) inliers.push_back(p);
                }
                if ((int)inliers.size() < 4) break;
                pts = inliers;
                cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
            }
            return line;
        };

    cv::Vec4f left_line = fitRobust(left_pts, 8.0f);
    cv::Vec4f right_line = fitRobust(right_pts, 8.0f);
    cv::Vec4f top_line = fitRobust(top_pts, 8.0f);
    cv::Vec4f bot_line = fitRobust(bot_pts, 8.0f);

    // ── Step 5: Intersect adjacent lines → 4 corners ─────────────────
    auto intersect = [](cv::Vec4f l1, cv::Vec4f l2) -> cv::Point2f {
        float x1 = l1[2], y1 = l1[3], dx1 = l1[0], dy1 = l1[1];
        float x2 = l2[2], y2 = l2[3], dx2 = l2[0], dy2 = l2[1];
        float denom = dx1 * dy2 - dy1 * dx2;
        if (std::abs(denom) < 1e-6) return { x1,y1 };
        float t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom;
        return { x1 + t * dx1, y1 + t * dy1 };
        };

    cv::Point2f tl = intersect(top_line, left_line);
    cv::Point2f tr = intersect(top_line, right_line);
    cv::Point2f bl = intersect(bot_line, left_line);
    cv::Point2f br = intersect(bot_line, right_line);

    // ── Step 6: Sanity check ─────────────────────────────────────────
    float margin = std::min(W, H) * 0.40f;
    auto inBounds = [&](cv::Point2f p) {
        return p.x > -margin && p.x < W + margin &&
            p.y > -margin && p.y < H + margin;
        };
    // Also check corners make a sensible quadrilateral
    bool tl_ok = tl.x < tr.x && tl.y < bl.y;
    bool br_ok = br.x > bl.x && br.y > tr.y;

    if (!inBounds(tl) || !inBounds(tr) || !inBounds(bl) || !inBounds(br)
        || !tl_ok || !br_ok) {
        if (debug) {
            cv::Mat dbg = deskewed_img.clone();
            cv::putText(dbg, "FAIL: bad corners",
                { 10,30 }, cv::FONT_HERSHEY_SIMPLEX,
                0.7, { 0,0,255 }, 2);
            for (auto& [pt, name] : std::vector<std::pair<cv::Point2f,
                std::string>>{{tl, "TL"}, { tr,"TR" },
                { bl,"BL" }, { br,"BR" }})
            {
                cv::Point ipt(cvRound(pt.x), cvRound(pt.y));
                if (ipt.x >= 0 && ipt.x < W && ipt.y >= 0 && ipt.y < H)
                    cv::circle(dbg, ipt, 10, { 0,0,255 }, -1);
                cv::putText(dbg, name, { ipt.x + 5,ipt.y },
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0,0,255 }, 2);
            }
            cv::imwrite("debug_refine_fail_corners.png", dbg);
        }
        return deskewed_img;
    }

    // ── Step 7: Debug draw ────────────────────────────────────────────
    if (debug)
    {
        cv::Mat dbg = deskewed_img.clone();

        auto drawExt = [&](cv::Vec4f lf, cv::Scalar col,
            const std::string& label) {
                float vx = lf[0], vy = lf[1], x0 = lf[2], y0 = lf[3];
                cv::line(dbg,
                    { (int)(x0 - 2000 * vx),(int)(y0 - 2000 * vy) },
                    { (int)(x0 + 2000 * vx),(int)(y0 + 2000 * vy) },
                    col, 2);
                cv::putText(dbg, label,
                    { (int)(x0 + 10 * vx + 5),(int)(y0 + 10 * vy + 5) },
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
            };
        drawExt(left_line, { 255,  0,  0 }, "LEFT");
        drawExt(right_line, { 0,  0,255 }, "RIGHT");
        drawExt(top_line, { 0,200,  0 }, "TOP");
        drawExt(bot_line, { 0,200,200 }, "BOT");

        struct PtLabel { cv::Point2f p; std::string n; cv::Scalar c; };
        for (auto& [p, name, col] : std::vector<PtLabel>{
            {tl,"TL",{255,0,0}},{tr,"TR",{0,0,255}},
            {bl,"BL",{0,200,0}},{br,"BR",{0,200,200}} })
        {
            cv::Point ip(cvRound(p.x), cvRound(p.y));
            cv::circle(dbg, ip, 10, col, -1);
            cv::putText(dbg, name, { ip.x + 12,ip.y + 5 },
                cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 2);
        }
        cv::imwrite("debug_refine_corners.png", dbg);
        printf("[refine] TL=(%.1f,%.1f) TR=(%.1f,%.1f) "
            "BL=(%.1f,%.1f) BR=(%.1f,%.1f)\n",
            tl.x, tl.y, tr.x, tr.y, bl.x, bl.y, br.x, br.y);
    }

    // ── Step 8: Corrective homography ────────────────────────────────
    float out_l = std::min(tl.x, bl.x);
    float out_r = std::max(tr.x, br.x);
    float out_t = std::min(tl.y, tr.y);
    float out_b = std::max(bl.y, br.y);

    std::vector<cv::Point2f> src = { tl, tr, br, bl };
    std::vector<cv::Point2f> dst = {
        {out_l, out_t}, {out_r, out_t},
        {out_r, out_b}, {out_l, out_b}
    };

    cv::Mat H_mat = cv::getPerspectiveTransform(src, dst);
    cv::Mat refined;
    cv::warpPerspective(deskewed_img, refined, H_mat,
        deskewed_img.size(),
        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    if (debug) cv::imwrite("debug_refine_result.png", refined);
    return refined;
}

//-------------------------------------------------------------------------
//------------- HOUGH PERSPECTIVE RECOVERY---------------------------------
//-------------------------------------------------------------------------

bool hough_perspective_recovery(
    const cv::Mat& img,
    std::vector<cv::Point2f>& corners_out,
    bool debug = false)
{
    // ── METHOD 1: Blob silhouette approach ────────────────────────────────────
    // Find the largest dark/light blob, get its convex hull,
    // fit lines to the 4 dominant sides, intersect for corners.
    // Works even when edges are partially cut off or obscured.
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // Try both polarities — dark target on light bg and light target on dark bg
        std::vector<cv::Mat> thresh_candidates;
        cv::Mat t_otsu, t_inv;
        cv::threshold(gray, t_otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::threshold(gray, t_inv, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        thresh_candidates.push_back(t_otsu);
        thresh_candidates.push_back(t_inv);

        for (auto& thresh : thresh_candidates)
        {
            // Clean up small noise
            cv::Mat ke = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, ke);
            cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, ke);

            // Find largest blob
            ContourVec blobs;
            cv::findContours(thresh, blobs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (blobs.empty())
                continue;

            auto& biggest = *std::max_element(blobs.begin(), blobs.end(),
                [](const Contour& a, const Contour& b)
                { return cv::contourArea(a) < cv::contourArea(b); });

            double blob_area = cv::contourArea(biggest);
            double img_area = (double)img.rows * img.cols;

            // Must be at least 10% of image — rules out small noise blobs
            if (blob_area < img_area * 0.10)
                continue;

            // Get convex hull
            Contour hull;
            cv::convexHull(biggest, hull);
            int hn = (int)hull.size();
            if (hn < 6)
                continue;

            // Compute each hull edge angle
            struct HullEdge
            {
                int i;
                double angle;
                double len;
            };
            std::vector<HullEdge> edges;
            for (int i = 0; i < hn; i++)
            {
                cv::Point2f p1 = hull[i], p2 = hull[(i + 1) % hn];
                double dx = p2.x - p1.x, dy = p2.y - p1.y;
                double len = std::hypot(dx, dy);
                double ang = std::fmod(std::atan2(dy, dx) * 180.0 / CV_PI + 180.0, 180.0);
                edges.push_back({ i, ang, len });
            }

            // Cluster edges into H and V groups only — reject diagonals
            std::vector<cv::Point2f> h0_pts, h1_pts, v0_pts, v1_pts;
            // We'll cluster by splitting hull into top/bottom/left/right
            // using position relative to centroid — more reliable than angle
            // on a skewed hull where diagonal edges look ambiguous

            cv::Moments mu = cv::moments(hull);
            cv::Point2f cen((float)(mu.m10 / (mu.m00 + 1e-9)),
                (float)(mu.m01 / (mu.m00 + 1e-9)));

            // For each edge, classify by which side of centroid its midpoint is on
            // AND by whether the edge is more horizontal or vertical
            struct Side
            {
                std::string label; // "top","bottom","left","right"
                std::vector<cv::Point2f> pts;
            };
            std::map<std::string, std::vector<cv::Point2f>> side_pts;

            for (auto& e : edges)
            {
                cv::Point2f p1 = hull[e.i], p2 = hull[(e.i + 1) % hn];
                cv::Point2f mid = (p1 + p2) * 0.5f;

                // Normalize angle to [0,90]
                double norm_ang = e.angle;
                if (norm_ang > 90.0)
                    norm_ang = 180.0 - norm_ang;

                // Reject diagonals (30-60 deg range)
                if (norm_ang > 25.0 && norm_ang < 65.0)
                    continue;

                // Classify by position AND orientation
                bool is_horiz = (norm_ang <= 25.0);
                if (is_horiz)
                {
                    if (mid.y < cen.y)
                    {
                        side_pts["top"].push_back(p1);
                        side_pts["top"].push_back(p2);
                    }
                    else
                    {
                        side_pts["bottom"].push_back(p1);
                        side_pts["bottom"].push_back(p2);
                    }
                }
                else
                {
                    if (mid.x < cen.x)
                    {
                        side_pts["left"].push_back(p1);
                        side_pts["left"].push_back(p2);
                    }
                    else
                    {
                        side_pts["right"].push_back(p1);
                        side_pts["right"].push_back(p2);
                    }
                }
            }

            // Need all 4 sides
            if (side_pts.count("top") == 0 || side_pts.count("bottom") == 0 ||
                side_pts.count("left") == 0 || side_pts.count("right") == 0)
            {
                if (debug)
                    std::cout << "  [blob] missing sides: found "
                    << side_pts.size() << "/4\n";
                continue;
            }

            // Fit a line to each side
            auto fit_side = [](const std::vector<cv::Point2f>& pts)
                -> std::pair<cv::Point2f, cv::Point2f>
                {
                    cv::Vec4f lf;
                    cv::fitLine(pts, lf, cv::DIST_L2, 0, 0.01, 0.01);
                    return { cv::Point2f(lf[2], lf[3]), cv::Point2f(lf[0], lf[1]) };
                };

            auto [tp, td] = fit_side(side_pts["top"]);
            auto [bp, bd] = fit_side(side_pts["bottom"]);
            auto [lp, ld] = fit_side(side_pts["left"]);
            auto [rp, rd] = fit_side(side_pts["right"]);

            // Intersect
            auto isect = [](cv::Point2f p1, cv::Point2f d1,
                cv::Point2f p2, cv::Point2f d2,
                cv::Point2f& out) -> bool
                {
                    float den = d1.x * d2.y - d1.y * d2.x;
                    if (std::abs(den) < 1e-6f)
                        return false;
                    float t = ((p2.x - p1.x) * d2.y - (p2.y - p1.y) * d2.x) / den;
                    out = p1 + t * d1;
                    return true;
                };

            cv::Point2f TL, TR, BR, BL;
            if (!isect(tp, td, lp, ld, TL) || !isect(tp, td, rp, rd, TR) ||
                !isect(bp, bd, rp, rd, BR) || !isect(bp, bd, lp, ld, BL))
            {
                if (debug)
                    std::cout << "  [blob] intersection failed\n";
                continue;
            }

            // Sanity: corners must be within 30% of image bounds
            float mx = img.cols * 0.30f, my = img.rows * 0.30f;
            bool ok = true;
            for (auto& p : { TL, TR, BR, BL })
                if (p.x < -mx || p.x > img.cols + mx || p.y < -my || p.y > img.rows + my)
                {
                    ok = false;
                    break;
                }
            if (!ok)
            {
                if (debug)
                    std::cout << "  [blob] corners out of bounds\n";
                continue;
            }

            // Check TL-TR width and TL-BL height are reasonable
            float w = std::hypot(TR.x - TL.x, TR.y - TL.y);
            float h = std::hypot(BL.x - TL.x, BL.y - TL.y);
            if (w < img.cols * 0.10f || h < img.rows * 0.10f)
            {
                if (debug)
                    std::cout << "  [blob] degenerate quad w=" << w << " h=" << h << "\n";
                continue;
            }

            if (debug)
                std::cout << "  [blob] TL=" << TL << " TR=" << TR
                << " BR=" << BR << " BL=" << BL << "\n";

            corners_out = { TL, TR, BR, BL };
            return true;
        }
    }

    // ── METHOD 2: HoughLinesP fallback ────────────────────────────────────────
    // Only runs if blob method failed
    {
        cv::Mat gray, edges;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        double med = 0;
        {
            cv::Mat flat = gray.reshape(1, 1);
            cv::Mat sorted_mat;
            cv::sort(flat, sorted_mat, cv::SORT_ASCENDING);
            med = sorted_mat.at<uchar>(sorted_mat.cols / 2);
        }
        cv::Canny(gray, edges,
            std::max(0.0, med * 0.5),
            std::min(255.0, med * 1.5), 3);

        double diag = std::hypot(img.cols, img.rows);
        std::vector<cv::Vec4i> raw_lines;
        cv::HoughLinesP(edges, raw_lines, 1, CV_PI / 180,
            (int)(diag * 0.08), diag * 0.08, diag * 0.03);

        if (raw_lines.size() < 4)
        {
            if (debug)
                std::cout << "  [hough] too few lines\n";
            return false;
        }

        struct HL
        {
            cv::Point2f p1, p2;
            double angle;
        };
        std::vector<HL> lines;
        for (auto& l : raw_lines)
        {
            double dx = l[2] - l[0], dy = l[3] - l[1];
            double ang = std::fmod(std::atan2(dy, dx) * 180.0 / CV_PI + 180.0, 180.0);
            lines.push_back({ {(float)l[0], (float)l[1]}, {(float)l[2], (float)l[3]}, ang });
        }

        // Only keep H and V lines, reject diagonals
        std::vector<cv::Point2f> h_top, h_bot, v_left, v_right;
        double cy = img.rows / 2.0, cx = img.cols / 2.0;

        for (auto& l : lines)
        {
            double norm_ang = l.angle > 90.0 ? 180.0 - l.angle : l.angle;
            if (norm_ang > 25.0 && norm_ang < 65.0)
                continue; // reject diagonal
            cv::Point2f mid = (l.p1 + l.p2) * 0.5f;
            if (norm_ang <= 25.0)
            {
                if (mid.y < cy)
                {
                    h_top.push_back(l.p1);
                    h_top.push_back(l.p2);
                }
                else
                {
                    h_bot.push_back(l.p1);
                    h_bot.push_back(l.p2);
                }
            }
            else
            {
                if (mid.x < cx)
                {
                    v_left.push_back(l.p1);
                    v_left.push_back(l.p2);
                }
                else
                {
                    v_right.push_back(l.p1);
                    v_right.push_back(l.p2);
                }
            }
        }

        if (h_top.empty() || h_bot.empty() || v_left.empty() || v_right.empty())
        {
            if (debug)
                std::cout << "  [hough] missing H/V groups\n";
            return false;
        }

        auto fit_pts = [](const std::vector<cv::Point2f>& pts)
            -> std::pair<cv::Point2f, cv::Point2f>
            {
                cv::Vec4f lf;
                cv::fitLine(pts, lf, cv::DIST_L2, 0, 0.01, 0.01);
                return { cv::Point2f(lf[2], lf[3]), cv::Point2f(lf[0], lf[1]) };
            };

        auto [tp, td] = fit_pts(h_top);
        auto [bp, bd] = fit_pts(h_bot);
        auto [lp, ld] = fit_pts(v_left);
        auto [rp, rd] = fit_pts(v_right);

        auto isect2 = [](cv::Point2f p1, cv::Point2f d1,
            cv::Point2f p2, cv::Point2f d2,
            cv::Point2f& out) -> bool
            {
                float den = d1.x * d2.y - d1.y * d2.x;
                if (std::abs(den) < 1e-6f)
                    return false;
                float t = ((p2.x - p1.x) * d2.y - (p2.y - p1.y) * d2.x) / den;
                out = p1 + t * d1;
                return true;
            };

        cv::Point2f TL, TR, BR, BL;
        if (!isect2(tp, td, lp, ld, TL) || !isect2(tp, td, rp, rd, TR) ||
            !isect2(bp, bd, rp, rd, BR) || !isect2(bp, bd, lp, ld, BL))
            return false;

        float mx = img.cols * 0.30f, my = img.rows * 0.30f;
        for (auto& p : { TL, TR, BR, BL })
            if (p.x < -mx || p.x > img.cols + mx || p.y < -my || p.y > img.rows + my)
                return false;

        float w = std::hypot(TR.x - TL.x, TR.y - TL.y);
        float h = std::hypot(BL.x - TL.x, BL.y - TL.y);
        if (w < img.cols * 0.10f || h < img.rows * 0.10f)
            return false;

        if (debug)
            std::cout << "  [hough2] TL=" << TL << " TR=" << TR
            << " BR=" << BR << " BL=" << BL << "\n";

        corners_out = { TL, TR, BR, BL };
        return true;
    }
}

// =============================================================================
// STAGE D — VERTICAL EDGE WARP CORRECTION
// =============================================================================

namespace
{

    bool fitLine1D(const std::vector<std::pair<double, double>>& pts,
        double& slope, double& intercept)
    {
        int n = (int)pts.size();
        if (n < 2) return false;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (auto& [x, y] : pts) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
        double denom = n * sxx - sx * sx;
        if (std::abs(denom) < 1e-9) return false;
        slope = (n * sxy - sx * sy) / denom;
        intercept = (sy - slope * sx) / n;
        return true;
    }

    void gaussianSmooth1D(std::vector<double>& v, double sigma)
    {
        if (v.empty() || sigma < 0.5) return;
        int r = static_cast<int>(std::ceil(sigma * 2.5));
        std::vector<double> kernel(2 * r + 1);
        double ksum = 0;
        for (int i = -r; i <= r; i++) { kernel[i + r] = std::exp(-0.5 * (i * i) / (sigma * sigma)); ksum += kernel[i + r]; }
        for (auto& k : kernel) k /= ksum;
        std::vector<double> tmp(v.size());
        for (int j = 0; j < (int)v.size(); j++)
        {
            double acc = 0;
            for (int k = -r; k <= r; k++)
            {
                int idx = std::clamp(j + k, 0, (int)v.size() - 1);
                acc += v[idx] * kernel[k + r];
            }
            tmp[j] = acc;
        }
        v = tmp;
    }

    double rmsVec(const std::vector<double>& v)
    {
        if (v.empty()) return 0;
        double acc = 0;
        for (double x : v) acc += x * x;
        return std::sqrt(acc / v.size());
    }

} // namespace

cv::Mat applyVerticalEdgeWarpCorrection(
    const cv::Mat& deskewed_img,
    double rms_threshold_px = 1.5,   // keep — this is the "skip if clean" floor
    int    sample_rows = 60,
    double max_shift_px = 35.0,  // was 18 — raise to trust real distortion
    double smooth_sigma = 4.0)
{
    const int W = deskewed_img.cols;
    const int H = deskewed_img.rows;
    
    // --- Step 1: Vertical-only edge mask via Sobel ---
    cv::Mat gray, blur, sx, sy, absSx, absSy;
    cv::cvtColor(deskewed_img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.2);
    cv::Sobel(blur, sx, CV_32F, 1, 0, 3);
    cv::Sobel(blur, sy, CV_32F, 0, 1, 3);
    cv::convertScaleAbs(sx, absSx);
    cv::convertScaleAbs(sy, absSy);

    cv::Mat vertEdgeMask = cv::Mat::zeros(H, W, CV_8U);
    for (int r = 0; r < H; r++)
    {
        const uchar* px = absSx.ptr<uchar>(r);
        const uchar* py = absSy.ptr<uchar>(r);
        uchar* pm = vertEdgeMask.ptr<uchar>(r);
        for (int c = 0; c < W; c++)
            if (px[c] > 15 && px[c] > py[c] * 0.7f)
                pm[c] = 255;
    }

    // --- Step 2: HoughLinesP on vertical mask ---
    int minLen = std::max(20, (int)(H * 0.04));
    int maxGap = std::max(8, (int)(H * 0.015));
    int thresh = std::max(30, (int)(H * 0.025));

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(vertEdgeMask, lines, 1, CV_PI / 180.0, thresh, minLen, maxGap);

    if (lines.empty())
    {
        std::cout << "  [VertWarp] No Hough lines — skipping.\n";
        return deskewed_img.clone();
    }

    // --- Step 3: Bucket segments — OUTER EDGE ONLY ---
    double leftBound = W * 0.22;
    double rightBound = W * 0.78;

    std::vector<std::pair<double, double>> leftSamples, rightSamples;

    for (const auto& seg : lines)
    {
        double x1 = seg[0], y1 = seg[1], x2 = seg[2], y2 = seg[3];
        double dx = x2 - x1, dy = y2 - y1;
        if (std::abs(dx) > std::abs(dy) * 0.45) continue;

        double meanX = (x1 + x2) / 2.0;
        int steps = std::max(2, (int)(std::abs(dy) / 20));
        for (int s = 0; s <= steps; s++)
        {
            double t = (double)s / steps;
            double sx2 = x1 + t * dx;
            double sy2 = y1 + t * dy;
            if (meanX < leftBound)  leftSamples.push_back({ sy2, sx2 });
            else if (meanX > rightBound) rightSamples.push_back({ sy2, sx2 });
        }
    }

    // DIAGNOSTIC
    {
        double lxMin = 1e9, lxMax = -1e9, rxMin = 1e9, rxMax = -1e9;
        for (auto& [y, x] : leftSamples) { lxMin = std::min(lxMin, x); lxMax = std::max(lxMax, x); }
        for (auto& [y, x] : rightSamples) { rxMin = std::min(rxMin, x); rxMax = std::max(rxMax, x); }
        std::cout << "  [VertWarp] W=" << W << " leftBound=" << leftBound << " rightBound=" << rightBound << "\n";
        std::cout << "  [VertWarp] LeftBucket  X range: " << lxMin << " - " << lxMax << "  count=" << leftSamples.size() << "\n";
        std::cout << "  [VertWarp] RightBucket X range: " << rxMin << " - " << rxMax << "  count=" << rightSamples.size() << "\n";
    }

    std::cout << "  [VertWarp] Left=" << leftSamples.size()
        << " Right=" << rightSamples.size() << " samples\n";

    if ((int)leftSamples.size() < 8 || (int)rightSamples.size() < 8)
    {
        std::cout << "  [VertWarp] Insufficient samples — skipping.\n";
        return deskewed_img.clone();
    }

    // --- Step 4: Extract outermost X per row-band, then fit ideal line ---
    const int FIT_BANDS = 20;
    int bandH = std::max(1, H / FIT_BANDS);

    std::vector<std::pair<double, double>> leftFitPts, rightFitPts;
    std::vector<std::vector<double>> lBandX(FIT_BANDS), rBandX(FIT_BANDS);

    for (auto& [y, x] : leftSamples)
    {
        int b = std::clamp((int)(y / bandH), 0, FIT_BANDS - 1);
        lBandX[b].push_back(x);
    }
    for (auto& [y, x] : rightSamples)
    {
        int b = std::clamp((int)(y / bandH), 0, FIT_BANDS - 1);
        rBandX[b].push_back(x);
    }

    for (int b = 0; b < FIT_BANDS; b++)
    {
        double bandY = (b + 0.5) * bandH;

        if (!lBandX[b].empty())
        {
            std::sort(lBandX[b].begin(), lBandX[b].end());
            int take = std::max(1, (int)(lBandX[b].size() * 0.20));
            double sum = 0;
            for (int k = 0; k < take; k++) sum += lBandX[b][k];
            leftFitPts.push_back({ bandY, sum / take });
        }

        if (!rBandX[b].empty())
        {
            std::sort(rBandX[b].begin(), rBandX[b].end());
            int n = (int)rBandX[b].size();
            int take = std::max(1, (int)(n * 0.20));
            double sum = 0;
            for (int k = n - take; k < n; k++) sum += rBandX[b][k];
            rightFitPts.push_back({ bandY, sum / take });
        }
    }

    std::cout << "  [VertWarp] Fit points — Left=" << leftFitPts.size()
        << " Right=" << rightFitPts.size() << "\n";

    if ((int)leftFitPts.size() < 4 || (int)rightFitPts.size() < 4)
    {
        std::cout << "  [VertWarp] Insufficient fit points — skipping.\n";
        return deskewed_img.clone();
    }

    double leftSlope, leftIntercept, rightSlope, rightIntercept;
    if (!fitLine1D(leftFitPts, leftSlope, leftIntercept) ||
        !fitLine1D(rightFitPts, rightSlope, rightIntercept))
    {
        std::cout << "  [VertWarp] Line fit failed — skipping.\n";
        return deskewed_img.clone();
    }

    std::cout << "  [VertWarp] Left  ideal line: top x=" << leftIntercept
        << " bot x=" << (leftSlope * H + leftIntercept) << "\n";
    std::cout << "  [VertWarp] Right ideal line: top x=" << rightIntercept
        << " bot x=" << (rightSlope * H + rightIntercept) << "\n";

    // --- Step 5: Per-row median actual X vs ideal X → displacement profile ---
    int rowStep = std::max(1, H / sample_rows);
    std::vector<std::vector<double>> leftBins(sample_rows), rightBins(sample_rows);

    for (auto& [y, x] : leftSamples)
    {
        int bin = std::clamp((int)(y / rowStep), 0, sample_rows - 1);
        leftBins[bin].push_back(x);
    }
    for (auto& [y, x] : rightSamples)
    {
        int bin = std::clamp((int)(y / rowStep), 0, sample_rows - 1);
        rightBins[bin].push_back(x);
    }

    std::vector<double> leftDisp(sample_rows, 0.0), rightDisp(sample_rows, 0.0);
    std::vector<bool>   leftHit(sample_rows, false), rightHit(sample_rows, false);

    for (int i = 0; i < sample_rows; i++)
    {
        double rowY = (i + 0.5) * rowStep;

        if (!leftBins[i].empty())
        {
            std::sort(leftBins[i].begin(), leftBins[i].end());
            int take = std::max(1, (int)(leftBins[i].size() * 0.20));  // was 0.30
            double sum = 0;
            for (int k = 0; k < take; k++) sum += leftBins[i][k];
            leftDisp[i] = (sum / take) - (leftSlope * rowY + leftIntercept);
            leftHit[i] = true;
        }

        if (!rightBins[i].empty())
        {
            std::sort(rightBins[i].begin(), rightBins[i].end());
            int n = (int)rightBins[i].size();
            int take = std::max(1, (int)(n * 0.20));  // was 0.30
            double sum = 0;
            for (int k = n - take; k < n; k++) sum += rightBins[i][k];
            rightDisp[i] = (sum / take) - (rightSlope * rowY + rightIntercept);
            rightHit[i] = true;
        }
    }

    // Fill empty bins by interpolating from neighbours
    for (int i = 0; i < sample_rows; i++)
    {
        if (!leftHit[i])
        {
            int prev = i - 1, next = i + 1;
            while (prev >= 0 && !leftHit[prev]) prev--;
            while (next < sample_rows && !leftHit[next]) next++;
            if (prev >= 0 && next < sample_rows)
                leftDisp[i] = (leftDisp[prev] + leftDisp[next]) * 0.5;
            else if (prev >= 0)          leftDisp[i] = leftDisp[prev];
            else if (next < sample_rows) leftDisp[i] = leftDisp[next];
        }
        if (!rightHit[i])
        {
            int prev = i - 1, next = i + 1;
            while (prev >= 0 && !rightHit[prev]) prev--;
            while (next < sample_rows && !rightHit[next]) next++;
            if (prev >= 0 && next < sample_rows)
                rightDisp[i] = (rightDisp[prev] + rightDisp[next]) * 0.5;
            else if (prev >= 0)          rightDisp[i] = rightDisp[prev];
            else if (next < sample_rows) rightDisp[i] = rightDisp[next];
        }
    }

    // --- Step 6: Skip if wobble is below threshold ---
    double leftRMS = rmsVec(leftDisp);
    double rightRMS = rmsVec(rightDisp);
    std::cout << "  [VertWarp] Left RMS=" << leftRMS << "px  Right RMS=" << rightRMS << "px\n";

    if (leftRMS < rms_threshold_px && rightRMS < rms_threshold_px)
    {
        std::cout << "  [VertWarp] Clean image — no correction needed.\n";
        return deskewed_img.clone();
    }

    // --- Step 7: Smooth displacement profiles ---
    gaussianSmooth1D(leftDisp, smooth_sigma);
    gaussianSmooth1D(rightDisp, smooth_sigma);

    // --- Safety clamp ---
    for (int i = 0; i < sample_rows; i++)
    {
        if (std::abs(leftDisp[i]) > 40 || std::abs(rightDisp[i]) > 40)
        {
            std::cout << "  [VertWarp] Correction exceeds clamp — edge detection unreliable, skipping.\n";
            return deskewed_img.clone();
        }
    }

    // --- Step 8: Build full-resolution remap maps ---
    std::vector<double> leftDispFull(H), rightDispFull(H), idealLX(H), idealRX(H);
    for (int row = 0; row < H; row++)
    {
        double rowY = row + 0.5;
        double t = (rowY / rowStep) - 0.5;
        int    i0 = std::clamp((int)std::floor(t), 0, sample_rows - 1);
        int    i1 = std::clamp(i0 + 1, 0, sample_rows - 1);
        double f = t - i0;
        leftDispFull[row] = leftDisp[i0] * (1 - f) + leftDisp[i1] * f;
        rightDispFull[row] = rightDisp[i0] * (1 - f) + rightDisp[i1] * f;
        idealLX[row] = leftSlope * rowY + leftIntercept;
        idealRX[row] = rightSlope * rowY + rightIntercept;
    }

    cv::Mat map_x(H, W, CV_32F);
    cv::Mat map_y(H, W, CV_32F);

    for (int row = 0; row < H; row++)
    {
        float* mx = map_x.ptr<float>(row);
        float* my = map_y.ptr<float>(row);
        double dL = leftDispFull[row];
        double dR = rightDispFull[row];
        double xL = idealLX[row];
        double xR = idealRX[row];
        double span = xR - xL;

        for (int col = 0; col < W; col++)
        {
            double alpha = (span > 1e-3)
                ? std::clamp((col - xL) / span, 0.0, 1.0)
                : 0.5;
            double correction = dL * (1.0 - alpha) + dR * alpha;
            mx[col] = (float)(col + correction);
            my[col] = (float)row;
        }
    }

    // --- Step 9: Apply remap ---
    cv::Mat corrected;
    cv::remap(deskewed_img, corrected, map_x, map_y,
        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    std::cout << "  [VertWarp] Done. Left RMS=" << leftRMS
        << "px  Right RMS=" << rightRMS << "px\n";
    return corrected;
}
// =============================================================================
// 2. DESKEWING FUNCTION
// =============================================================================
// Corrects image rotation and perspective using internal nested helpers.
cv::Mat deskewing(const cv::Mat& img, double angle_threshold = 10.0)
{
    // Local structs and lambdas encompassing all deskew helper functions.
    struct Line
    {
        double a, b, c;
    };

    auto lineFromPts = [](cv::Point2d p1, cv::Point2d p2) -> Line
        {
            double a = p2.y - p1.y;
            double b = p1.x - p2.x;
            double c = -(a * p1.x + b * p1.y);
            return { a, b, c };
        };

    auto intersect = [](Line l1, Line l2, cv::Point2d& out) -> bool
        {
            double det = l1.a * l2.b - l2.a * l1.b;
            if (std::abs(det) < 1e-9)
                return false;
            out.x = (l1.b * l2.c - l2.b * l1.c) / det;
            out.y = (l2.a * l1.c - l1.a * l2.c) / det;
            return true;
        };

    auto get_octagon_corners = [&lineFromPts, &intersect](const Contour& poly, std::vector<cv::Point2f>& corners_out, bool debug = false) -> bool
        {
            if ((int)poly.size() != 8)
            {
                if (debug)
                    std::cerr << "  [octagon] expected 8 pts, got " << poly.size() << "\n";
                return false;
            }

            std::vector<cv::Point2d> pts(8);
            for (int i = 0; i < 8; i++)
                pts[i] = cv::Point2d(poly[i].x, poly[i].y);

            std::vector<double> elen(8);
            for (int i = 0; i < 8; i++)
            {
                auto& p1 = pts[i];
                auto& p2 = pts[(i + 1) % 8];
                elen[i] = std::hypot(p2.x - p1.x, p2.y - p1.y);
            }
            double se = 0, so = 0;
            for (int k = 0; k < 4; k++)
            {
                se += elen[k * 2];
                so += elen[k * 2 + 1];
            }
            int ss = (se > so) ? 0 : 1;

            struct Side
            {
                cv::Point2d p1, p2, mid;
            };
            std::vector<Side> sides(4);
            for (int k = 0; k < 4; k++)
            {
                int i = (ss + k * 2) % 8;
                sides[k].p1 = pts[i];
                sides[k].p2 = pts[(i + 1) % 8];
                sides[k].mid = { (pts[i].x + pts[(i + 1) % 8].x) / 2, (pts[i].y + pts[(i + 1) % 8].y) / 2 };
            }

            std::sort(sides.begin(), sides.end(), [](const Side& a, const Side& b)
                { return a.mid.y < b.mid.y; });
            Side top_s = sides[0];
            Side bottom_s = sides[3];
            std::vector<Side> rem = { sides[1], sides[2] };
            std::sort(rem.begin(), rem.end(), [](const Side& a, const Side& b)
                { return a.mid.x < b.mid.x; });
            Side left_s = rem[0];
            Side right_s = rem[1];

            Line L_top = lineFromPts(top_s.p1, top_s.p2);
            Line L_bottom = lineFromPts(bottom_s.p1, bottom_s.p2);
            Line L_left = lineFromPts(left_s.p1, left_s.p2);
            Line L_right = lineFromPts(right_s.p1, right_s.p2);

            cv::Point2d TL, TR, BR, BL;
            if (!intersect(L_top, L_left, TL) ||
                !intersect(L_top, L_right, TR) ||
                !intersect(L_bottom, L_right, BR) ||
                !intersect(L_bottom, L_left, BL))
            {
                if (debug)
                    std::cerr << "  [octagon] line intersection failed\n";
                return false;
            }

            corners_out = {
                cv::Point2f((float)TL.x, (float)TL.y),
                cv::Point2f((float)TR.x, (float)TR.y),
                cv::Point2f((float)BR.x, (float)BR.y),
                cv::Point2f((float)BL.x, (float)BL.y) };

            if (debug)
                std::cout << "  Corners TL=" << TL << " TR=" << TR << " BR=" << BR << " BL=" << BL << "\n";

            return true;
        };

    auto quad_needs_perspective_fix = [](const std::vector<cv::Point2f>& c, double angle_tol = 2.0, double ratio_tol = 0.02) -> std::pair<bool, double>
        {
            cv::Point2f TL = c[0], TR = c[1], BR = c[2], BL = c[3];

            auto vlen = [](cv::Point2f a, cv::Point2f b)
                { return std::hypot(b.x - a.x, b.y - a.y); };
            double top_len = vlen(TL, TR), bot_len = vlen(BL, BR);
            double left_len = vlen(TL, BL), right_len = vlen(TR, BR);

            double w_ratio = std::min(top_len, bot_len) / std::max(top_len, bot_len + 1e-9);
            double h_ratio = std::min(left_len, right_len) / std::max(left_len, right_len + 1e-9);

            auto angle_at = [](cv::Point2f pp, cv::Point2f pc, cv::Point2f pn)
                {
                    double v1x = pp.x - pc.x, v1y = pp.y - pc.y;
                    double v2x = pn.x - pc.x, v2y = pn.y - pc.y;
                    double dot = v1x * v2x + v1y * v2y;
                    double n = vecNorm(v1x, v1y) * vecNorm(v2x, v2y) + 1e-9;
                    return std::acos(std::max(-1.0, std::min(1.0, dot / n))) * 180.0 / CV_PI;
                };

            double angles[4] = {
                angle_at(BL, TL, TR), angle_at(TL, TR, BR),
                angle_at(TR, BR, BL), angle_at(BR, BL, TL) };
            double max_dev = 0;
            for (double a : angles)
                max_dev = std::max(max_dev, std::abs(a - 90.0));

            double ratio_dev = std::max(1.0 - w_ratio, 1.0 - h_ratio);
            bool needs_fix = (max_dev > angle_tol) || (ratio_dev > ratio_tol);
            double score = max_dev + ratio_dev * 100.0;
            return { needs_fix, score };
        };

    auto expand_quad_to_fit_contour = [](const std::vector<cv::Point2f>& quad, const Contour& target_pts, double padding_ratio = 0.0, double max_scale = 2.5) -> std::pair<std::vector<cv::Point2f>, double>
        {
            cv::Point2d C(0, 0);
            for (const auto& p : quad)
            {
                C.x += p.x;
                C.y += p.y;
            }
            C.x /= 4;
            C.y /= 4;

            std::vector<cv::Point2d> Q(4);
            for (int i = 0; i < 4; i++)
                Q[i] = { quad[i].x, quad[i].y };

            int ei[4][2] = { {0, 1}, {1, 2}, {2, 3}, {3, 0} };
            double required_k = 1.0;

            for (auto& e : ei)
            {
                cv::Point2d p1 = Q[e[0]], p2 = Q[e[1]];
                double dx = p2.x - p1.x, dy = p2.y - p1.y;
                double nx = dy, ny = -dx;
                double nl = std::hypot(nx, ny) + 1e-12;
                nx /= nl;
                ny /= nl;
                if (nx * (C.x - p1.x) + ny * (C.y - p1.y) > 0)
                {
                    nx = -nx;
                    ny = -ny;
                }
                double m = -(nx * (C.x - p1.x) + ny * (C.y - p1.y));
                if (m < 1e-9)
                    continue;

                for (const auto& tp : target_pts)
                {
                    double g = nx * (tp.x - p1.x) + ny * (tp.y - p1.y);
                    double k = 1.0 + g / m;
                    required_k = std::max(required_k, k);
                }
            }

            double k_final = required_k * (1.0 + padding_ratio);
            if (k_final > max_scale)
            {
                std::cout << "  [warn] scale " << k_final << " clamped to " << max_scale << "\n";
                k_final = max_scale;
            }

            std::vector<cv::Point2f> expanded(4);
            for (int i = 0; i < 4; i++)
            {
                expanded[i].x = (float)(C.x + k_final * (Q[i].x - C.x));
                expanded[i].y = (float)(C.y + k_final * (Q[i].y - C.y));
            }
            return { expanded, k_final };
        };

    auto warp_from_corners = [](const cv::Mat& img, const std::vector<cv::Point2f>& src, cv::Mat* M_out = nullptr) -> cv::Mat
        {
            double w_top = std::hypot(src[1].x - src[0].x, src[1].y - src[0].y);
            double w_bot = std::hypot(src[2].x - src[3].x, src[2].y - src[3].y);
            double h_left = std::hypot(src[3].x - src[0].x, src[3].y - src[0].y);
            double h_right = std::hypot(src[2].x - src[1].x, src[2].y - src[1].y);
            int ow = static_cast<int>(std::max(w_top, w_bot));
            int oh = static_cast<int>(std::max(h_left, h_right));
            ow = std::max(ow, 1);
            oh = std::max(oh, 1);

            std::vector<cv::Point2f> dst = {
                {0.f, 0.f}, {(float)(ow - 1), 0.f}, {(float)(ow - 1), (float)(oh - 1)}, {0.f, (float)(oh - 1)} };

            cv::Mat M = cv::getPerspectiveTransform(src, dst);
            if (M_out)
                *M_out = M;
            cv::Mat warped;
            cv::warpPerspective(img, warped, M, { ow, oh });
            return warped;
        };

    auto get_thresh = [](const cv::Mat& image) -> cv::Mat
        {
            cv::Mat g, b, t;
            cv::cvtColor(image, g, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(g, b, cv::Size(5, 5), 0);
            int bk = static_cast<int>(image.cols * 0.02);
            if (bk % 2 == 0)
                bk++;
            bk = std::max(bk, 11);
            cv::adaptiveThreshold(b, t, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                cv::THRESH_BINARY_INV, bk, 3);
            cv::Mat ones = cv::Mat::ones(3, 3, CV_8U);
            cv::morphologyEx(t, t, cv::MORPH_CLOSE, ones);
            return t;
        };

    auto find_torso_polygon = [&get_thresh](const cv::Mat& image, Contour& cnt_out, Contour& approx_out, cv::Rect& bbox_out) -> bool
        {
            int ih = image.rows, iw = image.cols;
            cv::Mat t = get_thresh(image);
            ContourVec conts;
            cv::findContours(t, conts, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            struct Cand
            {
                Contour cnt, approx;
                double area;
                cv::Rect bbox;
            };
            std::vector<Cand> candidates;

            for (const auto& cnt : conts)
            {
                double area = cv::contourArea(cnt);
                if (area < ih * iw * 0.02 || area > ih * iw * 0.80)
                    continue;
                if ((int)cnt.size() < 5)
                    continue;

                Contour hull;
                cv::convexHull(cnt, hull);
                double ha = cv::contourArea(hull);
                double sol = (ha > 0) ? area / ha : 0;
                if (sol < 0.70)
                    continue;

                double eps = 0.02 * cv::arcLength(cnt, true);
                Contour approx;
                cv::approxPolyDP(cnt, approx, eps, true);
                int n = static_cast<int>(approx.size());
                if (n < 5 || n > 16)
                    continue;

                cv::Rect br = cv::boundingRect(cnt);
                if ((double)br.width / br.height < 0.25 || (double)br.width / br.height > 2.0)
                    continue;
                if (br.x < 5 && br.y < 5 && br.width > iw * 0.9)
                    continue;

                candidates.push_back({ cnt, approx, area, br });
            }
            if (candidates.empty())
                return false;

            std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b)
                { return a.area > b.area; });
            cnt_out = candidates[0].cnt;
            approx_out = candidates[0].approx;
            bbox_out = candidates[0].bbox;
            return true;
        };

    // Main logic for deskewing starts here, utilizing the lambdas above.
    cv::Mat corrected = img.clone();
    bool res_needed = false;

    // ════════════════════════════════════════════════
    // STAGES A, B, & C — CONDITIONAL EXECUTION
    // ════════════════════════════════════════════════
    Contour cnt_a, approx_a;
    cv::Rect bbox_a;
    cv::Mat odeskewed;

    // If we CANNOT find the polygon, we skip A, B, and C entirely and just drop down to D
    if (!find_torso_polygon(corrected, cnt_a, approx_a, bbox_a))
    {
        std::cout << "  Deskew: no polygon found, skipping to resolution.\n";
    }
    else
    {
        // ════════════════════════════════════════════════
        // STAGE A — ROTATION
        // ════════════════════════════════════════════════
        std::vector<std::pair<double, double>> edge_angles;
        for (int i = 0; i < (int)approx_a.size(); i++)
        {
            cv::Point2f p1 = approx_a[i];
            cv::Point2f p2 = approx_a[(i + 1) % (int)approx_a.size()];
            double dx = p2.x - p1.x, dy = p2.y - p1.y;
            double len = std::hypot(dx, dy);
            double ang = std::atan2(dy, dx) * 180.0 / CV_PI;
            edge_angles.push_back({ len, ang });
        }
        std::sort(edge_angles.begin(), edge_angles.end(), [](const auto& a, const auto& b)
            { return a.first > b.first; });

        double norm = std::fmod(edge_angles[0].second, 180.0);
        if (norm < 0)
            norm += 180.0;
        if (norm > 90)
            norm -= 180.0;
        double rot = (std::abs(norm) <= 45) ? norm : (norm > 0 ? norm - 90 : norm + 90);

        std::cout << "  Deskew: rotation=" << rot << "°";
        if (std::abs(rot) >= angle_threshold)
        {
            int iw = corrected.cols, ih = corrected.rows;
            cv::Mat M = cv::getRotationMatrix2D({ iw / 2.f, ih / 2.f }, rot, 1.0);
            cv::warpAffine(corrected, corrected, M, { iw, ih }, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
            std::cout << " → rotated";
        }

        // ════════════════════════════════════════════════
        // STAGE B — AUTO ZOOM
        // ════════════════════════════════════════════════
        Contour cnt_z, approx_z;
        cv::Rect bbox_z;
        if (find_torso_polygon(corrected, cnt_z, approx_z, bbox_z))
        {
            int iw = corrected.cols, ih = corrected.rows;
            double area_ratio = (double)(bbox_z.width * bbox_z.height) / (iw * ih);
            std::cout << " | fill=" << area_ratio;

            if (area_ratio < 0.3)
            {
                double SCALE = 1.3;
                int new_w = static_cast<int>(bbox_z.width * SCALE);
                int new_h = static_cast<int>(bbox_z.height * SCALE);
                int pc_x = bbox_z.x + bbox_z.width / 2;
                int pc_y = bbox_z.y + bbox_z.height / 2;
                int PAD = static_cast<int>(std::max(new_w, new_h) * 0.08);

                int x1 = std::max(0, pc_x - new_w / 2 - PAD);
                int y1 = std::max(0, pc_y - new_h / 2 - PAD);
                int x2 = std::min(iw, pc_x + new_w / 2 + PAD);
                int y2 = std::min(ih, pc_y + new_h / 2 + PAD);

                int cw = x2 - x1, ch = y2 - y1;
                if (cw > 0 && ch > 0 &&
                    x1 >= 0 && y1 >= 0 &&
                    x1 + cw <= corrected.cols &&
                    y1 + ch <= corrected.rows)
                {
                    corrected = corrected(cv::Rect(x1, y1, cw, ch)).clone();
                }
                std::cout << " → zoomed to " << (x2 - x1) << "×" << (y2 - y1);
                res_needed = true;
            }
            else
            {
                std::cout << " → no zoom needed";
            }
        }
        else
        {
            std::cout << " | no polygon for zoom";
        }

        // ════════════════════════════════════════════════
        // STAGE C — PERSPECTIVE (convergence loop)
        // ════════════════════════════════════════════════
        const int MAX_ITERS = 4;
        const double SCORE_IMPROVE_MIN = 0.1;
        double prev_score = -1.0;
        int persp_passes = 0;

        std::cout << "\n--- Deskewing Loop Started ---";
        cv::Mat gray_check;
        cv::cvtColor(corrected, gray_check, cv::COLOR_BGR2GRAY);
        cv::Rect center_check(corrected.cols / 4, corrected.rows / 4,
            corrected.cols / 2, corrected.rows / 2);
        double check_mean = cv::mean(gray_check(center_check))[0];
        bool is_dark_bg = (check_mean < 100.0);

        // ── PRE-LOOP: Try Hough FIRST on the clean pre-warp image ─────────────────
        // On dark/angled targets, the contour path on Pass 1 often produces a bad
        // low-score quad and applies a damaging warp before Hough gets a chance.
        // Running Hough once on the untouched image avoids that.
        // ── PRE-LOOP: Try Hough FIRST — only on dark-bg targets ───────────────────
        {

            std::vector<cv::Point2f> hough_corners;
            bool hough_ok = is_dark_bg && hough_perspective_recovery(corrected, hough_corners, true);

            if (hough_ok)
            {
                auto [needs_fix, score] = quad_needs_perspective_fix(hough_corners);
                std::cout << "\nPre-pass [hough on clean image]: Score = "
                    << std::fixed << std::setprecision(3) << score << " ";

                if (needs_fix)
                {

                    // ── CHANGED: compute expanded corners purely from Hough geometry ──
                    // Do NOT use the outer torso contour for expansion — on dark angled
                    // targets it always has stray wall/tape points that blow up the scale.
                    // Instead: find the centroid of the 4 Hough corners and expand by a
                    // fixed 20% margin. This is enough to include the full torso since
                    // Hough lines are fitted to the outermost visible edges of the target.
                    cv::Point2f C = (hough_corners[0] + hough_corners[1] +
                        hough_corners[2] + hough_corners[3]) *
                        0.25f;

                    const float HOUGH_PAD = 1.20f; // 20% outward from the Hough quad
                    std::vector<cv::Point2f> expanded(4);
                    for (int i = 0; i < 4; i++)
                        expanded[i] = C + HOUGH_PAD * (hough_corners[i] - C);

                    double k_used = HOUGH_PAD;

                    // Sanity check
                    bool sane = true;
                    for (const auto& p : expanded)
                    {
                        if (p.x < -corrected.cols * 0.30f ||
                            p.x > corrected.cols * 1.30f ||
                            p.y < -corrected.rows * 0.30f ||
                            p.y > corrected.rows * 1.30f)
                        {
                            sane = false;
                            break;
                        }
                    }

                    if (sane)
                    {
                        cv::Mat candidate = warp_from_corners(corrected, expanded);
                        if (candidate.cols <= corrected.cols * 2 &&
                            candidate.rows <= corrected.rows * 2)
                        {
                            corrected = candidate;
                            res_needed = true;
                            prev_score = score;
                            persp_passes++;
                            std::cout << "-> pre-pass applied (scale=" << k_used << ")";
                        }
                        else
                        {
                            std::cout << "-> warp too large, skipped";
                        }
                    }
                    else
                    {
                        std::cout << "-> corners unsafe, skipped";
                    }
                }
                else
                {
                    std::cout << "-> no fix needed";
                    goto stage_c_done;
                }
            }
            else
            {
                std::cout << "\nPre-pass [hough]: failed to find lines, proceeding to loop";
            }
        }

        for (int iter = 0; iter < MAX_ITERS; iter++)
        {
            std::cout << "\nPass " << (iter + 1) << ": ";

            auto [uc, objs] = detection(corrected, ThreshMode::DESKEW);

            // ── Primary: find 8-pt octagon from detection ─────────────────────
            Contour* target_cnt = nullptr;
            for (int i = 0; i < (int)objs.size(); i++)
            {
                if (objs[i].type.find("8 pts") != std::string::npos)
                {
                    target_cnt = &uc[i];
                    break;
                }
            }
            // Fallback A: closest polygon to 8 pts with sufficient area
            if (!target_cnt)
            {
                double img_area = (double)corrected.rows * corrected.cols;
                for (int i = 0; i < (int)objs.size(); i++)
                {
                    const std::string& t = objs[i].type;
                    bool is_poly = t.find("polygon") != std::string::npos ||
                        t.find("pts") != std::string::npos;
                    if (is_poly && objs[i].area > img_area * 0.05)
                    {
                        if (target_cnt == nullptr || std::abs((int)uc[i].size() - 8) < std::abs((int)target_cnt->size() - 8))
                            target_cnt = &uc[i];
                    }
                }
            }

            std::vector<cv::Point2f> corners;
            bool corners_ok = false;

            // ── Try octagon corner recovery from contour ───────────────────────
            if (target_cnt && !uc.empty())
            {
                corners_ok = get_octagon_corners(*target_cnt, corners);
                if (corners_ok)
                    std::cout << "[contour path] ";
            }

            // ── Fallback B: HoughLinesP directly on image ──────────────────────
            // Fires when: no octagon found, OR octagon found but corners failed.
            // Works on raw edge pixels — immune to contour noise on dark targets.
            // ── Fallback B: HoughLinesP — only on dark-bg targets ─────────────
            if (!corners_ok && is_dark_bg)
            {
                std::cout << "[hough fallback] ";
                corners_ok = hough_perspective_recovery(corrected, corners, false);
            }

            // ── Both methods failed — stop this pass ───────────────────────────
            if (!corners_ok)
            {
                std::cout << "Both contour and Hough methods failed. Stopping.";
                break;
            }

            auto [needs_fix, score] = quad_needs_perspective_fix(corners);
            std::cout << "Score = " << std::fixed << std::setprecision(3) << score << " ";

            if (!needs_fix)
            {
                std::cout << "-> Perfect alignment achieved!";
                break;
            }

            if (prev_score >= 0)
            {
                double improvement = prev_score - score;
                if (improvement < 0)
                {
                    std::cout << "-> Score WORSENED by " << std::abs(improvement) << ". Stopping.";
                    break;
                }
                if (improvement < SCORE_IMPROVE_MIN)
                {
                    std::cout << "-> Improvement plateaued (diff = " << improvement << "). Stopping.";
                    break;
                }
            }

            // ── Expand to cover outer torso then warp ─────────────────────────
            int best_i = 0;
            for (int i = 1; i < (int)objs.size(); i++)
                if (objs[i].area > objs[best_i].area)
                    best_i = i;

            std::vector<cv::Point2f> expanded;
            double k_used;

            // If Hough was used, uc may be empty — warp from raw corners directly
            if (!uc.empty())
            {
                auto [exp2, k2] = expand_quad_to_fit_contour(corners, uc[best_i], 0.03, 2.5);
                expanded = exp2;
                k_used = k2;
            }
            else
            {
                // No contours available — scale corners up by fixed 1.15 margin
                cv::Point2f C = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
                expanded.resize(4);
                for (int i = 0; i < 4; i++)
                {
                    expanded[i] = C + 1.15f * (corners[i] - C);
                }
                k_used = 1.15;
            }

            /* Safety: skip if expanded corners are wildly outside image
            bool sane = true;
            for (const auto& p : expanded) {
                if (p.x < -corrected.cols * 0.25f || p.x > corrected.cols * 1.25f ||
                    p.y < -corrected.rows * 0.25f || p.y > corrected.rows * 1.25f) {
                    sane = false; break;
                }
            }
            if (!sane) { std::cout << "-> Expanded corners unsafe. Stopping."; break; }
            */
            cv::Mat warped_candidate = warp_from_corners(corrected, expanded);
            if (warped_candidate.cols > corrected.cols * 2 ||
                warped_candidate.rows > corrected.rows * 2)
            {
                std::cout << "-> Warp output too large. Stopping.";
                break;
            }

            corrected = warped_candidate;
            res_needed = true;
            prev_score = score;
            persp_passes++;
            std::cout << "-> pass applied (scale=" << k_used << ")";
        }

    stage_c_done:
        //cv::Mat straightened = straightenEdges(corrected, 10);
        //corrected = straightened;   // replace with straightened version

        std::cout << "\n--- Deskewing Finished ...\n";
        // ════════════════════════════════════════════════
        // STAGE D — RESOLUTION NORMALIZATION
        // ════════════════════════════════════════════════
        const int TARGET_W = 850;
        const int TARGET_H = 1550;

        if (corrected.cols != TARGET_W || corrected.rows != TARGET_H)
        {
            int interp = (TARGET_W < corrected.cols) ? cv::INTER_AREA : cv::INTER_CUBIC;
            std::cout << " → resolution normalized from " << corrected.cols << "x" << corrected.rows << "to" << TARGET_W << "×" << TARGET_H;
            cv::resize(corrected, corrected, cv::Size(TARGET_W, TARGET_H), 0, 0, interp);
            std::cout << " → resolution: " << corrected.cols << "x" << corrected.rows;
        }

        std::cout << "\n";
    }

    return corrected;
}
// =============================================================================
// 3. PLOTTING FUNCTION
// =============================================================================
// Maps over processed images, performs final detection, and builds graphical UI.
int plotting(const std::vector<cv::Mat>& deskewed_imgs, const std::vector<std::string>& image_paths, std::vector<cv::Mat>& odeskewed_imgs)
{
    // Local nested lambda for creating side-by-side display images
    auto make_side_by_side = [](const cv::Mat& left, const cv::Mat& right,
        const std::string& title_left,
        const std::string& title_right,
        int font_height = 50) -> cv::Mat
        {
            int h = std::max(left.rows, right.rows) + font_height + 8;
            int w = left.cols + right.cols + 4;
            cv::Mat out(h, w, CV_8UC3, cv::Scalar(30, 30, 30));

            cv::Mat roi_l = out(cv::Rect(0, font_height + 4, left.cols, left.rows));
            left.copyTo(roi_l);
            cv::putText(out, title_left, { 4, font_height },
                cv::FONT_HERSHEY_SIMPLEX, 0.65, { 220, 220, 220 }, 1, cv::LINE_AA);

            int rx = left.cols + 4;
            cv::Mat roi_r = out(cv::Rect(rx, font_height + 4, right.cols, right.rows));
            right.copyTo(roi_r);
            cv::putText(out, title_right, { rx + 4, font_height },
                cv::FONT_HERSHEY_SIMPLEX, 0.65, { 220, 220, 220 }, 1, cv::LINE_AA);

            return out;
        };

    std::vector<cv::Mat> all_panels;

    // Generate visualizations for every deskewed and raw image pairing.
    for (int img_idx = 0; img_idx < (int)deskewed_imgs.size(); img_idx++)
    {
        const std::string& raw_path = image_paths[img_idx];
        const cv::Mat& raw_img = deskewed_imgs[img_idx];
        const cv::Mat& odeskew = odeskewed_imgs[img_idx];
        //cv::Mat raw_original = cv::imread(raw_path);
        cv::Mat raw_original = deskewed_imgs[img_idx];

        auto [unique_contours, detected_objects] = detection(raw_img, ThreshMode::DETECTION);

        int n_cnt = (int)unique_contours.size();

        cv::Mat overlay = raw_img.clone();
        int thickness = std::max(3, (int)std::round(3.0 * raw_img.cols / 850.0));
        cv::drawContours(overlay, unique_contours, -1, cv::Scalar(0, 0, 255), thickness);

        std::string t_left = "Raw Image #" + std::to_string(img_idx + 1);
        std::string t_right = std::to_string(n_cnt) + " Shapes Detected";

        cv::Mat raw_disp;
        if (!raw_original.empty())
        {
            double sf = (double)overlay.rows / raw_original.rows;
            cv::resize(raw_original, raw_disp, {}, sf, sf, cv::INTER_AREA);
        }
        else
        {
            raw_disp = overlay.clone();
        }

        cv::Mat panel = make_side_by_side(odeskew, raw_img, t_left, t_right);
        //cv::Mat panel = make_side_by_side(odeskew, raw_disp, t_left, t_right);

        const int DISPLAY_W = 900;
        if (panel.cols > DISPLAY_W)
        {
            double sf2 = (double)DISPLAY_W / panel.cols;
            cv::resize(panel, panel, {}, sf2, sf2, cv::INTER_AREA);
        }

        std::ostringstream fname;
        fname << "result_" << std::setfill('0') << std::setw(3) << (img_idx + 1) << ".jpg";
        cv::imwrite(fname.str(), panel);

        all_panels.push_back(panel.clone());
        std::cout << "Image #" << (img_idx + 1) << ": "
            << n_cnt << " shapes detected → saved " << fname.str() << "\n";
    }

    // SCROLLABLE MOSAIC execution phase
    if (all_panels.empty())
    {
        return 0;
    }

    int mosaic_w = all_panels[0].cols;
    const int MAX_ROW_H = 500;
    int row_h = std::min(all_panels[0].rows, MAX_ROW_H);

    for (auto& p : all_panels)
    {
        if (p.rows > MAX_ROW_H)
        {
            double sf3 = (double)MAX_ROW_H / p.rows;
            cv::resize(p, p, {}, sf3, sf3, cv::INTER_AREA);
        }
    }
    int mosaic_h = row_h * (int)all_panels.size();

    cv::Mat mosaic(mosaic_h, mosaic_w, CV_8UC3, cv::Scalar(20, 20, 20));
    for (int i = 0; i < (int)all_panels.size(); i++)
    {
        int panel_h = all_panels[i].rows;
        int panel_w = std::min(mosaic_w, all_panels[i].cols);
        int copy_h = std::min(row_h, panel_h);
        int dest_y = i * row_h;
        if (dest_y + copy_h <= mosaic_h && panel_w > 0 && copy_h > 0)
        {
            cv::Mat src_roi = all_panels[i](cv::Rect(0, 0, panel_w, copy_h));
            cv::Mat dst_roi = mosaic(cv::Rect(0, dest_y, panel_w, copy_h));
            src_roi.copyTo(dst_roi);
        }
    }

    const int VIEWPORT_H = 800;
    int max_scroll = std::max(0, mosaic_h - VIEWPORT_H);
    int scroll_pos = 0;

    const std::string WIN = "IDPA Detection Results  [trackbar=scroll | q=quit]";
    cv::namedWindow(WIN, cv::WINDOW_NORMAL);
    cv::resizeWindow(WIN, mosaic_w, VIEWPORT_H);

    cv::createTrackbar("Scroll", WIN, &scroll_pos, std::max(max_scroll, 1));

    while (true)
    {
        scroll_pos = std::max(0, std::min(scroll_pos, max_scroll));

        int slice_h = std::min(VIEWPORT_H, mosaic_h - scroll_pos);
        if (slice_h <= 0)
            continue;
        cv::Mat view = mosaic(cv::Rect(0, scroll_pos, mosaic_w, slice_h)).clone();

        int bar_h = (int)((double)VIEWPORT_H * VIEWPORT_H / mosaic_h);
        int bar_y = (int)((double)scroll_pos / mosaic_h * VIEWPORT_H);
        bar_h = std::max(bar_h, 10);
        cv::rectangle(view,
            { mosaic_w - 8, bar_y },
            { mosaic_w - 2, std::min(bar_y + bar_h, VIEWPORT_H - 1) },
            cv::Scalar(100, 200, 100), -1);

        cv::imshow(WIN, view);

        int key = cv::waitKey(30);
        if (key == 'q' || key == 27)
            break;

        if (key == 82 || key == 'w')
            scroll_pos -= row_h / 4;
        if (key == 84 || key == 's')
            scroll_pos += row_h / 4;
        if (key == 85)
            scroll_pos -= VIEWPORT_H;
        if (key == 86)
            scroll_pos += VIEWPORT_H;
        if (key == 'f')
            scroll_pos = 0;
        if (key == 'l')
            scroll_pos = max_scroll;

        cv::setTrackbarPos("Scroll", WIN, scroll_pos);
    }

    cv::destroyAllWindows();
    return 0;
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================
int main(int argc, char** argv)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    // Default folder path
    std::string folder_path = "C:\\Users\\PMLS\\Downloads\\IDPA Target Detection\\test images";

    // Allow overriding the folder path via command line argument
    if (argc > 1)
    {
        folder_path = argv[1];
    }

    std::vector<std::string> image_paths;

    // Read the directory contents
    try
    {
        for (const auto& entry : std::filesystem::directory_iterator(folder_path))
        {
            if (entry.is_regular_file())
            {
                std::string ext = entry.path().extension().string();

                // Convert extension to lowercase to catch .JPG, .PNG, etc.
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
                {
                    image_paths.push_back(entry.path().string());
                }
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        std::cerr << "Error reading directory: " << e.what() << "\n";
        return -1;
    }

    if (image_paths.empty())
    {
        std::cerr << "No images found in directory: " << folder_path << "\n";
        return -1;
    }

    std::cout << "Found " << image_paths.size() << " images in folder.\n";

    std::vector<cv::Mat> deskewed_imgs;
    std::vector<cv::Mat> odeskewed_imgs;

    // Feed raw images sequentially into the overarching deskew process
    int img_num = 1;
    for (const auto& path : image_paths)
    {
        cv::Mat raw = cv::imread(path);
        if (raw.empty())
        {
            std::cout << "Error: could not load " << path << "\n";
            continue;
        }
        std::cout << "Processing image #" << img_num++ << ": " << path << "...\n";
        //cv::Mat deskewed = deskewing(raw, 10.0);
        //deskewed_imgs.push_back(refineWarpAfterDeskew(deskewed));
        cv::Mat odeskew = deskewing(raw, 10.0);
        std::cout << "\n--- Stage D: Vertical Edge Warp ---\n";
        cv:: Mat straightened = applyVerticalEdgeWarpCorrection(
            odeskew,
            /*rms_threshold_px*/ 1.5,
            /*sample_rows*/      60,
            /*max_shift_px*/     35.0,   // was 18
            /*smooth_sigma*/     4.0
        ); 
        deskewed_imgs.push_back(odeskew);
        odeskewed_imgs.push_back(straightened);
        //deskewed_imgs.push_back(deskewing(raw, 10.0));
        // 1. Remove ANY previous declarations of 'corrected' or 'odeskewed' 
       // 1. Unpack the two outputs directly into separate variables\

        // 2. Push them into your respective vectors

    }

    plotting(deskewed_imgs, image_paths, odeskewed_imgs);
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n\n";
    cv::waitKey(0);
    return 0;
}