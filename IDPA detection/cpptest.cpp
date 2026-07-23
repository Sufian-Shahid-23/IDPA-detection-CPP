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
#include <execution>
#include <atomic>
#include <mutex>

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

// =============================================================================
// CENTRAL CONFIGURATION
// =============================================================================
struct PipelineConfig {
    // Shape Size Thresholds (relative to image area)
    static constexpr double minAreaRatio = 0.002;
    static constexpr double largeShapeRatio = 0.45;

    // Bullet Hole Removal - Torso
    static constexpr double torsoMaxHoleWidth = 50.0;
    static constexpr double torsoMinHoleDepth = 0.5;
    static constexpr int torsoWindowSize = 50;

    // Bullet Hole Removal - Octagon
    static constexpr double octagonMaxHoleWidth = 60.0;
    static constexpr double octagonMinHoleDepth = 0.1;
    static constexpr int octagonWindowSize = 20;

    // Geometric Protections
    static constexpr double cornerProtectRadius = 50.0;
    static constexpr double inkOffset = 4.0;
};

// Struct to pass masks between pipeline stages
struct ProcessedMasks {
    cv::Mat thresh_healed_outer;
    cv::Mat thresh_healed;
    cv::Mat color_mask;
};

// =============================================================================
// GEOMETRY & DESKEW UTILITIES
// =============================================================================
namespace GeometryUtils {

    struct Line {
        double a, b, c;
    };

    inline Line lineFromPts(cv::Point2d p1, cv::Point2d p2) {
        double a = p2.y - p1.y;
        double b = p1.x - p2.x;
        double c = -(a * p1.x + b * p1.y);
        return { a, b, c };
    }

    inline bool intersect(Line l1, Line l2, cv::Point2d& out) {
        double det = l1.a * l2.b - l2.a * l1.b;
        if (std::abs(det) < 1e-9) return false;
        out.x = (l1.b * l2.c - l2.b * l1.c) / det;
        out.y = (l2.a * l1.c - l1.a * l2.c) / det;
        return true;
    }

    inline bool get_octagon_corners(const Contour& poly, std::vector<cv::Point2f>& corners_out, bool debug = false) {
        if ((int)poly.size() != 8) {
            if (debug) std::cerr << "  [octagon] expected 8 pts, got " << poly.size() << "\n";
            return false;
        }

        std::vector<cv::Point2d> pts(8);
        for (int i = 0; i < 8; i++) pts[i] = cv::Point2d(poly[i].x, poly[i].y);

        std::vector<double> elen(8);
        for (int i = 0; i < 8; i++) {
            auto& p1 = pts[i];
            auto& p2 = pts[(i + 1) % 8];
            elen[i] = std::hypot(p2.x - p1.x, p2.y - p1.y);
        }

        double se = 0, so = 0;
        for (int k = 0; k < 4; k++) {
            se += elen[k * 2];
            so += elen[k * 2 + 1];
        }
        int ss = (se > so) ? 0 : 1;

        struct Side { cv::Point2d p1, p2, mid; };
        std::vector<Side> sides(4);
        for (int k = 0; k < 4; k++) {
            int i = (ss + k * 2) % 8;
            sides[k].p1 = pts[i];
            sides[k].p2 = pts[(i + 1) % 8];
            sides[k].mid = { (pts[i].x + pts[(i + 1) % 8].x) / 2, (pts[i].y + pts[(i + 1) % 8].y) / 2 };
        }

        std::sort(sides.begin(), sides.end(), [](const Side& a, const Side& b) { return a.mid.y < b.mid.y; });
        Side top_s = sides[0];
        Side bottom_s = sides[3];
        std::vector<Side> rem = { sides[1], sides[2] };
        std::sort(rem.begin(), rem.end(), [](const Side& a, const Side& b) { return a.mid.x < b.mid.x; });
        Side left_s = rem[0];
        Side right_s = rem[1];

        Line L_top = lineFromPts(top_s.p1, top_s.p2);
        Line L_bottom = lineFromPts(bottom_s.p1, bottom_s.p2);
        Line L_left = lineFromPts(left_s.p1, left_s.p2);
        Line L_right = lineFromPts(right_s.p1, right_s.p2);

        cv::Point2d TL, TR, BR, BL;
        if (!intersect(L_top, L_left, TL) || !intersect(L_top, L_right, TR) ||
            !intersect(L_bottom, L_right, BR) || !intersect(L_bottom, L_left, BL)) {
            if (debug) std::cerr << "  [octagon] line intersection failed\n";
            return false;
        }

        corners_out = {
            cv::Point2f((float)TL.x, (float)TL.y), cv::Point2f((float)TR.x, (float)TR.y),
            cv::Point2f((float)BR.x, (float)BR.y), cv::Point2f((float)BL.x, (float)BL.y)
        };

        if (debug) std::cout << "  Corners TL=" << TL << " TR=" << TR << " BR=" << BR << " BL=" << BL << "\n";
        return true;
    }

    inline std::pair<bool, double> quad_needs_perspective_fix(const std::vector<cv::Point2f>& c, double angle_tol = 2.0, double ratio_tol = 0.02) {
        cv::Point2f TL = c[0], TR = c[1], BR = c[2], BL = c[3];

        auto vlen = [](cv::Point2f a, cv::Point2f b) { return std::hypot(b.x - a.x, b.y - a.y); };
        double top_len = vlen(TL, TR), bot_len = vlen(BL, BR);
        double left_len = vlen(TL, BL), right_len = vlen(TR, BR);

        double w_ratio = std::min(top_len, bot_len) / std::max(top_len, bot_len + 1e-9);
        double h_ratio = std::min(left_len, right_len) / std::max(left_len, right_len + 1e-9);

        auto angle_at = [](cv::Point2f pp, cv::Point2f pc, cv::Point2f pn) {
            double v1x = pp.x - pc.x, v1y = pp.y - pc.y;
            double v2x = pn.x - pc.x, v2y = pn.y - pc.y;
            double dot = v1x * v2x + v1y * v2y;
            // vecNorm is defined in your math helpers globally
            double n = vecNorm(v1x, v1y) * vecNorm(v2x, v2y) + 1e-9;
            return std::acos(std::max(-1.0, std::min(1.0, dot / n))) * 180.0 / CV_PI;
            };

        double angles[4] = {
            angle_at(BL, TL, TR), angle_at(TL, TR, BR),
            angle_at(TR, BR, BL), angle_at(BR, BL, TL)
        };

        double max_dev = 0;
        for (double a : angles) max_dev = std::max(max_dev, std::abs(a - 90.0));

        double ratio_dev = std::max(1.0 - w_ratio, 1.0 - h_ratio);
        bool needs_fix = (max_dev > angle_tol) || (ratio_dev > ratio_tol);
        double score = max_dev + ratio_dev * 100.0;
        return { needs_fix, score };
    }

    inline std::pair<std::vector<cv::Point2f>, double> expand_quad_to_fit_contour(const std::vector<cv::Point2f>& quad, const Contour& target_pts, double padding_ratio = 0.0, double max_scale = 2.5) {
        cv::Point2d C(0, 0);
        for (const auto& p : quad) { C.x += p.x; C.y += p.y; }
        C.x /= 4; C.y /= 4;

        std::vector<cv::Point2d> Q(4);
        for (int i = 0; i < 4; i++) Q[i] = { quad[i].x, quad[i].y };

        int ei[4][2] = { {0, 1}, {1, 2}, {2, 3}, {3, 0} };
        double required_k = 1.0;

        for (auto& e : ei) {
            cv::Point2d p1 = Q[e[0]], p2 = Q[e[1]];
            double dx = p2.x - p1.x, dy = p2.y - p1.y;
            double nx = dy, ny = -dx;
            double nl = std::hypot(nx, ny) + 1e-12;
            nx /= nl; ny /= nl;
            if (nx * (C.x - p1.x) + ny * (C.y - p1.y) > 0) { nx = -nx; ny = -ny; }
            double m = -(nx * (C.x - p1.x) + ny * (C.y - p1.y));
            if (m < 1e-9) continue;

            for (const auto& tp : target_pts) {
                double g = nx * (tp.x - p1.x) + ny * (tp.y - p1.y);
                double k = 1.0 + g / m;
                required_k = std::max(required_k, k);
            }
        }

        double k_final = required_k * (1.0 + padding_ratio);
        if (k_final > max_scale) {
            std::cout << "  [warn] scale " << k_final << " clamped to " << max_scale << "\n";
            k_final = max_scale;
        }

        std::vector<cv::Point2f> expanded(4);
        for (int i = 0; i < 4; i++) {
            expanded[i].x = (float)(C.x + k_final * (Q[i].x - C.x));
            expanded[i].y = (float)(C.y + k_final * (Q[i].y - C.y));
        }
        return { expanded, k_final };
    }

    inline cv::Mat warp_from_corners(const cv::Mat& img, const std::vector<cv::Point2f>& src, cv::Mat* M_out = nullptr) {
        double w_top = std::hypot(src[1].x - src[0].x, src[1].y - src[0].y);
        double w_bot = std::hypot(src[2].x - src[3].x, src[2].y - src[3].y);
        double h_left = std::hypot(src[3].x - src[0].x, src[3].y - src[0].y);
        double h_right = std::hypot(src[2].x - src[1].x, src[2].y - src[1].y);
        int ow = static_cast<int>(std::max(w_top, w_bot));
        int oh = static_cast<int>(std::max(h_left, h_right));
        ow = std::max(ow, 1); oh = std::max(oh, 1);

        std::vector<cv::Point2f> dst = { {0.f, 0.f}, {(float)(ow - 1), 0.f}, {(float)(ow - 1), (float)(oh - 1)}, {0.f, (float)(oh - 1)} };

        cv::Mat M = cv::getPerspectiveTransform(src, dst);
        if (M_out) *M_out = M;
        cv::Mat warped;
        cv::warpPerspective(img, warped, M, { ow, oh });
        return warped;
    }

    inline cv::Mat get_thresh(const cv::Mat& image) {
        cv::Mat g, b, t;
        cv::cvtColor(image, g, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(g, b, cv::Size(5, 5), 0);
        int bk = static_cast<int>(image.cols * 0.02);
        if (bk % 2 == 0) bk++;
        bk = std::max(bk, 11);
        cv::adaptiveThreshold(b, t, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, bk, 3);
        cv::Mat ones = cv::Mat::ones(3, 3, CV_8U);
        cv::morphologyEx(t, t, cv::MORPH_CLOSE, ones);
        return t;
    }

    inline bool find_torso_polygon(const cv::Mat& image, Contour& cnt_out, Contour& approx_out, cv::Rect& bbox_out) {
        int ih = image.rows, iw = image.cols;
        cv::Mat t = get_thresh(image);
        ContourVec conts;
        cv::findContours(t, conts, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        struct Cand { Contour cnt, approx; double area; cv::Rect bbox; };
        std::vector<Cand> candidates;

        for (const auto& cnt : conts) {
            double area = cv::contourArea(cnt);
            if (area < ih * iw * 0.02 || area > ih * iw * 0.80) continue;
            if ((int)cnt.size() < 5) continue;

            Contour hull;
            cv::convexHull(cnt, hull);
            double ha = cv::contourArea(hull);
            double sol = (ha > 0) ? area / ha : 0;
            if (sol < 0.70) continue;

            double eps = 0.02 * cv::arcLength(cnt, true);
            Contour approx;
            cv::approxPolyDP(cnt, approx, eps, true);
            int n = static_cast<int>(approx.size());
            if (n < 5 || n > 16) continue;

            cv::Rect br = cv::boundingRect(cnt);
            if ((double)br.width / br.height < 0.25 || (double)br.width / br.height > 2.0) continue;
            if (br.x < 5 && br.y < 5 && br.width > iw * 0.9) continue;

            candidates.push_back({ cnt, approx, area, br });
        }
        if (candidates.empty()) return false;

        std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) { return a.area > b.area; });
        cnt_out = candidates[0].cnt;
        approx_out = candidates[0].approx;
        bbox_out = candidates[0].bbox;
        return true;
    }

    inline bool hough_perspective_recovery(
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

} // namespace GeometryUtils

// =============================================================================
// CONTOUR UTILITIES (Optimized)
// =============================================================================
namespace ContourUtils {

    // Helper to calculate squared distance to avoid expensive std::hypot roots
    inline double distSq(const cv::Point2d& a, const cv::Point2d& b) {
        double dx = b.x - a.x;
        double dy = b.y - a.y;
        return dx * dx + dy * dy;
    }

    Contour densifyContour(const Contour& cnt, double maxSpacing = 2.0)
    {
        int n = (int)cnt.size();
        if (n == 0) return {};

        // OPTIMIZATION 1: Pre-calculate total required size to prevent reallocations
        int total_points = 0;
        std::vector<int> steps_per_segment(n);
        for (int i = 0; i < n; i++)
        {
            cv::Point2d p1(cnt[i].x, cnt[i].y);
            cv::Point2d p2(cnt[(i + 1) % n].x, cnt[(i + 1) % n].y);
            double dist = std::hypot(p2.x - p1.x, p2.y - p1.y);
            int steps = (int)(dist / maxSpacing);
            steps_per_segment[i] = steps;
            total_points += 1 + std::max(0, steps - 1);
        }

        Contour dense;
        dense.reserve(total_points); // Massive speedup for large contours

        for (int i = 0; i < n; i++)
        {
            cv::Point2d p1(cnt[i].x, cnt[i].y);
            cv::Point2d p2(cnt[(i + 1) % n].x, cnt[(i + 1) % n].y);
            dense.push_back(cnt[i]);

            int steps = steps_per_segment[i];
            for (int s = 1; s < steps; s++)
            {
                double t = (double)s / steps;
                dense.push_back(cv::Point(
                    (int)std::round(p1.x + t * (p2.x - p1.x)),
                    (int)std::round(p1.y + t * (p2.y - p1.y))
                ));
            }
        }
        return dense;
    }

    Contour removeBulletHoles(const Contour& input,
        const std::vector<bool>& protected_,
        double maxHoleWidth = 50.0,
        double minHoleDepth = 0.5,
        int    windowSize = 8)
    {
        int n = (int)input.size();
        if (n < 24) return input;

        std::vector<double> depth(n, 0.0);
        std::vector<bool>   concave(n, false);

        for (int i = 0; i < n; i++)
        {
            int iPrev = (i - windowSize + n) % n;
            int iNext = (i + windowSize) % n;

            cv::Point2d prev(input[iPrev].x, input[iPrev].y);
            cv::Point2d curr(input[i].x, input[i].y);
            cv::Point2d next(input[iNext].x, input[iNext].y);

            double chordX = next.x - prev.x;
            double chordY = next.y - prev.y;
            double chordLenSq = chordX * chordX + chordY * chordY;
            if (chordLenSq < 1e-12) continue;

            double chordLen = std::sqrt(chordLenSq);
            double nx = -chordY / chordLen;
            double ny = chordX / chordLen;

            double midX = (prev.x + next.x) * 0.5;
            double midY = (prev.y + next.y) * 0.5;
            double d = (curr.x - midX) * nx + (curr.y - midY) * ny;

            double travelX = next.x - prev.x;
            double travelY = next.y - prev.y;
            double toX = curr.x - prev.x;
            double toY = curr.y - prev.y;
            double cross = travelX * toY - travelY * toX;

            depth[i] = std::abs(d);
            concave[i] = (cross > 0);
        }

        std::vector<bool> remove(n, false);
        double maxHoleWidthSq = maxHoleWidth * maxHoleWidth; // OPTIMIZATION 2: Pre-square

        for (int i = 0; i < n; i++)
        {
            if (depth[i] < minHoleDepth || remove[i] || protected_[i]) continue;

            double peakDepth = depth[i];
            int leftEdge = i;
            int rightEdge = i;

            for (int k = 1; k < windowSize * 3; k++)
            {
                int iTest = (i - k + n) % n;
                leftEdge = iTest;
                if (depth[iTest] < peakDepth * 0.50 || protected_[iTest]) break;
            }
            for (int k = 1; k < windowSize * 3; k++)
            {
                int iTest = (i + k) % n;
                rightEdge = iTest;
                if (depth[iTest] < peakDepth * 0.50 || protected_[iTest]) break;
            }

            cv::Point2d pL(input[leftEdge].x, input[leftEdge].y);
            cv::Point2d pR(input[rightEdge].x, input[rightEdge].y);

            // Compare squared distance to avoid square root
            double holeWidthSq = distSq(pL, pR);
            int spanPoints = (rightEdge - leftEdge + n) % n;

            if (protected_[i] || spanPoints < 10 || holeWidthSq > maxHoleWidthSq) continue;

            int idx = (leftEdge + 1) % n;
            int safety = 0;
            while (idx != rightEdge && safety++ < n)
            {
                if (!protected_[idx])
                    remove[idx] = true;
                idx = (idx + 1) % n;
            }
        }

        Contour result;
        result.reserve(n); // Ensure fast push_back
        for (int i = 0; i < n; i++) {
            if (!remove[i]) result.push_back(input[i]);
        }

        return result.empty() ? input : result;
    }

    std::pair<std::vector<cv::Point2d>, bool> findTrueCorners(
        const Contour& dense,
        bool isOctagon,
        double protectRadius = 50.0)
    {
        std::vector<cv::Point2d> corners;

        if (isOctagon)
        {
            // Precompute centroid for outward normal direction
            cv::Point2d centroid(0, 0);
            for (auto& dp : dense) { centroid.x += dp.x; centroid.y += dp.y; }
            centroid.x /= (double)dense.size();
            centroid.y /= (double)dense.size();

            Contour ch;
            cv::convexHull(dense, ch);
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
                if (best_diff == 0) break;
            }

            // Signed area helper
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
                    double lp = std::hypot(pc.x - pp.x, pc.y - pp.y);
                    double ln = std::hypot(pc.x - pn.x, pc.y - pn.y);
                    if (lp >= side_thresh || ln >= side_thresh) continue;
                    Contour trial = approx;
                    trial.erase(trial.begin() + i);
                    double trial_area = std::abs(signed_area(trial));
                    if (trial_area < base_area * 0.998) continue;
                    double score = 1.0 / (lp + ln + 1e-6);
                    if (score > worst_score)
                    {
                        worst_score = score;
                        worst_idx = i;
                    }
                }
                if (worst_idx >= 0) { approx.erase(approx.begin() + worst_idx); changed = true; }
            }

            if ((int)approx.size() < 6)
                cv::approxPolyDP(ch, approx, 0.015 * hull_perim, true);


            // ── NOW upgrade each rough corner to precise intersection ──────
            // This fixes the bullet-hole-at-corner problem:
            // The hull corner is pulled toward the bullet hole,
            // but the fitted lines through the full edges are not affected.
            int m = (int)approx.size();
            for (int i = 0; i < m; i++)
            {
                cv::Point2f p0 = approx[(i - 1 + m) % m];
                cv::Point2f p1 = approx[i];
                cv::Point2f p2 = approx[(i + 1) % m];

                // Collect dense points near edge (p0→p1)
                auto collectEdgePts = [&](cv::Point2f ea, cv::Point2f eb)
                    -> std::vector<cv::Point2f>
                    {
                        std::vector<cv::Point2f> pts;
                        double elen = std::hypot(eb.x - ea.x, eb.y - ea.y);
                        if (elen < 1e-6) return pts;
                        double edx = (eb.x - ea.x) / elen, edy = (eb.y - ea.y) / elen;
                        double nx = -edy, ny = edx;
                        for (auto& dp : dense)
                        {
                            double tx = dp.x - ea.x, ty = dp.y - ea.y;
                            double perp = std::abs(tx * nx + ty * ny);
                            double proj = tx * edx + ty * edy;
                            if (perp < 18.0 && proj > -20.0 && proj < elen + 20.0)
                                pts.push_back(cv::Point2f(dp.x, dp.y));
                        }
                        return pts;
                    };

                auto fitEdgeLine = [&](cv::Point2f ea, cv::Point2f eb,
                    double& ax, double& ay,
                    double& dx, double& dy) -> bool
                    {
                        auto pts = collectEdgePts(ea, eb);
                        if ((int)pts.size() < 6)
                        {
                            double elen = std::hypot(eb.x - ea.x, eb.y - ea.y) + 1e-9;
                            ax = ea.x; ay = ea.y;
                            dx = (eb.x - ea.x) / elen; dy = (eb.y - ea.y) / elen;
                            return false;
                        }
                        cv::Vec4f lp;
                        cv::fitLine(pts, lp, cv::DIST_L2, 0, 0.01, 0.01);
                        ax = lp[2]; ay = lp[3]; dx = lp[0]; dy = lp[1];
                        return true;
                    };
                // Fit lines for the two edges meeting at this corner
                double ax1, ay1, dx1, dy1, ax2, ay2, dx2, dy2;
                fitEdgeLine(p0, p1, ax1, ay1, dx1, dy1);
                fitEdgeLine(p1, p2, ax2, ay2, dx2, dy2);

                // Intersect the two fitted lines
                double ddx = ax2 - ax1, ddy = ay2 - ay1;
                double denom = dx1 * dy2 - dy1 * dx2;

                if (std::abs(denom) < 1e-9)
                {
                    // Parallel — keep original rough corner
                    corners.push_back({ p1.x, p1.y });
                }
                else
                {
                    double t = (ddx * dy2 - ddy * dx2) / denom;
                    corners.push_back({ ax1 + t * dx1, ay1 + t * dy1 });
                }
            }
            return { corners, false }; // octagon never measures straightness
        }
        else
        {
            // ── TORSO: convex corners via hull + neck corners via defects ──

            // Part 1: 10 convex corners same sweep approach
            Contour hull;
            cv::convexHull(dense, hull);
            double hullP = cv::arcLength(hull, true);

            Contour best;
            int bestDiff = 9999;
            for (double ef = 0.01; ef <= 0.12; ef += 0.001)
            {
                Contour trial;
                cv::approxPolyDP(hull, trial, ef * hullP, true);
                int diff = std::abs((int)trial.size() - 10);
                if (diff < bestDiff) { bestDiff = diff; best = trial; }
                if (bestDiff == 0) break;
            }

            // Upgrade each rough torso corner to precise intersection
            int m = (int)best.size();
            for (int i = 0; i < m; i++)
            {
                cv::Point2f p0 = best[(i - 1 + m) % m];
                cv::Point2f p1 = best[i];
                cv::Point2f p2 = best[(i + 1) % m];

                auto collectEdgePts = [&](cv::Point2f ea, cv::Point2f eb)
                    -> std::vector<cv::Point2f>
                    {
                        std::vector<cv::Point2f> pts;
                        double elen = std::hypot(eb.x - ea.x, eb.y - ea.y);
                        if (elen < 1e-6) return pts;
                        double edx = (eb.x - ea.x) / elen, edy = (eb.y - ea.y) / elen;
                        double nx = -edy, ny = edx;

                        // Compute centroid of dense contour to know which way is "inward"
                        cv::Point2d centroid(0, 0);
                        for (auto& dp : dense) { centroid.x += dp.x; centroid.y += dp.y; }
                        centroid.x /= dense.size(); centroid.y /= dense.size();

                        // Mid-point of this edge
                        cv::Point2f emid((ea.x + eb.x) * 0.5f, (ea.y + eb.y) * 0.5f);

                        // If normal points toward centroid, flip it so nx,ny points OUTWARD
                        double toCx = centroid.x - emid.x, toCy = centroid.y - emid.y;
                        if (nx * toCx + ny * toCy > 0) { nx = -nx; ny = -ny; }

                        for (auto& dp : dense)
                        {
                            double tx = dp.x - ea.x, ty = dp.y - ea.y;
                            double perp = tx * nx + ty * ny;   // signed: + = outward, - = inward
                            double proj = tx * edx + ty * edy;
                            // Collect points in a band: allow up to 4px inward, 20px outward
                            // This pulls fitLine toward the OUTER edge of the black ink
                            if (perp >= -4.0 && perp <= 20.0 && proj > -20.0 && proj < elen + 20.0)
                                pts.push_back(cv::Point2f(dp.x, dp.y));
                        }
                        return pts;
                    };

                double ax1, ay1, dx1, dy1, ax2, ay2, dx2, dy2;

                auto fitEdge = [&](cv::Point2f ea, cv::Point2f eb,
                    double& ax, double& ay,
                    double& dx, double& dy)
                    {
                        auto pts = collectEdgePts(ea, eb);
                        if ((int)pts.size() < 6)
                        {
                            double el = std::hypot(eb.x - ea.x, eb.y - ea.y) + 1e-9;
                            ax = ea.x;ay = ea.y;dx = (eb.x - ea.x) / el;dy = (eb.y - ea.y) / el;
                            return;
                        }
                        cv::Vec4f lp;
                        cv::fitLine(pts, lp, cv::DIST_L2, 0, 0.01, 0.01);
                        ax = lp[2];ay = lp[3];dx = lp[0];dy = lp[1];
                    };

                fitEdge(p0, p1, ax1, ay1, dx1, dy1);
                fitEdge(p1, p2, ax2, ay2, dx2, dy2);

                double ddx = ax2 - ax1, ddy = ay2 - ay1;
                double denom = dx1 * dy2 - dy1 * dx2;
                if (std::abs(denom) < 1e-9)
                    corners.push_back({ p1.x,p1.y });
                else
                {
                    double t = (ddx * dy2 - ddy * dx2) / denom;
                    corners.push_back({ ax1 + t * dx1, ay1 + t * dy1 });
                }
            }
            // Measure straightness from the found corners
            bool isFlatTarget = false;
            if (corners.size() >= 8)
            {
                // Only use the 8 convex hull corners, not the 2 neck corners
                std::vector<cv::Point2d> convexOnly(corners.begin(), corners.begin() + 8);

                double minX = 1e9, maxX = -1e9;
                for (auto& c : convexOnly) { minX = std::min(minX, c.x); maxX = std::max(maxX, c.x); }

                std::vector<double> leftXs, rightXs;
                double width = maxX - minX;
                for (auto& c : convexOnly)
                {
                    if (c.x < minX + width * 0.25) leftXs.push_back(c.x);
                    if (c.x > maxX - width * 0.25) rightXs.push_back(c.x);
                }

                if (leftXs.size() >= 2 && rightXs.size() >= 2)
                {
                    auto variance = [](const std::vector<double>& v) -> double {
                        double mean = 0; for (double x : v) mean += x; mean /= v.size();
                        double var = 0; for (double x : v) var += (x - mean) * (x - mean);
                        return var / v.size();
                        };

                    double leftVar = variance(leftXs);
                    double rightVar = variance(rightXs);


                    // In straightness check — increase variance threshold
                    isFlatTarget = (leftVar < 5000.0 && rightVar < 5000.0);  // was 100.0
                }
            }

            // Part 2: Add 2 neck corners via deepest convexity defect
            // — upgraded to line-fit intersection
            std::vector<int> hullIdx;
            cv::convexHull(dense, hullIdx, false, false);
            std::vector<cv::Vec4i> defects;
            cv::convexityDefects(dense, hullIdx, defects);

            if (!defects.empty())
            {
                for (int d = 0; d < std::min(5, (int)defects.size()); d++)
                {
                    int si = defects[d][0], ei = defects[d][1];
                    int rawDepth = defects[d][3];
                    /*std::cout << "  [DEFECT PRE-SORT " << d << "] rawdepth=" << rawDepth
                        << " start=(" << dense[si].x << "," << dense[si].y
                        << ") end=(" << dense[ei].x << "," << dense[ei].y << ")\n";*/
                }

                std::sort(defects.begin(), defects.end(),
                    [](const cv::Vec4i& a, const cv::Vec4i& b)
                    { return a[3] > b[3]; });

                int startIdx = defects[0][0];
                int endIdx = defects[0][1];
                int N = (int)dense.size();

                // For each raw neck junction, collect points along the two
                // edges that meet there and intersect their fitted lines.
                // Window: look windowW points to each side of the junction.
                int windowW = 60;

                // Replace the entire upgradeNeckCorner lambda and its calls with just:
                cv::Point2d neckLeft = { (double)dense[startIdx].x, (double)dense[startIdx].y };
                cv::Point2d neckRight = { (double)dense[endIdx].x,   (double)dense[endIdx].y };

                corners.push_back(neckLeft);
                corners.push_back(neckRight);

                /*std::cout << "  [CORNERS] Neck corners (raw): ("
                    << neckLeft.x << "," << neckLeft.y << ") and ("
                    << neckRight.x << "," << neckRight.y << ")\n";*/

            }

            return { corners, isFlatTarget };
        }
    }

} // namespace ContourUtils

// =============================================================================
// 1A. PIPELINE: PREPROCESS IMAGE
// =============================================================================
ProcessedMasks preprocessImage(const cv::Mat& img, ThreshMode mode)
{
    cv::Mat gray, blurred;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // ── DARK CIRCLE PRE-PASS ──
    cv::Mat gray_preblur;
    cv::GaussianBlur(gray, gray_preblur, cv::Size(5, 5), 0);
    cv::Mat dark_circle_mask;
    cv::threshold(gray_preblur, dark_circle_mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    ContourVec dark_blob_cnts;
    cv::findContours(dark_circle_mask, dark_blob_cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double IMG_AREA_PRE = (double)img.rows * img.cols;
    ContourVec saved_dark_circles;
    for (const auto& dc : dark_blob_cnts)
    {
        double a = cv::contourArea(dc);
        if (a < IMG_AREA_PRE * 0.005 || a > IMG_AREA_PRE * 0.40) continue;
        Contour dh;
        cv::convexHull(dc, dh);
        double ha = cv::contourArea(dh);
        double sol = (ha > 0) ? a / ha : 0;
        if (sol < 0.80) continue;
        cv::Rect dbr = cv::boundingRect(dc);
        double dar = (dbr.height > 0) ? (double)dbr.width / dbr.height : 0;
        if (dar < 0.50 || dar > 2.0) continue;
        saved_dark_circles.push_back(dc);
    }

    // ── SMART POLARITY INVERSION ──
    cv::Rect center_roi(img.cols / 4, img.rows / 4, img.cols / 2, img.rows / 2);
    double center_mean = cv::mean(gray(center_roi))[0];
    if (center_mean < 100) cv::bitwise_not(gray, gray);

    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    int block = static_cast<int>(img.cols * 0.02);
    if (block % 2 == 0) block++;
    block = std::max(block, 21);

    int adaptive_C = (mode == ThreshMode::DESKEW) ? 3 : 2;
    int kernel_size = (mode == ThreshMode::DESKEW) ? 3 : 2;

    cv::Mat thresh_light, thresh_healed;
    cv::adaptiveThreshold(blurred, thresh_light, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, block, adaptive_C);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(thresh_light, thresh_healed, cv::MORPH_CLOSE, k);

    // ── BOUNDARY BIAS CORRECTION (Dark backgrounds) ──
    cv::Mat thresh_healed_outer = thresh_healed.clone();
    if (center_mean < 80.0)
    {
        cv::Mat gray_orig, bin_orig;
        cv::cvtColor(img, gray_orig, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_orig, bin_orig, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat filled = bin_orig.clone(), padded;
        cv::copyMakeBorder(filled, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
        cv::floodFill(padded, cv::Point(0, 0), 128);
        cv::Mat filled_crop = padded(cv::Rect(1, 1, filled.cols, filled.rows));

        cv::Mat outer_mask = cv::Mat::zeros(bin_orig.size(), CV_8UC1);
        for (int y = 0; y < filled_crop.rows; y++) {
            for (int x = 0; x < filled_crop.cols; x++) {
                if (filled_crop.at<uchar>(y, x) != 128 || bin_orig.at<uchar>(y, x) == 255)
                    outer_mask.at<uchar>(y, x) = 255;
            }
        }

        cv::Mat k_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(outer_mask, outer_mask, cv::MORPH_CLOSE, k_close);
        cv::Mat k_erode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(outer_mask, outer_mask, k_erode, cv::Point(-1, -1), 2);
        thresh_healed_outer = outer_mask;
    }

    // ── COLOR-ZONE MASK ──
    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);
    cv::Mat blur_b, blur_r, blur_g;
    cv::GaussianBlur(bgr[0], blur_b, cv::Size(5, 5), 0);
    cv::GaussianBlur(bgr[2], blur_r, cv::Size(5, 5), 0);
    cv::GaussianBlur(bgr[1], blur_g, cv::Size(5, 5), 0);

    cv::Mat diff_rb, diff_rg, diff_gr, diff_bg, diff_br;
    cv::subtract(blur_r, blur_b, diff_rb);
    cv::subtract(blur_r, blur_g, diff_rg);
    cv::subtract(blur_g, blur_r, diff_gr);
    cv::subtract(blur_b, blur_g, diff_bg);
    cv::subtract(blur_b, blur_r, diff_br);

    cv::Mat red_mask_b, red_mask_g, red_mask, green_mask, blue_mask_g_th, blue_mask_r_th, blue_mask, color_mask;
    cv::threshold(diff_rb, red_mask_b, 40, 255, cv::THRESH_BINARY);
    cv::threshold(diff_rg, red_mask_g, 40, 255, cv::THRESH_BINARY);
    cv::bitwise_and(red_mask_b, red_mask_g, red_mask);
    cv::threshold(diff_gr, green_mask, 30, 255, cv::THRESH_BINARY);
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
    ContourVec cm_contours;
    cv::findContours(color_mask, cm_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double img_area = (double)color_mask.rows * color_mask.cols;
    for (const auto& c : cm_contours)
    {
        double a = cv::contourArea(c);
        if (a < img_area * 0.001) continue;
        Contour hull;
        cv::convexHull(c, hull);
        double solidity = (cv::contourArea(hull) > 0) ? a / cv::contourArea(hull) : 0.0;
        cv::Rect br = cv::boundingRect(c);
        double fill_ratio = (br.area() > 0) ? a / (double)br.area() : 0.0;
        if (solidity > 0.85 && fill_ratio > 0.35)
            cv::drawContours(color_mask_filtered, ContourVec{ c }, 0, 255, -1);
    }

    if (mode == ThreshMode::DETECTION)
    {
        cv::bitwise_or(thresh_healed, color_mask_filtered, thresh_healed);
        for (const auto& dc : saved_dark_circles)
        {
            cv::Rect dbr = cv::boundingRect(dc);
            cv::Mat region = color_mask(dbr);
            if (cv::countNonZero(region) / (cv::contourArea(dc) + 1) < 0.20)
                cv::drawContours(thresh_healed, ContourVec{ dc }, 0, 255, -1);
        }
    }

    return { thresh_healed_outer, thresh_healed, color_mask };
}

// =============================================================================
// 1B. PIPELINE: EXTRACT CONTOURS
// =============================================================================
ContourVec extractContours(const ProcessedMasks& masks, double imgArea, ThreshMode mode)
{
    double LARGE_THRESH = imgArea * PipelineConfig::largeShapeRatio;

    ContourVec contours_outer, contours_inner, contours;
    std::vector<cv::Vec4i> hier_outer, hier_inner;

    cv::findContours(masks.thresh_healed_outer, contours_outer, hier_outer, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(masks.thresh_healed, contours_inner, hier_inner, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours_outer.size(); i++)
        if (cv::contourArea(contours_outer[i]) >= LARGE_THRESH)
            contours.push_back(contours_outer[i]);

    for (size_t i = 0; i < contours_inner.size(); i++)
        if (cv::contourArea(contours_inner[i]) < LARGE_THRESH)
            contours.push_back(contours_inner[i]);

    if (mode == ThreshMode::DETECTION)
    {
        ContourVec color_contours;
        cv::findContours(masks.color_mask, color_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto& cc : color_contours) contours.push_back(cc);
    }

    return contours;
}

// =============================================================================
// 1C. PIPELINE: CLASSIFY SHAPES
// =============================================================================
std::pair<ContourVec, ObjVec> classifyShapes(const ContourVec& contours, const cv::Mat& threshHealed, const cv::Mat& img, ThreshMode mode)
{
    ContourVec unique_contours;
    ObjVec detected_objects;

    double IMG_AREA = (double)img.rows * img.cols;
    double IMG_DIAG = std::sqrt((double)img.rows * img.rows + (double)img.cols * img.cols);
    double MIN_AREA = IMG_AREA * PipelineConfig::minAreaRatio;
    double PERIM_LARGE = IMG_DIAG * 1;
    double SPIKE_MAX_LEN = IMG_DIAG * 0.1;
    int TOLERANCE_PX = static_cast<int>(IMG_DIAG * 0.03);

    std::vector<cv::Rect> seen_boxes;
    bool isFlatTarget = false;

    for (const auto& cnt : contours)
    {
        double area = cv::contourArea(cnt);
        double perimeter = cv::arcLength(cnt, true);
        if (area < MIN_AREA || area > IMG_AREA * 0.85) continue;

        double compactness = (area > 0) ? (perimeter * perimeter) / area : 9999.0;
        if (compactness > 500 && area < IMG_AREA * 0.05) continue;

        cv::Rect br = cv::boundingRect(cnt);
        double aspect_ratio = (br.height > 0) ? (double)br.width / br.height : 0.0;
        if (aspect_ratio < 0.15 || aspect_ratio > 6.0) continue;

        Contour hull_pts;
        cv::convexHull(cnt, hull_pts);
        double solidity = (cv::contourArea(hull_pts) > 0) ? area / cv::contourArea(hull_pts) : 0.0;
        if (solidity < 0.45) continue;

        bool is_dup = false;
        for (const auto& sb : seen_boxes) {
            if (std::abs(br.x - sb.x) < TOLERANCE_PX && std::abs(br.y - sb.y) < TOLERANCE_PX &&
                std::abs(br.width - sb.width) < TOLERANCE_PX && std::abs(br.height - sb.height) < TOLERANCE_PX) {
                is_dup = true; break;
            }
        }
        if (is_dup) continue;
        seen_boxes.push_back(br);

        bool is_circle = false;
        cv::RotatedRect ell_rect;
        double ew = 0, eh = 0;

        if (cnt.size() >= 5) {
            ell_rect = cv::fitEllipse(cnt);
            ew = ell_rect.size.width; eh = ell_rect.size.height;
            double circle_ratio = (eh > 0) ? ew / eh : 0.0;
            cv::RotatedRect min_r = cv::minAreaRect(cnt);
            double rect_extent = (min_r.size.width * min_r.size.height > 0) ? area / (min_r.size.width * min_r.size.height) : 0.0;

            Contour test_approx;
            cv::approxPolyDP(cnt, test_approx, 0.01 * cv::arcLength(cnt, true), true);
            if (circle_ratio > 0.65 && circle_ratio < 1.50 && rect_extent >= 0.72 && rect_extent <= 0.82 &&
                test_approx.size() > 8 && solidity > 0.95) is_circle = true;
        }

        std::string shape_type;
        Contour contour_to_add;

        if (is_circle) {
            double cr = (eh > 0) ? ew / eh : 0.0;
            shape_type = (cr > 0.90 && cr < 1.10) ? "circle" : "ellipse";
            cv::ellipse2Poly(cv::Point((int)ell_rect.center.x, (int)ell_rect.center.y),
                cv::Size((int)ew / 2, (int)eh / 2), (int)ell_rect.angle, 0, 360, 4, contour_to_add);
        }
        else {
            if (perimeter > PERIM_LARGE) {
                cv::Mat iso = cv::Mat::zeros(threshHealed.size(), threshHealed.type());
                cv::drawContours(iso, ContourVec{ cnt }, 0, 255, -1);

                int ks = std::max(3, std::min(15, (int)(perimeter * 0.003) + ((int)(perimeter * 0.003) % 2 == 0 ? 1 : 0)));
                cv::Mat ke = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks, ks));
                cv::Mat cleaned;
                cv::morphologyEx(iso, cleaned, cv::MORPH_OPEN, ke);
                cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, ke);

                ContourVec smooth_cnts;
                cv::findContours(cleaned, smooth_cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                if (!smooth_cnts.empty()) {
                    auto& main_s = *std::max_element(smooth_cnts.begin(), smooth_cnts.end(),
                        [](const Contour& a, const Contour& b) { return cv::contourArea(a) > cv::contourArea(b); });

                    double largest_so_far = 0;
                    for (const auto& uc : unique_contours) largest_so_far = std::max(largest_so_far, cv::contourArea(uc));
                    bool is_outer_torso = (cv::contourArea(cnt) >= largest_so_far);

                    Contour approx;
                    cv::approxPolyDP(cnt, approx, 0.005 * cv::arcLength(cnt, true), true);

                    if (mode == ThreshMode::DESKEW) {
                        if (!is_outer_torso) {
                            Contour ch; cv::convexHull(cnt, ch);
                            cv::approxPolyDP(ch, contour_to_add, 0.01 * cv::arcLength(ch, true), true);
                        }
                        else {
                            cv::Rect xb = cv::boundingRect(main_s);
                            int idx = 0;
                            while (idx < (int)approx.size() && (int)approx.size() > 4) {
                                cv::Point2f pc = approx[idx];
                                cv::Point2f pp = approx[(idx - 1 + approx.size()) % approx.size()];
                                cv::Point2f pn = approx[(idx + 1) % approx.size()];
                                if (ptNorm(pc, pp) < SPIKE_MAX_LEN && ptNorm(pc, pn) < SPIKE_MAX_LEN) {
                                    double v1x = pp.x - pc.x, v1y = pp.y - pc.y, v2x = pn.x - pc.x, v2y = pn.y - pc.y;
                                    double cos_a = std::max(-1.0, std::min(1.0, (v1x * v2x + v1y * v2y) / (vecNorm(v1x, v1y) * vecNorm(v2x, v2y) + 1e-6)));
                                    if (std::acos(cos_a) * 180.0 / CV_PI < 115 && !(pc.y < xb.y + xb.height * 0.40)) {
                                        approx.erase(approx.begin() + idx); continue;
                                    }
                                }
                                idx++;
                            }
                            contour_to_add = approx;
                        }
                    }
                    else {
                        if (!is_outer_torso) {
                            Contour dense = ContourUtils::densifyContour(cnt, 2.0);
                            auto [trueCorners, dummy] = ContourUtils::findTrueCorners(dense, true);

                            std::vector<bool> protected_(dense.size(), false);
                            for (int i = 0; i < (int)dense.size(); i++) {
                                for (auto& c : trueCorners) {
                                    if (std::hypot(dense[i].x - c.x, dense[i].y - c.y) < PipelineConfig::cornerProtectRadius) {
                                        protected_[i] = true;
                                        break;
                                    }
                                }
                            }

                            auto detectHorseshoe = [&](const Contour& cnt_in) -> bool {
                                cv::Rect bb = cv::boundingRect(cnt_in);
                                double topThreshold = bb.y + bb.height * 0.35;
                                std::vector<double> topXs;
                                for (auto& p : cnt_in) {
                                    if (p.y < topThreshold) topXs.push_back(p.x);
                                }
                                if (topXs.size() < 10) return false;

                                double runningMax = topXs[0];
                                double runningMin = topXs[0];
                                for (int i = 1; i < (int)topXs.size(); i++) {
                                    runningMax = std::max(runningMax, topXs[i]);
                                    runningMin = std::min(runningMin, topXs[i]);
                                }

                                int reversals = 0;
                                double prevDx = 0;
                                for (int i = 1; i < (int)topXs.size(); i++) {
                                    double dx = topXs[i] - topXs[i - 1];
                                    if (std::abs(dx) < 2.0) continue;
                                    if (prevDx != 0 && ((dx > 0) != (prevDx > 0)))
                                        reversals++;
                                    prevDx = dx;
                                }
                                return (reversals > 6);
                                };

                            Contour cntclean = ContourUtils::removeBulletHoles(dense, protected_, PipelineConfig::octagonMaxHoleWidth, PipelineConfig::octagonMinHoleDepth, PipelineConfig::octagonWindowSize);

                            bool isHorseshoe = detectHorseshoe(cntclean);
                            std::cout << "  [HORSESHOE] detected=" << isHorseshoe << "\n";
                            std::cout << "[FLATTARGET] is flat target ==" << isFlatTarget << "\n";

                            Contour baseContour;
                            if (isHorseshoe || isFlatTarget)
                            {
                                std::cout << "  [OCTAGON] Using trueCorners directly\n";
                                for (auto& c : trueCorners)
                                    baseContour.push_back(cv::Point((int)std::round(c.x), (int)std::round(c.y)));
                            }
                            else
                            {
                                cv::approxPolyDP(cntclean, baseContour, 0.5, true);
                            }

                            cv::Point2d centroidOff(0, 0);
                            for (auto& p : baseContour) { centroidOff.x += p.x; centroidOff.y += p.y; }
                            centroidOff.x /= baseContour.size();
                            centroidOff.y /= baseContour.size();

                            for (int i = 0; i < (int)baseContour.size(); i++) {
                                int prev = (i - 1 + baseContour.size()) % baseContour.size();
                                int next = (i + 1) % baseContour.size();
                                double ex = baseContour[next].x - baseContour[prev].x;
                                double ey = baseContour[next].y - baseContour[prev].y;
                                double elen = std::hypot(ex, ey);
                                if (elen < 1e-6) { contour_to_add.push_back(baseContour[i]); continue; }
                                double nx = -ey / elen, ny = ex / elen;
                                if (nx * (centroidOff.x - baseContour[i].x) + ny * (centroidOff.y - baseContour[i].y) > 0) { nx = -nx; ny = -ny; }
                                contour_to_add.push_back(cv::Point((int)std::round(baseContour[i].x + nx * PipelineConfig::inkOffset), (int)std::round(baseContour[i].y + ny * PipelineConfig::inkOffset)));
                            }
                        }
                        else {
                            Contour dense = ContourUtils::densifyContour(cnt, 2.0);
                            auto [trueCorners, flatFlag] = ContourUtils::findTrueCorners(dense, false);
                            isFlatTarget = flatFlag;
                            std::vector<bool> protected_(dense.size(), false);
                            cv::Rect bb = cv::boundingRect(dense);
                            for (int i = 0; i < (int)dense.size(); i++) {
                                if (dense[i].y < bb.y + bb.height * 0.35) protected_[i] = true;
                                else {
                                    for (auto& c : trueCorners) {
                                        if (std::hypot(dense[i].x - c.x, dense[i].y - c.y) < PipelineConfig::cornerProtectRadius) {
                                            protected_[i] = true; break;
                                        }
                                    }
                                }
                            }
                            Contour cnt_clean = ContourUtils::removeBulletHoles(dense, protected_, PipelineConfig::torsoMaxHoleWidth, PipelineConfig::torsoMinHoleDepth, PipelineConfig::torsoWindowSize);
                            cv::approxPolyDP(cnt_clean, contour_to_add, 0.5, true);
                        }
                    }
                }
            }
            else {
                Contour ch; cv::convexHull(cnt, ch);
                cv::approxPolyDP(ch, contour_to_add, 0.006 * cv::arcLength(ch, true), true);
            }
        }

        if (contour_to_add.empty()) continue;

        if (!is_circle) {
            Contour fa; cv::approxPolyDP(contour_to_add, fa, 0.02 * cv::arcLength(contour_to_add, true), true);
            int fv = static_cast<int>(fa.size());
            if (fv == 3) continue;
            shape_type = (fv <= 8) ? "polygon (" + std::to_string(fv) + " pts)" : "polygon";
        }

        detected_objects.push_back({ shape_type, (int)unique_contours.size(), cv::contourArea(contour_to_add) });
        unique_contours.push_back(contour_to_add);
    }

    // Post-loop cleanup
    if (!unique_contours.empty()) {
        double max_area = 0;
        for (const auto& o : detected_objects) max_area = std::max(max_area, o.area);
        ContourVec fc; ObjVec fo;
        for (size_t i = 0; i < unique_contours.size(); i++) {
            Contour hull_pts; cv::convexHull(unique_contours[i], hull_pts);
            if (detected_objects[i].area >= max_area * 0.02 && (detected_objects[i].area / cv::contourArea(hull_pts)) > 0.40) {
                fc.push_back(unique_contours[i]); fo.push_back(detected_objects[i]);
            }
        }
        unique_contours = fc; detected_objects = fo;
    }

    // Inner zone erasure
    if (detected_objects.size() >= 2) {
        std::vector<int> order(detected_objects.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return detected_objects[a].area > detected_objects[b].area; });

        std::vector<int> indices_to_erase = { order[1] };
        for (size_t i = 2; i < order.size(); i++) {
            if (detected_objects[order[i]].type.find("rectangle") != std::string::npos || detected_objects[order[i]].type.find("4 pts") != std::string::npos)
                indices_to_erase.push_back(order[i]);
        }
        std::sort(indices_to_erase.rbegin(), indices_to_erase.rend());
        for (int idx : indices_to_erase) {
            unique_contours.erase(unique_contours.begin() + idx);
            detected_objects.erase(detected_objects.begin() + idx);
        }
        for (int i = 0; i < (int)detected_objects.size(); i++) detected_objects[i].index = i;
    }

    return { unique_contours, detected_objects };
}

// =============================================================================
// 1. MAIN DETECTION ORCHESTRATOR
// =============================================================================
std::pair<ContourVec, ObjVec> detection(const cv::Mat& img, ThreshMode mode = ThreshMode::DETECTION)
{
    // 1. Generate core masks (adaptive thresh, polarity checks, color zones)
    ProcessedMasks masks = preprocessImage(img, mode);

    double imgArea = (double)img.rows * img.cols;

    // 2. Extract and filter initial contours based on image mode
    ContourVec initialContours = extractContours(masks, imgArea, mode);

    // 3. Classify shapes, remove noise, handle IDPA scoring hierarchy
    return classifyShapes(initialContours, masks.thresh_healed, img, mode);
}
   
// =============================================================================
// 2. DESKEWING FUNCTION
// =============================================================================
cv::Mat deskewing(const cv::Mat& img, double angle_threshold = 10.0)
{
    cv::Mat corrected = img.clone();
    bool res_needed = false;

    Contour cnt_a, approx_a;
    cv::Rect bbox_a;

    if (!GeometryUtils::find_torso_polygon(corrected, cnt_a, approx_a, bbox_a))
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
        std::sort(edge_angles.begin(), edge_angles.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        double norm = std::fmod(edge_angles[0].second, 180.0);
        if (norm < 0) norm += 180.0;
        if (norm > 90) norm -= 180.0;
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
        if (GeometryUtils::find_torso_polygon(corrected, cnt_z, approx_z, bbox_z))
        {
            int iw = corrected.cols, ih = corrected.rows;
            double area_ratio = (double)(bbox_z.width * bbox_z.height) / (iw * ih);
            std::cout << " | fill=" << area_ratio;

            double target_fill_w = (double)bbox_z.width / iw;
            double target_fill_h = (double)bbox_z.height / ih;
            std::cout << " | target_fill_w=" << target_fill_w << " | target_fill_h=" << target_fill_h;

            bool needs_zoom = (area_ratio < 0.7) && ((target_fill_w < 0.80) || (target_fill_h < 0.80));

            if (needs_zoom)
            {
                double scale_w = (iw * 0.80) / bbox_z.width;
                double scale_h = (ih * 0.80) / bbox_z.height;
                double SCALE = std::min({ scale_w, scale_h, 3.0 });

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
                if (cw > 0 && ch > 0 && x1 >= 0 && y1 >= 0 && x1 + cw <= corrected.cols && y1 + ch <= corrected.rows)
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
        // STAGE C — PERSPECTIVE 
        // ════════════════════════════════════════════════
        const int MAX_ITERS = 4;
        const double SCORE_IMPROVE_MIN = 0.1;
        double prev_score = -1.0;
        int persp_passes = 0;

        std::cout << "\n--- Deskewing Loop Started ---";
        cv::Mat gray_check;
        cv::cvtColor(corrected, gray_check, cv::COLOR_BGR2GRAY);
        cv::Rect center_check(corrected.cols / 4, corrected.rows / 4, corrected.cols / 2, corrected.rows / 2);
        double check_mean = cv::mean(gray_check(center_check))[0];
        bool is_dark_bg = (check_mean < 100.0);

        {
            std::vector<cv::Point2f> hough_corners;
            bool hough_ok = is_dark_bg && GeometryUtils::hough_perspective_recovery(corrected, hough_corners, true);

            if (hough_ok)
            {
                auto [needs_fix, score] = GeometryUtils::quad_needs_perspective_fix(hough_corners);
                std::cout << "\nPre-pass [hough on clean image]: Score = " << std::fixed << std::setprecision(3) << score << " ";

                if (needs_fix)
                {
                    cv::Point2f C = (hough_corners[0] + hough_corners[1] + hough_corners[2] + hough_corners[3]) * 0.25f;
                    const float HOUGH_PAD = 1.20f;
                    std::vector<cv::Point2f> expanded(4);
                    for (int i = 0; i < 4; i++) expanded[i] = C + HOUGH_PAD * (hough_corners[i] - C);

                    bool sane = true;
                    for (const auto& p : expanded) {
                        if (p.x < -corrected.cols * 0.30f || p.x > corrected.cols * 1.30f ||
                            p.y < -corrected.rows * 0.30f || p.y > corrected.rows * 1.30f) {
                            sane = false; break;
                        }
                    }

                    if (sane)
                    {
                        cv::Mat candidate = GeometryUtils::warp_from_corners(corrected, expanded);
                        if (candidate.cols <= corrected.cols * 2 && candidate.rows <= corrected.rows * 2)
                        {
                            corrected = candidate;
                            res_needed = true;
                            prev_score = score;
                            persp_passes++;
                            std::cout << "-> pre-pass applied (scale=" << HOUGH_PAD << ")";
                        }
                        else std::cout << "-> warp too large, skipped";
                    }
                    else std::cout << "-> corners unsafe, skipped";
                }
                else
                {
                    std::cout << "-> no fix needed";
                    goto stage_c_done;
                }
            }
            else std::cout << "\nPre-pass [hough]: failed to find lines, proceeding to loop";
        }

        for (int iter = 0; iter < MAX_ITERS; iter++)
        {
            std::cout << "\nPass " << (iter + 1) << ": ";
            auto [uc, objs] = detection(corrected, ThreshMode::DESKEW);

            Contour* target_cnt = nullptr;
            for (int i = 0; i < (int)objs.size(); i++) {
                if (objs[i].type.find("8 pts") != std::string::npos) {
                    target_cnt = &uc[i]; break;
                }
            }

            if (!target_cnt) {
                double img_area = (double)corrected.rows * corrected.cols;
                for (int i = 0; i < (int)objs.size(); i++) {
                    const std::string& t = objs[i].type;
                    bool is_poly = t.find("polygon") != std::string::npos || t.find("pts") != std::string::npos;
                    if (is_poly && objs[i].area > img_area * 0.05) {
                        if (target_cnt == nullptr || std::abs((int)uc[i].size() - 8) < std::abs((int)target_cnt->size() - 8)) {
                            target_cnt = &uc[i];
                        }
                    }
                }
            }

            std::vector<cv::Point2f> corners;
            bool corners_ok = false;

            if (target_cnt && !uc.empty()) {
                corners_ok = GeometryUtils::get_octagon_corners(*target_cnt, corners);
                if (corners_ok) std::cout << "[contour path] ";
            }

            if (!corners_ok && is_dark_bg) {
                std::cout << "[hough fallback] ";
                corners_ok = GeometryUtils::hough_perspective_recovery(corrected, corners, false);
            }

            if (!corners_ok) {
                std::cout << "Both contour and Hough methods failed. Stopping.";
                break;
            }

            auto [needs_fix, score] = GeometryUtils::quad_needs_perspective_fix(corners);
            std::cout << "Score = " << std::fixed << std::setprecision(3) << score << " ";

            if (!needs_fix) {
                std::cout << "-> Perfect alignment achieved!"; break;
            }

            if (prev_score >= 0) {
                double improvement = prev_score - score;
                if (improvement < 0) {
                    std::cout << "-> Score WORSENED by " << std::abs(improvement) << ". Stopping."; break;
                }
                if (improvement < SCORE_IMPROVE_MIN) {
                    std::cout << "-> Improvement plateaued (diff = " << improvement << "). Stopping."; break;
                }
            }

            int best_i = 0;
            for (int i = 1; i < (int)objs.size(); i++) {
                if (objs[i].area > objs[best_i].area) best_i = i;
            }

            std::vector<cv::Point2f> expanded;
            double k_used;

            if (!uc.empty()) {
                auto [exp2, k2] = GeometryUtils::expand_quad_to_fit_contour(corners, uc[best_i], 0.03, 2.5);
                expanded = exp2;
                k_used = k2;
            }
            else {
                cv::Point2f C = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
                expanded.resize(4);
                for (int i = 0; i < 4; i++) expanded[i] = C + 1.15f * (corners[i] - C);
                k_used = 1.15;
            }

            cv::Mat warped_candidate = GeometryUtils::warp_from_corners(corrected, expanded);
            if (warped_candidate.cols > corrected.cols * 2 || warped_candidate.rows > corrected.rows * 2) {
                std::cout << "-> Warp output too large. Stopping."; break;
            }

            corrected = warped_candidate;
            res_needed = true;
            prev_score = score;
            persp_passes++;
            std::cout << "-> pass applied (scale=" << k_used << ")";
        }

    stage_c_done:
        std::cout << "\n--- Deskewing Finished | Total successful passes: " << persp_passes << " ---\n";

        // ════════════════════════════════════════════════
        // STAGE D — RESOLUTION NORMALIZATION
        // ════════════════════════════════════════════════
        const int TARGET_W = 850;
        const int TARGET_H = 1550;

        if (corrected.cols != TARGET_W || corrected.rows != TARGET_H) {
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
// 3. PLOTTING FUNCTION (AUTOMATED BACKGROUND STREAMING VERSION)
// =============================================================================
// Maps over processed images, performs final detection, and saves the graphical panels.
int plotting(const std::vector<cv::Mat>& deskewed_imgs, const std::vector<std::string>& image_paths)
{
    // Local nested lambda for creating side-by-side display images
    auto make_side_by_side = [](const cv::Mat& left, const cv::Mat& right,
        const std::string& title_left,
        const std::string& title_right,
        int font_height = 28) -> cv::Mat {
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

    // Generate visualizations for every deskewed and raw image pairing.
    for (int img_idx = 0; img_idx < (int)deskewed_imgs.size(); img_idx++)
    {
        const std::string& raw_path = image_paths[img_idx];
        const cv::Mat& raw_img = deskewed_imgs[img_idx];
        cv::Mat raw_original = cv::imread(raw_path);

        // Run final detection
        auto [unique_contours, detected_objects] = detection(raw_img, ThreshMode::DETECTION);

        int n_cnt = (int)unique_contours.size();

        // Build display overlay
        cv::Mat overlay = raw_img.clone();
        int thickness = std::max(1, (int)std::round(3.0 * raw_img.cols / 850.0));
        cv::drawContours(overlay, unique_contours, -1, cv::Scalar(0, 0, 255), thickness);

        // Extract the actual filename from the full path to make titles clean
        std::filesystem::path p(raw_path);
        std::string filename_only = p.filename().string();

        std::string t_left = "Original: " + filename_only;
        std::string t_right = std::to_string(n_cnt) + " Target Zones Detected";

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

        // Combine left and right sides
        cv::Mat panel = make_side_by_side(raw_disp, overlay, t_left, t_right);

        // Resize down to standard width if it's too massive
        const int DISPLAY_W = 1200;
        if (panel.cols > DISPLAY_W)
        {
            double sf2 = (double)DISPLAY_W / panel.cols;
            cv::resize(panel, panel, {}, sf2, sf2, cv::INTER_AREA);
        }

        // Generate dynamic sequential filename based on original file name or index
        std::ostringstream fname;
        fname << "result_" << std::setfill('0') << std::setw(3) << (img_idx + 1) << "_" << p.stem().string() << ".jpg";

        // Save directly to disk
        cv::imwrite(fname.str(), panel);

        std::cout << "   -> Saved visual report: " << fname.str() << "\n";
    }

    // Return immediately to main loop without creating windows or blocking execution
    return 0;
}
// =============================================================================
// MAIN ENTRY POINT (STREAMING PIPELINE VERSION)
// =============================================================================
int main(int argc, char** argv)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    // Default folder path
    std::string folder_path = "C:\\Users\\PMLS\\Downloads\\IDPA Target Detection\\test images";

    if (argc > 1)
    {
        folder_path = argv[1];
    }

    std::vector<std::string> image_paths;

    // 1. Gather all file paths (Very lightweight, just strings)
    try
    {
        for (const auto& entry : std::filesystem::directory_iterator(folder_path))
        {
            if (entry.is_regular_file())
            {
                std::string ext = entry.path().extension().string();
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

    std::cout << "Found " << image_paths.size() << " images to process.\n\n";

    // 2. Stream images through the pipeline IN PARALLEL
    std::atomic<int> processed_count{ 0 };
    int total_images = (int)image_paths.size();
    std::mutex log_mutex; // Prevents console output from scrambling

    std::for_each(std::execution::par, image_paths.begin(), image_paths.end(), [&](const std::string& path)
        {
            // Step A: Load the raw image into RAM
            cv::Mat raw = cv::imread(path);
            if (raw.empty())
            {
                std::lock_guard<std::mutex> lock(log_mutex);
                std::cerr << "   Error: Could not load image: " << path << "\n";
                return; // equivalent to 'continue' in a standard loop
            }

            // Step B: Run deskewing on the single image
            cv::Mat deskewed = deskewing(raw, 10.0);

            // Step C: Wrap single items into temporary vectors for plotting compatibility
            std::vector<cv::Mat> temp_deskewed_vec = { deskewed };
            std::vector<std::string> temp_path_vec = { path };

            // Step D: Run detection, build UI, and display/save the single panel
            plotting(temp_deskewed_vec, temp_path_vec);

            // Safely log progress from this thread
            int current = ++processed_count;
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "[" << current << "/" << total_images << "] Processed: " << path << "\n";
        });
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n\n";
    std::cout << "All images processed successfully.\n";
    return 0;
}