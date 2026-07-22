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

Contour removeBulletHoles(const Contour& input,
    const std::vector<bool>& protected_,
    double maxHoleWidth = 50.0,
    double minHoleDepth = 0.5,
    int    windowSize = 8)
{
    int n = (int)input.size();
    if (n < 24) return input;

    cv::Point2d C(0, 0);
    for (auto& p : input) { C.x += p.x; C.y += p.y; }
    C.x /= n; C.y /= n;

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
        double chordLen = std::hypot(chordX, chordY);
        if (chordLen < 1e-6) continue;

        // Perpendicular to chord
        double nx = -chordY / chordLen;
        double ny = chordX / chordLen;

        double midX = (prev.x + next.x) * 0.5;
        double midY = (prev.y + next.y) * 0.5;
        double d = (curr.x - midX) * nx + (curr.y - midY) * ny;

        // Positive cross = left of travel = concave for CW contour
        // Negative cross = right of travel = convex for CW contour
        double travelX = next.x - prev.x;
        double travelY = next.y - prev.y;
        double toX = curr.x - prev.x;
        double toY = curr.y - prev.y;
        double cross = travelX * toY - travelY * toX;

        depth[i] = std::abs(d);
        // For OpenCV contours (typically CW): 
        // cross < 0 means point bulges to the right of travel = outward spike
        // cross > 0 means point bulges to the left of travel = inward dent
        // We want to remove BOTH so just use depth regardless of sign
        concave[i] = (cross > 0);  // still tracked but no longer used as filter
    }
    std::vector<bool> remove(n, false);
    //step 2
    for (int i = 0; i < n; i++)
    {
        if (depth[i] < minHoleDepth) continue;
        if (remove[i]) continue;
        if (protected_[i]) continue;  

        double peakDepth = depth[i];
        int leftEdge = i;
        int rightEdge = i;

        for (int k = 1; k < windowSize * 3; k++)
        {
            int iTest = (i - k + n) % n;
            if (depth[iTest] < peakDepth * 0.50 || protected_[iTest])
            {
                leftEdge = iTest;
                break;
            }
            leftEdge = iTest;
        }
        for (int k = 1; k < windowSize * 3; k++)
        {
            int iTest = (i + k) % n;
            if (depth[iTest] < peakDepth * 0.50 || protected_[iTest])
            {
                rightEdge = iTest;
                break;
            }
            rightEdge = iTest;
        }

        cv::Point2d pL(input[leftEdge].x, input[leftEdge].y);
        cv::Point2d pR(input[rightEdge].x, input[rightEdge].y);
        double holeWidth = std::hypot(pR.x - pL.x, pR.y - pL.y);
        int spanPoints = (rightEdge - leftEdge + n) % n;

        if (protected_[i]) continue;
        if (spanPoints < 10) continue;
        if (holeWidth > maxHoleWidth) continue;

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
    result.reserve(n);
    for (int i = 0; i < n; i++)
        if (!remove[i])
            result.push_back(input[i]);

    return result.empty() ? input : result;
}



Contour densifyContour(const Contour& cnt, double maxSpacing = 2.0)
{
    Contour dense;
    int n = (int)cnt.size();
    for (int i = 0; i < n; i++)
    {
        cv::Point2d p1(cnt[i].x, cnt[i].y);
        cv::Point2d p2(cnt[(i + 1) % n].x, cnt[(i + 1) % n].y);
        dense.push_back(cnt[i]);

        double dist = std::hypot(p2.x - p1.x, p2.y - p1.y);
        int steps = (int)(dist / maxSpacing);
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

                std::cout << "  [STRAIGHTNESS] leftVar=" << leftVar
                    << " rightVar=" << rightVar << "\n";

                // In straightness check — increase variance threshold
                isFlatTarget = (leftVar < 5000.0 && rightVar < 5000.0);  // was 100.0
            }
        }

        // Part 2: Add 2 neck corners via deepest convexity defect
        // — upgraded to line-fit intersection, same as convex corners
        std::vector<int> hullIdx;
        cv::convexHull(dense, hullIdx, false, false);
        std::vector<cv::Vec4i> defects;
        cv::convexityDefects(dense, hullIdx, defects);

        if (!defects.empty())
        {
            std::cout << "  [DEFECT COUNT] total=" << defects.size() << "\n";
            std::cout << "  [DENSE SIZE] " << dense.size() << "\n";
            for (int d = 0; d < std::min(5, (int)defects.size()); d++)
            {
                int si = defects[d][0], ei = defects[d][1];
                int rawDepth = defects[d][3];
                std::cout << "  [DEFECT PRE-SORT " << d << "] rawdepth=" << rawDepth
                    << " start=(" << dense[si].x << "," << dense[si].y
                    << ") end=(" << dense[ei].x << "," << dense[ei].y << ")\n";
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

            std::cout << "  [CORNERS] Neck corners (raw): ("
                << neckLeft.x << "," << neckLeft.y << ") and ("
                << neckRight.x << "," << neckRight.y << ")\n";

        }

        std::cout << "  [STRAIGHTNESS] isFlatTarget=" << isFlatTarget << "\n";
        return { corners, isFlatTarget };
    }
}
// =============================================================================
// 1. DETECTION FUNCTION
// =============================================================================
// Handles image preprocessing and shape classification based on given mode.
std::pair<ContourVec, ObjVec> detection(const cv::Mat& img, ThreshMode mode = ThreshMode::DETECTION)
{
    ContourVec unique_contours;
    ObjVec detected_objects;

    // ── Preprocessing ────────────────────────────────────────────────────────
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

    // Correct outward boundary bias from blur+adaptiveThreshold.
    // Only fires when center_mean was dark (dark-interior targets like
    // black-on-white) — these are the only ones showing the gap.
    // All other images (colored zones, tan, photographed targets) have
    // center_mean >= 80 and skip this entirely.
    cv::Mat thresh_healed_outer = thresh_healed.clone();
    // For dark-bg images, supplement the outer mask with Canny edges
    // to handle uneven vignetting that makes adaptive threshold inconsistent
    // around the outer boundary.
    if (center_mean < 80.0)
    {
        // For dark-bg/white-line targets, flood-fill from the image corner
        // to find the true outer boundary of the torso silhouette.
        // This is robust to vignetting, double-edges, and junction artifacts
        // because it works on connected regions, not gradient magnitude.

        // Step 1: Get a clean binary of the original (pre-flip) gray
        // where the dark torso silhouette is foreground.
        cv::Mat gray_orig;
        cv::cvtColor(img, gray_orig, cv::COLOR_BGR2GRAY);
        cv::Mat bin_orig;
        cv::threshold(gray_orig, bin_orig, 0, 255,
            cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        // Step 2: Flood-fill from top-left corner to mark the background.
        // We need a border-padded copy for floodFill.
        cv::Mat filled = bin_orig.clone();
        cv::Mat padded;
        cv::copyMakeBorder(filled, padded, 1, 1, 1, 1,
            cv::BORDER_CONSTANT, 0);
        cv::floodFill(padded, cv::Point(0, 0), 128);
        // Remove the border padding
        cv::Mat filled_crop = padded(
            cv::Rect(1, 1, filled.cols, filled.rows));

        // Step 3: Build outer mask — pixels that are NOT background (128)
        // and NOT already foreground in bin_orig = the torso interior fill.
        // Combined with the original binary gives us the full silhouette.
        cv::Mat outer_mask = cv::Mat::zeros(bin_orig.size(), CV_8UC1);
        for (int y = 0; y < filled_crop.rows; y++)
        {
            for (int x = 0; x < filled_crop.cols; x++)
            {
                uchar fc = filled_crop.at<uchar>(y, x);
                uchar bc = bin_orig.at<uchar>(y, x);
                // Keep pixels that are part of the torso (not background)
                if (fc != 128 || bc == 255)
                    outer_mask.at<uchar>(y, x) = 255;
            }
        }

        // Step 4: Light morphological close to seal any thin gaps in
        // the white stroke, then erode back to true edge center.
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

    // ── FILTER: keep only SOLID color blobs, drop thin outline strokes ────────
    // A filled zone (e.g. red center circle) has high solidity and reasonable
    // area. A traced boundary line picked up by color thresholding is thin and
    // has very low solidity relative to its bounding box - exclude those so
    // they never inflate the outer torso contour via fusion.
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
            // Solid filled blob: high solidity AND reasonably fills its bbox.
            // Thin outline strokes fail fill_ratio even if solidity looks ok.
            if (solidity > 0.85 && fill_ratio > 0.35)
            {
                cv::drawContours(color_mask_filtered, ContourVec{ c }, 0, 255, -1);
            }
        }
    }

    // ── MASK FUSION — DETECTION MODE ONLY ────────────────────────────────────
    // During DESKEW, keep thresh_healed clean for octagon/polygon finding
    if (mode == ThreshMode::DETECTION)
    {
        cv::bitwise_or(thresh_healed, color_mask_filtered, thresh_healed);
    }
    // ── INJECT SAVED DARK CIRCLES — DETECTION MODE ONLY ──────────────────────
    // During DESKEW, dark blob injection breaks find_torso_polygon
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
    // Extract contours from both masks and merge.
    // Large contours come from thresh_healed_outer (bias-corrected boundary).
    // Small/inner contours come from thresh_healed (uncorrected, preserves thin lines).
    double IMG_AREA_MERGE = (double)img.rows * img.cols;
    double LARGE_THRESH = IMG_AREA_MERGE * 0.45; // shapes > 5% of image = large

    ContourVec contours_outer, contours_inner;
    std::vector<cv::Vec4i> hier_outer, hier_inner;
    cv::findContours(thresh_healed_outer, contours_outer, hier_outer,
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(thresh_healed, contours_inner, hier_inner,
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Take large shapes from corrected mask, small shapes from original mask
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

    // Color contours appended only during final detection, not deskew
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
    double IMG_DIAG = std::sqrt((double)img.rows * img.rows + (double)img.cols * img.cols);
    double MIN_AREA = IMG_AREA * 0.002;
    double PERIM_LARGE = IMG_DIAG * 1;
    double SPIKE_MAX_LEN = IMG_DIAG * 0.1;
    int TOLERANCE_PX = static_cast<int>(IMG_DIAG * 0.03);
    

    std::vector<cv::Rect> seen_boxes;
    bool isFlatTarget = false;
    // Iterate through found contours to identify shapes.
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

        // Deduplication
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
                // ── Large shape: isolate, clean, smooth ───────────────────────
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

                if (!smooth_cnts.empty())
                {
                    auto& main_s = *std::max_element(smooth_cnts.begin(), smooth_cnts.end(),
                        [](const Contour& a, const Contour& b)
                        { return cv::contourArea(a) > cv::contourArea(b); });

                    double cur_area = cv::contourArea(cnt);
                    double largest_so_far = 0;
                    for (const auto& uc : unique_contours)
                        largest_so_far = std::max(largest_so_far, cv::contourArea(uc));

                    bool is_outer_torso = (cur_area >= largest_so_far);

                    double eps = 0.005 * cv::arcLength(cnt, true);
                    Contour approx;
                    cv::approxPolyDP(cnt, approx, eps, true);
                    


                    if (mode == ThreshMode::DESKEW) {
                        if (!is_outer_torso) {
                            // Deskew only needs rough shape — keep existing fast path
                            Contour ch;
                            cv::convexHull(cnt, ch);
                            double eps = 0.01 * cv::arcLength(ch, true);
                            cv::approxPolyDP(ch, contour_to_add, eps, true);
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
                    else {
                        if (!is_outer_torso) {
                            Contour dense = densifyContour(cnt, 2.0);

                            auto [trueCorners, dummy] = findTrueCorners(dense, true);         // octagon
                            std::cout << "  [CORNERS] Found Octagon ==>" << trueCorners.size() << " true corners\n";

                            // Mark contour points near any true corner as protected
                            std::vector<bool> protected_(dense.size(), false);
                            double protectRadius = 50.0; // px radius around each corner to protect

                            for (int i = 0; i < (int)dense.size(); i++)
                            {
                                for (auto& c : trueCorners)
                                {
                                    double dist = std::hypot(dense[i].x - c.x, dense[i].y - c.y);
                                    if (dist < protectRadius)
                                    {
                                        protected_[i] = true;
                                        break;
                                    }
                                }
                            }
                            cv::Rect bb = cv::boundingRect(dense);
                            std::cout << "  [OCTAGON BB] x=" << bb.x << " y=" << bb.y
                                << " w=" << bb.width << " h=" << bb.height << "\n";
                            // Pass protected_ into removeBulletHoles
                            // After removeBulletHoles:
                            // Detect horseshoe: check if contour self-intersects at the top
                            auto detectHorseshoe = [&](const Contour& cnt) -> bool
                                {
                                    cv::Rect bb = cv::boundingRect(cnt);
                                    double topThreshold = bb.y + bb.height * 0.35;

                                    // Collect X coordinates of points in top region in contour order
                                    std::vector<double> topXs;
                                    for (auto& p : cnt)
                                        if (p.y < topThreshold)
                                            topXs.push_back(p.x);

                                    if (topXs.size() < 10) return false;

                                    // Find the maximum X drop followed by rise in the sequence
                                    // A normal octagon top goes left-to-right (or right-to-left) monotonically
                                    // A horseshoe goes one direction, drops back, then continues — big reversal
                                    double maxReversal = 0;
                                    double runningMax = topXs[0];
                                    double runningMin = topXs[0];

                                    for (int i = 1; i < (int)topXs.size(); i++)
                                    {
                                        runningMax = std::max(runningMax, topXs[i]);
                                        runningMin = std::min(runningMin, topXs[i]);
                                    }
                                    double totalRange = runningMax - runningMin;

                                    // Check for direction reversals — count times X changes direction significantly
                                    int reversals = 0;
                                    double prevDx = 0;
                                    for (int i = 1; i < (int)topXs.size(); i++)
                                    {
                                        double dx = topXs[i] - topXs[i - 1];
                                        if (std::abs(dx) < 2.0) continue; // ignore tiny jitter
                                        if (prevDx != 0 && ((dx > 0) != (prevDx > 0)))
                                            reversals++;
                                        prevDx = dx;
                                    }

                                    std::cout << "  [HORSESHOE] topPts=" << topXs.size()
                                        << " totalRange=" << totalRange
                                        << " reversals=" << reversals << "\n";

                                    // A horseshoe has many reversals in the top region
                                    // A clean octagon top has 1 reversal at most (left diagonal → right diagonal meeting at top)
                                    return (reversals > 6);
                                };

                            Contour cntclean = removeBulletHoles(dense, protected_, 60.0, 0.1, 20);
                            bool isHorseshoe = detectHorseshoe(cntclean);
                            std::cout << "  [HORSESHOE] detected=" << isHorseshoe << "\n";
                            std::cout << "[FLATTARGET] is flat target ==" << isFlatTarget << "\n";
                            Contour baseContour;
                            if (isHorseshoe || isFlatTarget)
                            {
                                std::cout << "USING CONVEX HULL"<< "\n";
                                for (auto& c : trueCorners)
                                    baseContour.push_back(cv::Point((int)std::round(c.x), (int)std::round(c.y)));
                            }
                            else
                            {
                                cv::approxPolyDP(cntclean, baseContour, 0.5, true);
                            }
                            

                            // Push every contour point outward by ink thickness
                            // so all edges sit on the outer black guide line, not the inner edge
                            cv::Point2d centroidOff(0, 0);
                            for (auto& p : baseContour) { centroidOff.x += p.x; centroidOff.y += p.y; }
                            centroidOff.x /= baseContour.size();
                            centroidOff.y /= baseContour.size();

                            double inkOffset = 4.0; // tune this
                            Contour cntoffset;
                            for (int i = 0; i < (int)baseContour.size(); i++)
                            {
                                int prev = (i - 1 + (int)baseContour.size()) % (int)baseContour.size();
                                int next = (i + 1) % (int)baseContour.size();

                                double ex = baseContour[next].x - baseContour[prev].x;
                                double ey = baseContour[next].y - baseContour[prev].y;
                                double elen = std::hypot(ex, ey);
                                if (elen < 1e-6) { cntoffset.push_back(baseContour[i]); continue; }

                                double nx = -ey / elen, ny = ex / elen;

                                // Flip to point away from centroid (outward)
                                double toCx = centroidOff.x - baseContour[i].x;
                                double toCy = centroidOff.y - baseContour[i].y;
                                if (nx * toCx + ny * toCy > 0) { nx = -nx; ny = -ny; }

                                cntoffset.push_back(cv::Point(
                                    (int)std::round(baseContour[i].x + nx * inkOffset),
                                    (int)std::round(baseContour[i].y + ny * inkOffset)
                                ));
                            }

                            std::cout << "  [BHR] Octagon after approx=" << baseContour.size() << "\n";
                            contour_to_add = cntoffset;
                        }
                        else {
                            // Densify first so bullet holes and curves have enough points
                            Contour dense = densifyContour(cnt, 2.0);  // 1 point every 2px

                            auto [trueCorners, flatFlag] = findTrueCorners(dense, false);
                            isFlatTarget = flatFlag;

                            std::cout << "  [CORNERS] Found Torso ==>" << trueCorners.size() << " true corners\n";
                            for (auto& c : trueCorners)
                                std::cout << "    corner at (" << c.x << ", " << c.y << ")\n";

                            // Mark contour points near any true corner as protected
                            // Mark contour points near any true corner as protected
                            std::vector<bool> protected_(dense.size(), false);
                            double protectRadius = 50.0;  // was 50.0 — increase to protect full corner transitions

                            for (int i = 0; i < (int)dense.size(); i++)
                            {
                                for (auto& c : trueCorners)
                                {
                                    double dist = std::hypot(dense[i].x - c.x, dense[i].y - c.y);
                                    if (dist < protectRadius)
                                    {
                                        protected_[i] = true;
                                        break;
                                    }
                                }

                                // Also protect any point in the upper 35% of the contour bounding box
                                // This saves the shoulder-to-neck diagonal edges on both sides
                                cv::Rect bb = cv::boundingRect(dense);
                                if (dense[i].y < bb.y + bb.height * 0.35)
                                    protected_[i] = true;
                            }

                            // Pass protected_ into removeBulletHoles
                            Contour cnt_clean = removeBulletHoles(dense, protected_, 50.0, 0.5, 50);
                            double eps = 0.5;  // fixed 0.5px — not proportional, not aggressive
                            Contour result;
                            cv::approxPolyDP(cnt_clean, result, eps, true);
                            std::cout << "  [BHR]Torso After approx=" << result.size() << "\n";


                            contour_to_add = result;
                        }
                    }
                    
                }
            }
            else
            {
                // ── Small shape: hull + approx ────────────────────────────────
                Contour ch;
                cv::convexHull(cnt, ch);
                double eps = 0.006 * cv::arcLength(ch, true);
                cv::approxPolyDP(ch, contour_to_add, eps, true);
            }
        }

        if (contour_to_add.empty())
            continue;

        // ── Final shape classification ─────────────────────────────────────────
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
    
    // ── Post-loop noise removal ───────────────────────────────────────────────
    if (!unique_contours.empty())
    {
        double max_area = 0;
        for (const auto& o : detected_objects)
            max_area = std::max(max_area, o.area);
        double min_rel = max_area * 0.02;

        ContourVec fc;
        ObjVec fo;
        for (size_t i = 0; i < unique_contours.size(); i++)
        {
            // 1. Calculate metrics for THIS specific shape [i]
            double current_area = cv::contourArea(unique_contours[i]);

            Contour hull_pts;
            cv::convexHull(unique_contours[i], hull_pts);
            double hull_area = cv::contourArea(hull_pts);

            double solidity = (hull_area > 0) ? (current_area / hull_area) : 0.0;

            // 2. Apply filtering logic
            if (detected_objects[i].area >= min_rel && solidity > 0.40)
            {
                fc.push_back(unique_contours[i]);
                fo.push_back(detected_objects[i]);
            }
        }
        unique_contours = fc;
        detected_objects = fo;
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

            // Stage B — existing ratio check PLUS minimum size check
            double area_ratio = (double)(bbox_z.width * bbox_z.height) / (iw * ih);
            std::cout << " | fill=" << area_ratio;
           

            // Zoom if target is too small relative to image OR too small in absolute pixels
            // TARGET_W=850, TARGET_H=1550 is our standard output size
            // A properly filled target should be at least 60% of output dimensions
            double target_fill_w = (double)bbox_z.width / iw;
            double target_fill_h = (double)bbox_z.height / ih;
            std::cout << " | target_fill_w=" << target_fill_w;
            std::cout << " | target_fill_h=" << target_fill_h;
            bool needs_zoom = (area_ratio < 0.7) &&
                ((target_fill_w < 0.80) ||
                (target_fill_h < 0.80));

            if (needs_zoom)
            {
                // Compute scale needed to bring target to ~80% of frame
                double scale_w = (iw * 0.80) / bbox_z.width;
                double scale_h = (ih * 0.80) / bbox_z.height;
                double SCALE = std::min(scale_w, scale_h);  // use the limiting dimension
                SCALE = std::min(SCALE, 3.0);  // cap at 3x to avoid absurd zoom

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
        std::cout << "\n--- Deskewing Finished | Total successful passes: "
            << persp_passes << " ---\n";
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
int plotting(const std::vector<cv::Mat>& deskewed_imgs, const std::vector<std::string>& image_paths)
{
    // Local nested lambda for creating side-by-side display images
    auto make_side_by_side = [](const cv::Mat& left, const cv::Mat& right,
        const std::string& title_left,
        const std::string& title_right,
        int font_height = 28) -> cv::Mat
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
        cv::Mat raw_original = cv::imread(raw_path);

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

        cv::Mat panel = make_side_by_side(raw_disp, overlay, t_left, t_right);

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
        deskewed_imgs.push_back(deskewing(raw, 10.0));
    }

    plotting(deskewed_imgs, image_paths);
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total execution time: " << std::fixed << std::setprecision(2) << elapsed << " seconds\n\n";
    cv::waitKey(0);
    return 0;
}