/*
 * KF-GINS ROS2 node: GNSS/INS integrated navigation
 */

#include <cmath>
#include <deque>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "common/angle.h"
#include "common/earth.h"
#include "common/rotation.h"
#include "kf-gins/gi_engine.h"

namespace {

struct ImuSample {
    IMU imu;
    rclcpp::Time stamp;
};

struct GnssSample {
    GNSS gnss;
    rclcpp::Time stamp;
};

Eigen::Vector3d nedToEnu(const Eigen::Vector3d &ned) {
    return {ned.y(), ned.x(), -ned.z()};
}

Eigen::Matrix3d nedToEnuMatrix() {
    Eigen::Matrix3d r;
    r << 0, 1, 0, 1, 0, 0, 0, 0, -1;
    return r;
}

Eigen::Matrix3d frdToFluMatrix() {
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
    r(1, 1)           = -1.0;
    r(2, 2)           = -1.0;
    return r;
}

double wrapAngleRad(double a) {
    while (a > M_PI) {
        a -= 2.0 * M_PI;
    }
    while (a < -M_PI) {
        a += 2.0 * M_PI;
    }
    return a;
}

Eigen::Vector3d quatToRpy(const Eigen::Quaterniond &q_in) {
    Eigen::Quaterniond q = q_in.normalized();
    const double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
    const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    const double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    const double roll      = std::atan2(sinr_cosp, cosr_cosp);
    const double sinp      = std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0);
    const double pitch     = std::asin(sinp);
    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double yaw       = std::atan2(siny_cosp, cosy_cosp);
    return {roll, pitch, yaw};
}

Eigen::Quaterniond rpyToQuat(double roll, double pitch, double yaw) {
    Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
    return Eigen::Quaterniond(rz * ry * rx);
}

bool parseFilterScheme(const std::string &value, FilterScheme &scheme) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (s == "ESKF" || s == "EKF") {
        scheme = FilterScheme::ESKF;
        return true;
    }
    if (s == "UKF") {
        scheme = FilterScheme::UKF;
        return true;
    }
    if (s == "SR_UKF" || s == "SRUKF") {
        scheme = FilterScheme::SR_UKF;
        return true;
    }
    if (s == "ADAPTIVE_UKF" || s == "AUKF") {
        scheme = FilterScheme::ADAPTIVE_UKF;
        return true;
    }
    if (s == "ROBUST_UKF" || s == "RUKF" || s == "RAUKF") {
        scheme = FilterScheme::ROBUST_UKF;
        return true;
    }
    return false;
}

enum class HeadingMode : int {
    NONE = 0,
    SINGLE_GNSS_COURSE = 1,
    DUAL_GNSS_HEADING = 2,
};

const char *headingModeName(HeadingMode mode) {
    switch (mode) {
    case HeadingMode::NONE:
        return "none";
    case HeadingMode::SINGLE_GNSS_COURSE:
        return "single_gnss_course";
    case HeadingMode::DUAL_GNSS_HEADING:
        return "dual_gnss_heading";
    default:
        return "unknown";
    }
}

bool parseHeadingMode(const std::string &value, HeadingMode &mode) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "none") {
        mode = HeadingMode::NONE;
        return true;
    }
    if (s == "single_gnss_course" || s == "single" || s == "course") {
        mode = HeadingMode::SINGLE_GNSS_COURSE;
        return true;
    }
    if (s == "dual_gnss_heading" || s == "dual" || s == "dual_gnss") {
        mode = HeadingMode::DUAL_GNSS_HEADING;
        return true;
    }
    return false;
}

enum class AutoInitYawMode : int {
    CONFIG_ONLY = 0,
    GNSS_COURSE_OR_CONFIG = 1,
};

bool parseAutoInitYawMode(const std::string &value, AutoInitYawMode &mode) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "config_only" || s == "config") {
        mode = AutoInitYawMode::CONFIG_ONLY;
        return true;
    }
    if (s == "gnss_course_or_config" || s == "gnss_course" || s == "course_or_config") {
        mode = AutoInitYawMode::GNSS_COURSE_OR_CONFIG;
        return true;
    }
    return false;
}

bool parseGnssPosMeasMode(const std::string &value, GnssPosMeasMode &mode) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "xyz" || s == "3d" || s == "full") {
        mode = GnssPosMeasMode::XYZ;
        return true;
    }
    if (s == "xy" || s == "2d" || s == "horizontal") {
        mode = GnssPosMeasMode::XY;
        return true;
    }
    return false;
}

} // namespace

class KfGinsNode : public rclcpp::Node {
public:
    KfGinsNode()
        : rclcpp::Node("kf_gins_node") {
        imu_topic_                = declare_parameter<std::string>("imu_topic", "/imu/data");
        gps_topic_                = declare_parameter<std::string>("gps_topic", "/gps/fix");
        odom_topic_               = declare_parameter<std::string>("odom_topic", "/kf_gins/odom");
        odom_fused_topic_         = declare_parameter<std::string>("odom_fused_topic", "/kf_gins/odom_fused");
        path_topic_               = declare_parameter<std::string>("path_topic", "/kf_gins/path");
        navsat_topic_             = declare_parameter<std::string>("navsat_topic", "/kf_gins/gps/fix");
        nis_topic_                = declare_parameter<std::string>("nis_topic", "/kf_gins/nis");
        frame_id_                 = declare_parameter<std::string>("frame_id", "map");
        path_frame_id_            = declare_parameter<std::string>("path_frame_id", "");
        navsat_frame_id_          = declare_parameter<std::string>("navsat_frame_id", "");
        child_frame_id_           = declare_parameter<std::string>("child_frame_id", "base_link");
        publish_tf_               = declare_parameter<bool>("publish_tf", true);
        publish_path_             = declare_parameter<bool>("publish_path", true);
        publish_navsat_           = declare_parameter<bool>("publish_navsat", true);
        publish_nis_              = declare_parameter<bool>("publish_nis", false);
        path_max_size_            = declare_parameter<int>("path_max_size", 2000);
        path_incremental_output_  = declare_parameter<bool>("path_incremental_output", false);
        odom_publish_rate_        = declare_parameter<double>("odom_publish_rate", 0.0);
        navsat_publish_rate_      = declare_parameter<double>("navsat_publish_rate", -1.0);
        path_publish_rate_        = declare_parameter<double>("path_publish_rate", 10.0);
        input_stale_timeout_      = declare_parameter<double>("input_stale_timeout", 0.5);
        output_enu_               = declare_parameter<bool>("output_enu", true);
        imu_in_flu_               = declare_parameter<bool>("imu_in_flu", true);
        odom_orientation_flu_     = declare_parameter<bool>("odom_orientation_flu", true);
        imu_rate_                 = declare_parameter<double>("imu_rate", 200.0);
        max_imu_dt_               = declare_parameter<double>("max_imu_dt", 0.1);
        max_queue_size_           = declare_parameter<int>("max_queue_size", 2000);
        use_navsatfix_covariance_ = declare_parameter<bool>("use_navsatfix_covariance", true);
        gnss_std_                 = declare_parameter<std::vector<double>>("gnss_std", {1.0, 1.0, 2.0});
        gnss_pre_gate_enable_     = declare_parameter<bool>("gnss_pre_gate_enable", false);
        gnss_pre_gate_max_hstd_   = declare_parameter<double>("gnss_pre_gate_max_hstd", -1.0);
        gnss_pre_gate_max_vstd_   = declare_parameter<double>("gnss_pre_gate_max_vstd", -1.0);
        gnss_pre_gate_max_speed_  = declare_parameter<double>("gnss_pre_gate_max_speed", -1.0);
        gnss_pre_gate_min_dt_     = declare_parameter<double>("gnss_pre_gate_min_dt", 0.2);
        gnss_update_mode_name_    = declare_parameter<std::string>("gnss_update_mode", "xyz");
        gnss_nis_gate_mode_name_  = declare_parameter<std::string>("gnss_nis_gate_mode", "xyz");
        gnss_time_offset_sec_     = declare_parameter<double>("gnss_time_offset_sec", 0.0);
        start_time_               = declare_parameter<double>("start_time", 0.0);
        end_time_                 = declare_parameter<double>("end_time", -1.0);
        use_absolute_time_        = declare_parameter<bool>("use_absolute_time", false);
        max_imu_ahead_            = declare_parameter<double>("max_imu_ahead", 0.0);
        use_wall_time_stamp_      = declare_parameter<bool>("use_wall_time_stamp", true);
        filter_scheme_name_       = declare_parameter<std::string>("filter_scheme", "ESKF");
        declare_parameter<double>("ukf_alpha", 1.0e-3);
        declare_parameter<double>("ukf_beta", 2.0);
        declare_parameter<double>("ukf_kappa", 0.0);
        declare_parameter<double>("adaptive_q_scale", 1.0);
        declare_parameter<double>("adaptive_r_scale", 1.0);
        declare_parameter<double>("robust_huber_delta", 2.5);
        declare_parameter<bool>("gnss_nis_gate_enable", false);
        declare_parameter<double>("gnss_nis_gate_threshold", 11.34);
        heading_mode_name_        = declare_parameter<std::string>("heading_mode", "none");
        heading_fusion_enable_    = declare_parameter<bool>("heading_fusion_enable", false);
        heading_fallback_to_imu_  = declare_parameter<bool>("heading_fallback_to_imu", true);
        single_heading_min_speed_ = declare_parameter<double>("single_heading_min_speed", 1.0);
        single_heading_std_deg_   = declare_parameter<double>("single_heading_std_deg", 5.0);
        dual_heading_std_deg_     = declare_parameter<double>("dual_heading_std_deg", 1.0);
        dual_heading_quality_gate_= declare_parameter<bool>("dual_heading_quality_gate", true);
        dual_heading_nis_gate_    = declare_parameter<bool>("dual_heading_nis_gate", true);
        dual_heading_topic_       = declare_parameter<std::string>("dual_heading_topic", "/gnss/dual_heading");
        single_heading_blend_gain_= declare_parameter<double>("single_heading_blend_gain", 0.2);
        auto_init_enable_               = declare_parameter<bool>("auto_init_enable", false);
        auto_init_pos_only_             = declare_parameter<bool>("auto_init_pos_only", false);
        auto_init_gnss_window_sec_      = declare_parameter<double>("auto_init_gnss_window_sec", 8.0);
        auto_init_imu_window_sec_       = declare_parameter<double>("auto_init_imu_window_sec", 3.0);
        auto_init_min_gnss_samples_     = declare_parameter<int>("auto_init_min_gnss_samples", 5);
        auto_init_min_imu_samples_      = declare_parameter<int>("auto_init_min_imu_samples", 100);
        auto_init_use_gnss_median_      = declare_parameter<bool>("auto_init_use_gnss_median", true);
        auto_init_reject_gnss_outlier_  = declare_parameter<bool>("auto_init_reject_gnss_outlier", true);
        auto_init_gnss_outlier_sigma_   = declare_parameter<double>("auto_init_gnss_outlier_sigma", 3.5);
        auto_init_require_static_rp_    = declare_parameter<bool>("auto_init_require_static_rp", false);
        auto_init_max_acc_std_          = declare_parameter<double>("auto_init_max_acc_std", 0.5);
        auto_init_max_gyro_std_         = declare_parameter<double>("auto_init_max_gyro_std", 0.1);
        auto_init_yaw_mode_name_        = declare_parameter<std::string>("auto_init_yaw_mode", "gnss_course_or_config");
        auto_init_min_speed_for_yaw_    = declare_parameter<double>("auto_init_min_speed_for_yaw", 1.0);
        gnss_stats_log_period_sec_      = declare_parameter<double>("gnss_stats_log_period_sec", 5.0);

        if (gnss_std_.size() != 3) {
            RCLCPP_WARN(get_logger(), "Parameter 'gnss_std' must be 3 elements. Using default [1,1,2].");
            gnss_std_ = {1.0, 1.0, 2.0};
        }

        declareKfGinsParams();
        if (!loadOptionsFromParams()) {
            RCLCPP_ERROR(get_logger(), "Failed to load KF-GINS parameters from ROS2 params.");
            throw std::runtime_error("Failed to load params");
        }

        if (!parseAutoInitYawMode(auto_init_yaw_mode_name_, auto_init_yaw_mode_)) {
            RCLCPP_WARN(get_logger(),
                        "Unknown auto_init_yaw_mode '%s'. Fallback to gnss_course_or_config.",
                        auto_init_yaw_mode_name_.c_str());
            auto_init_yaw_mode_ = AutoInitYawMode::GNSS_COURSE_OR_CONFIG;
        }
        if (!auto_init_enable_) {
            origin_blh_ = options_.initstate.pos;
            giengine_   = std::make_unique<GIEngine>(options_);
        } else {
            RCLCPP_INFO(get_logger(),
                        "Auto initialization enabled: GNSS window %.1fs / IMU window %.1fs (min GNSS=%d, min IMU=%d)",
                        auto_init_gnss_window_sec_, auto_init_imu_window_sec_, auto_init_min_gnss_samples_,
                        auto_init_min_imu_samples_);
            if (auto_init_pos_only_) {
                RCLCPP_INFO(get_logger(), "Auto-init mode: position-only (keep configured attitude and IMU errors)");
            }
        }

        if (!parseHeadingMode(heading_mode_name_, heading_mode_)) {
            RCLCPP_WARN(get_logger(),
                        "Unknown heading_mode '%s'. Fallback to 'none'. Supported: none, "
                        "single_gnss_course, dual_gnss_heading",
                        heading_mode_name_.c_str());
            heading_mode_ = HeadingMode::NONE;
        }
        if (heading_fusion_enable_) {
            RCLCPP_INFO(get_logger(),
                        "Heading fusion enabled (mode=%s). NOTE: parameter skeleton is ready; heading update logic is "
                        "not wired into GIEngine yet.",
                        headingModeName(heading_mode_));
        }

        odom_pub_       = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
        odom_fused_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_fused_topic_, 10);
        if (publish_path_) {
            path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, 10);
            if (path_frame_id_.empty()) {
                path_frame_id_ = frame_id_;
            }
            path_msg_.header.frame_id = path_frame_id_;
        }
        if (publish_navsat_) {
            navsat_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>(navsat_topic_, 10);
            if (navsat_frame_id_.empty()) {
                navsat_frame_id_ = frame_id_;
            }
        }
        if (publish_nis_) {
            nis_pub_ = create_publisher<std_msgs::msg::Float64>(nis_topic_, 10);
        }
        if (publish_tf_) {
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
        }

        sensor_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_cb_group_  = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        if (odom_publish_rate_ > 0.0) {
            const auto period = std::chrono::duration<double>(1.0 / odom_publish_rate_);
            odom_timer_ = create_wall_timer(period, std::bind(&KfGinsNode::publishOdomTimer, this), timer_cb_group_);
        }
        if (publish_navsat_ && navsat_publish_rate_ > 0.0) {
            const auto period = std::chrono::duration<double>(1.0 / navsat_publish_rate_);
            navsat_timer_ =
                create_wall_timer(period, std::bind(&KfGinsNode::publishNavsatTimer, this), timer_cb_group_);
        }
        if (publish_path_ && path_publish_rate_ > 0.0) {
            const auto period = std::chrono::duration<double>(1.0 / path_publish_rate_);
            path_timer_ = create_wall_timer(period, std::bind(&KfGinsNode::publishPathTimer, this), timer_cb_group_);
        }
        if (gnss_stats_log_period_sec_ > 0.0) {
            const auto period = std::chrono::duration<double>(gnss_stats_log_period_sec_);
            gnss_stats_timer_ =
                create_wall_timer(period, std::bind(&KfGinsNode::logGnssStatsTimer, this), timer_cb_group_);
        }

        rclcpp::SubscriptionOptions sensor_sub_options;
        sensor_sub_options.callback_group = sensor_cb_group_;
        imu_sub_                          = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::SensorDataQoS(), std::bind(&KfGinsNode::imuCallback, this, std::placeholders::_1),
            sensor_sub_options);
        gnss_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
            gps_topic_, rclcpp::SensorDataQoS(), std::bind(&KfGinsNode::gnssCallback, this, std::placeholders::_1),
            sensor_sub_options);
    }

private:
    void declareKfGinsParams() {
        declare_parameter<std::vector<double>>("initpos", {30.5, 114.0, 20.0});
        declare_parameter<std::vector<double>>("initvel", {0.0, 0.0, 0.0});
        declare_parameter<std::vector<double>>("initatt", {0.0, 0.0, 0.0});
        declare_parameter<std::vector<double>>("initgyrbias", {0.0, 0.0, 0.0});
        declare_parameter<std::vector<double>>("initaccbias", {0.0, 0.0, 0.0});
        declare_parameter<std::vector<double>>("initgyrscale", {0.0, 0.0, 0.0});
        declare_parameter<std::vector<double>>("initaccscale", {0.0, 0.0, 0.0});
        declare_parameter<std::vector<double>>("initposstd", {0.1, 0.1, 0.2});
        declare_parameter<std::vector<double>>("initvelstd", {0.05, 0.05, 0.05});
        declare_parameter<std::vector<double>>("initattstd", {0.5, 0.5, 1.0});
        const std::vector<double> nan3 = {std::nan(""), std::nan(""), std::nan("")};
        declare_parameter<std::vector<double>>("initbgstd", nan3);
        declare_parameter<std::vector<double>>("initbastd", nan3);
        declare_parameter<std::vector<double>>("initsgstd", nan3);
        declare_parameter<std::vector<double>>("initsastd", nan3);
        declare_parameter<std::vector<double>>("imunoise.arw", {0.24, 0.24, 0.24});
        declare_parameter<std::vector<double>>("imunoise.vrw", {0.24, 0.24, 0.24});
        declare_parameter<std::vector<double>>("imunoise.gbstd", {50.0, 50.0, 50.0});
        declare_parameter<std::vector<double>>("imunoise.abstd", {250.0, 250.0, 250.0});
        declare_parameter<std::vector<double>>("imunoise.gsstd", {1000.0, 1000.0, 1000.0});
        declare_parameter<std::vector<double>>("imunoise.asstd", {1000.0, 1000.0, 1000.0});
        declare_parameter<double>("imunoise.corrtime", 1.0);
        declare_parameter<std::vector<double>>("antlever", {0.136, -0.301, -0.184});
    }

    bool getVecParam(const std::string &name, std::vector<double> &out, size_t expected, bool required) {
        if (!get_parameter(name, out) || out.size() != expected) {
            if (required) {
                RCLCPP_ERROR(get_logger(), "Parameter '%s' must be %zu elements.", name.c_str(), expected);
            }
            return false;
        }
        return true;
    }

    bool loadOptionsFromParams() {
        std::vector<double> vec1, vec2, vec3, vec4, vec5, vec6;

        if (!getVecParam("initpos", vec1, 3, true) || !getVecParam("initvel", vec2, 3, true) ||
            !getVecParam("initatt", vec3, 3, true)) {
            return false;
        }
        for (int i = 0; i < 3; i++) {
            options_.initstate.pos[i]   = vec1[i] * D2R;
            options_.initstate.vel[i]   = vec2[i];
            options_.initstate.euler[i] = vec3[i] * D2R;
        }
        options_.initstate.pos[2] *= R2D;

        if (!getVecParam("initgyrbias", vec1, 3, true) || !getVecParam("initaccbias", vec2, 3, true) ||
            !getVecParam("initgyrscale", vec3, 3, true) || !getVecParam("initaccscale", vec4, 3, true)) {
            return false;
        }
        for (int i = 0; i < 3; i++) {
            options_.initstate.imuerror.gyrbias[i]  = vec1[i] * D2R / 3600.0;
            options_.initstate.imuerror.accbias[i]  = vec2[i] * 1e-5;
            options_.initstate.imuerror.gyrscale[i] = vec3[i] * 1e-6;
            options_.initstate.imuerror.accscale[i] = vec4[i] * 1e-6;
        }

        if (!getVecParam("initposstd", vec1, 3, true) || !getVecParam("initvelstd", vec2, 3, true) ||
            !getVecParam("initattstd", vec3, 3, true)) {
            return false;
        }
        for (int i = 0; i < 3; i++) {
            options_.initstate_std.pos[i]   = vec1[i];
            options_.initstate_std.vel[i]   = vec2[i];
            options_.initstate_std.euler[i] = vec3[i] * D2R;
        }

        if (!getVecParam("imunoise.arw", vec1, 3, true) || !getVecParam("imunoise.vrw", vec2, 3, true) ||
            !getVecParam("imunoise.gbstd", vec3, 3, true) || !getVecParam("imunoise.abstd", vec4, 3, true) ||
            !getVecParam("imunoise.gsstd", vec5, 3, true) || !getVecParam("imunoise.asstd", vec6, 3, true)) {
            return false;
        }
        options_.imunoise.corr_time = get_parameter("imunoise.corrtime").as_double();
        for (int i = 0; i < 3; i++) {
            options_.imunoise.gyr_arw[i]      = vec1[i];
            options_.imunoise.acc_vrw[i]      = vec2[i];
            options_.imunoise.gyrbias_std[i]  = vec3[i];
            options_.imunoise.accbias_std[i]  = vec4[i];
            options_.imunoise.gyrscale_std[i] = vec5[i];
            options_.imunoise.accscale_std[i] = vec6[i];
        }

        std::vector<double> initbgstd, initbastd, initsgstd, initsastd;
        get_parameter("initbgstd", initbgstd);
        get_parameter("initbastd", initbastd);
        get_parameter("initsgstd", initsgstd);
        get_parameter("initsastd", initsastd);

        auto valid3 = [](const std::vector<double> &v) {
            return v.size() == 3 && std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
        };

        if (!valid3(initbgstd)) {
            initbgstd = {options_.imunoise.gyrbias_std.x(), options_.imunoise.gyrbias_std.y(),
                         options_.imunoise.gyrbias_std.z()};
        }
        if (!valid3(initbastd)) {
            initbastd = {options_.imunoise.accbias_std.x(), options_.imunoise.accbias_std.y(),
                         options_.imunoise.accbias_std.z()};
        }
        if (!valid3(initsgstd)) {
            initsgstd = {options_.imunoise.gyrscale_std.x(), options_.imunoise.gyrscale_std.y(),
                         options_.imunoise.gyrscale_std.z()};
        }
        if (!valid3(initsastd)) {
            initsastd = {options_.imunoise.accscale_std.x(), options_.imunoise.accscale_std.y(),
                         options_.imunoise.accscale_std.z()};
        }
        for (int i = 0; i < 3; i++) {
            options_.initstate_std.imuerror.gyrbias[i]  = initbgstd[i] * D2R / 3600.0;
            options_.initstate_std.imuerror.accbias[i]  = initbastd[i] * 1e-5;
            options_.initstate_std.imuerror.gyrscale[i] = initsgstd[i] * 1e-6;
            options_.initstate_std.imuerror.accscale[i] = initsastd[i] * 1e-6;
        }

        options_.imunoise.gyr_arw *= (D2R / 60.0);
        options_.imunoise.acc_vrw /= 60.0;
        options_.imunoise.gyrbias_std *= (D2R / 3600.0);
        options_.imunoise.accbias_std *= 1e-5;
        options_.imunoise.gyrscale_std *= 1e-6;
        options_.imunoise.accscale_std *= 1e-6;
        options_.imunoise.corr_time *= 3600;

        if (!getVecParam("antlever", vec1, 3, true)) {
            return false;
        }
        options_.antlever = Eigen::Vector3d(vec1.data());

        if (!parseFilterScheme(filter_scheme_name_, options_.filter_scheme)) {
            RCLCPP_WARN(get_logger(),
                        "Unknown filter_scheme '%s'. Fallback to ESKF. Supported: ESKF, UKF, SR_UKF, "
                        "ADAPTIVE_UKF, ROBUST_UKF",
                        filter_scheme_name_.c_str());
            options_.filter_scheme = FilterScheme::ESKF;
        }
        options_.ukf_alpha = get_parameter("ukf_alpha").as_double();
        options_.ukf_beta  = get_parameter("ukf_beta").as_double();
        options_.ukf_kappa = get_parameter("ukf_kappa").as_double();
        if (!parseGnssPosMeasMode(gnss_update_mode_name_, options_.gnss_update_mode)) {
            RCLCPP_WARN(get_logger(), "Unknown gnss_update_mode '%s'. Fallback to xyz.",
                        gnss_update_mode_name_.c_str());
            options_.gnss_update_mode = GnssPosMeasMode::XYZ;
        }
        if (!parseGnssPosMeasMode(gnss_nis_gate_mode_name_, options_.gnss_nis_gate_mode)) {
            RCLCPP_WARN(get_logger(), "Unknown gnss_nis_gate_mode '%s'. Fallback to xyz.",
                        gnss_nis_gate_mode_name_.c_str());
            options_.gnss_nis_gate_mode = GnssPosMeasMode::XYZ;
        }
        if (options_.gnss_nis_gate_mode != options_.gnss_update_mode) {
            RCLCPP_WARN(get_logger(),
                        "gnss_nis_gate_mode (%s) != gnss_update_mode (%s). Current implementation uses the update "
                        "measurement dimension for NIS gating; forcing gate mode to match update mode.",
                        gnssPosMeasModeName(options_.gnss_nis_gate_mode),
                        gnssPosMeasModeName(options_.gnss_update_mode));
            options_.gnss_nis_gate_mode = options_.gnss_update_mode;
        }
        options_.gnss_nis_gate_enable = get_parameter("gnss_nis_gate_enable").as_bool();
        options_.gnss_nis_gate_threshold = get_parameter("gnss_nis_gate_threshold").as_double();
        if (options_.ukf_alpha <= 0.0) {
            RCLCPP_WARN(get_logger(), "ukf_alpha must be > 0. Fallback to 1e-3");
            options_.ukf_alpha = 1.0e-3;
        }
        if (options_.gnss_nis_gate_threshold <= 0.0) {
            RCLCPP_WARN(get_logger(), "gnss_nis_gate_threshold must be > 0. Fallback to 11.34");
            options_.gnss_nis_gate_threshold = 11.34;
        }

        return true;
    }
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        const double msg_time = rclcpp::Time(msg->header.stamp).seconds();
        if (use_absolute_time_) {
            if (msg_time < start_time_) {
                return;
            }
            if (end_time_ > 0.0 && msg_time > end_time_) {
                return;
            }
        } else {
            if (!base_time_set_) {
                base_time_     = msg_time;
                base_time_set_ = true;
            }
            const double rel_time = msg_time - base_time_;
            if (rel_time < start_time_) {
                return;
            }
            if (end_time_ > 0.0 && rel_time > end_time_) {
                return;
            }
        }
        ImuSample sample;
        sample.stamp    = msg->header.stamp;
        sample.imu.time = rclcpp::Time(sample.stamp).seconds();

        double dt = 0.0;
        if (last_imu_time_ > 0.0) {
            dt = sample.imu.time - last_imu_time_;
        }
        if (dt <= 0.0 || dt > max_imu_dt_) {
            dt = 1.0 / imu_rate_;
        }
        sample.imu.dt  = dt;
        last_imu_time_ = sample.imu.time;

        Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        Eigen::Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

        if (imu_in_flu_) {
            // Convert ROS FLU to FRD expected by KF-GINS
            gyro.y() = -gyro.y();
            gyro.z() = -gyro.z();
            acc.y()  = -acc.y();
            acc.z()  = -acc.z();
        }

        sample.imu.dtheta = gyro * dt;
        sample.imu.dvel   = acc * dt;
        sample.imu.odovel = 0.0;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_input_wall_time_ = this->now();
            have_input_           = true;
        }

        imu_queue_.push_back(sample);
        trimQueue(imu_queue_);
        process();
    }

    void gnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        ++gnss_rx_count_;
        const double msg_time = rclcpp::Time(msg->header.stamp).seconds();
        const double fused_time = msg_time + gnss_time_offset_sec_;
        if (use_absolute_time_) {
            if (fused_time < start_time_) {
                ++gnss_time_window_drop_count_;
                return;
            }
            if (end_time_ > 0.0 && fused_time > end_time_) {
                ++gnss_time_window_drop_count_;
                return;
            }
        } else {
            if (!base_time_set_) {
                base_time_     = fused_time;
                base_time_set_ = true;
            }
            const double rel_time = fused_time - base_time_;
            if (rel_time < start_time_) {
                ++gnss_time_window_drop_count_;
                return;
            }
            if (end_time_ > 0.0 && rel_time > end_time_) {
                ++gnss_time_window_drop_count_;
                return;
            }
        }
        if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
            ++gnss_status_drop_count_;
            return;
        }

        GnssSample sample;
        sample.stamp       = msg->header.stamp;
        sample.gnss.time   = fused_time;
        sample.gnss.blh[0] = msg->latitude * D2R;
        sample.gnss.blh[1] = msg->longitude * D2R;
        sample.gnss.blh[2] = msg->altitude;

        bool used_cov = false;
        if (use_navsatfix_covariance_ &&
            msg->position_covariance_type != sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN) {
            const double xx = msg->position_covariance[0];
            const double yy = msg->position_covariance[4];
            const double zz = msg->position_covariance[8];
            if (xx > 0.0 && yy > 0.0 && zz > 0.0) {
                sample.gnss.std = Eigen::Vector3d(std::sqrt(xx), std::sqrt(yy), std::sqrt(zz));
                used_cov        = true;
            }
        }
        if (!used_cov) {
            sample.gnss.std = Eigen::Vector3d(gnss_std_[0], gnss_std_[1], gnss_std_[2]);
        }
        if (gnss_pre_gate_enable_ && shouldRejectGnssPreGate(sample, used_cov)) {
            ++gnss_pregate_drop_count_;
            return;
        }

        // Single-antenna course heading estimate from consecutive GNSS positions (ENU yaw, CCW from East).
        if (heading_fusion_enable_ && heading_mode_ == HeadingMode::SINGLE_GNSS_COURSE) {
            const Eigen::Vector3d curr_blh = sample.gnss.blh;
            if (have_prev_gnss_fix_for_course_) {
                const double dt = sample.gnss.time - prev_gnss_fix_time_for_course_;
                if (dt > 1e-3) {
                    const Eigen::Vector3d d_ned = Earth::global2local(prev_gnss_blh_for_course_, curr_blh);
                    const Eigen::Vector3d d_enu = nedToEnu(d_ned);
                    const double speed_xy       = std::hypot(d_enu.x(), d_enu.y()) / dt;
                    if (speed_xy >= single_heading_min_speed_ && std::hypot(d_enu.x(), d_enu.y()) > 1e-3) {
                        const double yaw_enu = std::atan2(d_enu.y(), d_enu.x());
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        latest_single_gnss_course_yaw_enu_ = yaw_enu;
                        latest_single_gnss_course_speed_   = speed_xy;
                        latest_single_gnss_course_stamp_   = sample.stamp;
                        have_single_gnss_course_heading_   = true;
                    }
                }
            }
            prev_gnss_blh_for_course_      = curr_blh;
            prev_gnss_fix_time_for_course_ = sample.gnss.time;
            have_prev_gnss_fix_for_course_ = true;
        }

        have_prev_gnss_for_pregate_ = true;
        prev_gnss_blh_for_pregate_   = sample.gnss.blh;
        prev_gnss_time_for_pregate_  = sample.gnss.time;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_input_wall_time_ = this->now();
            have_input_           = true;
        }

        gnss_queue_.push_back(sample);
        ++gnss_enqueue_count_;
        latest_gnss_time_ = sample.gnss.time;
        trimQueue(gnss_queue_);
        process();
    }

    template <typename T> void trimQueue(std::deque<T> &q) {
        while (static_cast<int>(q.size()) > max_queue_size_) {
            q.pop_front();
        }
    }

    void process() {
        if (!initialized_) {
            tryInitialize();
            return;
        }

        while (!imu_queue_.empty()) {
            if (latest_gnss_time_ > 0.0) {
                const double max_time = latest_gnss_time_ + max_imu_ahead_;
                if (imu_queue_.front().imu.time > max_time) {
                    break;
                }
            }
            const ImuSample sample = imu_queue_.front();
            imu_queue_.pop_front();

            const double imu_prev_time = sample.imu.time - sample.imu.dt;
            while (!gnss_queue_.empty() && gnss_queue_.front().gnss.time <= sample.imu.time) {
                const auto gnss_sample = gnss_queue_.front().gnss;
                gnss_queue_.pop_front();
                if (gnss_sample.time < imu_prev_time - 1e-6) {
                    ++gnss_stale_before_imu_count_;
                }
                // Keep original behavior: still pass GNSS to GIEngine.
                giengine_->addGnssData(gnss_sample);
                ++gnss_to_engine_count_;
            }

            const uint64_t nis_seq_before = giengine_->getLastNISSeq();
            giengine_->addImuData(sample.imu);
            giengine_->newImuProcess();
            const uint64_t nis_seq_after = giengine_->getLastNISSeq();
            if (nis_seq_after > nis_seq_before) {
                gnss_update_count_ += (nis_seq_after - nis_seq_before);
            }
            if (shouldPublish(sample.imu.time)) {
                if (!publishing_enabled_) {
                    publishing_enabled_ = true;
                    if (publish_path_) {
                        path_msg_.poses.clear();
                    }
                }
                updateLatestState(sample.stamp);
            }
        }
    }

    bool shouldPublish(double imu_time) const {
        if (use_absolute_time_) {
            return (imu_time >= start_time_) && (end_time_ <= 0.0 || imu_time <= end_time_);
        }
        if (!base_time_set_) {
            return false;
        }
        const double rel_time = imu_time - base_time_;
        return (rel_time >= start_time_) && (end_time_ <= 0.0 || rel_time <= end_time_);
    }

    void tryInitialize() {
        if (!giengine_) {
            if (!auto_init_enable_) {
                origin_blh_ = options_.initstate.pos;
                giengine_   = std::make_unique<GIEngine>(options_);
            } else if (!tryAutoInitializeState()) {
                return;
            }
        }
        if (imu_queue_.empty() || gnss_queue_.empty()) {
            return;
        }

        ImuSample imu0 = imu_queue_.front();

        while (!gnss_queue_.empty() && gnss_queue_.front().gnss.time <= imu0.imu.time) {
            gnss_queue_.pop_front();
        }
        if (gnss_queue_.empty()) {
            return;
        }

        GnssSample gnss0 = gnss_queue_.front();
        gnss_queue_.pop_front();

        giengine_->addImuData(imu0.imu, true);
        giengine_->addGnssData(gnss0.gnss);

        imu_queue_.pop_front();
        initialized_ = true;
        RCLCPP_INFO(get_logger(), "KF-GINS initialized. Start processing.");
    }

    void logGnssStatsTimer() {
        const uint64_t rx       = gnss_rx_count_.load();
        const uint64_t enq      = gnss_enqueue_count_.load();
        const uint64_t to_eng   = gnss_to_engine_count_.load();
        const uint64_t updates  = gnss_update_count_.load();
        const uint64_t drop_t   = gnss_time_window_drop_count_.load();
        const uint64_t drop_s   = gnss_status_drop_count_.load();
        const uint64_t drop_pre = gnss_pregate_drop_count_.load();
        const uint64_t stale    = gnss_stale_before_imu_count_.load();

        const uint64_t d_rx       = rx - last_gnss_rx_log_;
        const uint64_t d_enq      = enq - last_gnss_enq_log_;
        const uint64_t d_to_eng   = to_eng - last_gnss_to_engine_log_;
        const uint64_t d_updates  = updates - last_gnss_updates_log_;
        const uint64_t d_drop_t   = drop_t - last_gnss_drop_time_log_;
        const uint64_t d_drop_s   = drop_s - last_gnss_drop_status_log_;
        const uint64_t d_drop_pre = drop_pre - last_gnss_drop_pregate_log_;
        const uint64_t d_stale    = stale - last_gnss_stale_log_;

        last_gnss_rx_log_           = rx;
        last_gnss_enq_log_          = enq;
        last_gnss_to_engine_log_    = to_eng;
        last_gnss_updates_log_      = updates;
        last_gnss_drop_time_log_    = drop_t;
        last_gnss_drop_status_log_  = drop_s;
        last_gnss_drop_pregate_log_ = drop_pre;
        last_gnss_stale_log_        = stale;

        RCLCPP_INFO(
            get_logger(),
            "[GNSS-STATS] total(rx=%lu enq=%lu to_engine=%lu updates=%lu stale=%lu drop_time=%lu drop_status=%lu "
            "drop_pregate=%lu) delta(rx=%lu enq=%lu to_engine=%lu updates=%lu stale=%lu dt=%lu ds=%lu dp=%lu)",
            rx, enq, to_eng, updates, stale, drop_t, drop_s, drop_pre, d_rx, d_enq, d_to_eng, d_updates, d_stale,
            d_drop_t, d_drop_s, d_drop_pre);
    }

    static double vecMedian(std::vector<double> v) {
        if (v.empty()) {
            return 0.0;
        }
        const auto mid = v.begin() + static_cast<long>(v.size() / 2);
        std::nth_element(v.begin(), mid, v.end());
        double m = *mid;
        if ((v.size() % 2) == 0) {
            const auto mid2 = std::max_element(v.begin(), mid);
            m               = 0.5 * (m + *mid2);
        }
        return m;
    }

    static double vecStd(const std::vector<double> &v) {
        if (v.size() < 2) {
            return 0.0;
        }
        double mean = 0.0;
        for (double x : v) {
            mean += x;
        }
        mean /= static_cast<double>(v.size());
        double var = 0.0;
        for (double x : v) {
            const double d = x - mean;
            var += d * d;
        }
        var /= static_cast<double>(v.size() - 1);
        return std::sqrt(std::max(0.0, var));
    }

    bool shouldRejectGnssPreGate(const GnssSample &sample, bool used_covariance) {
        const double hstd = std::max(sample.gnss.std.x(), sample.gnss.std.y());
        const double vstd = sample.gnss.std.z();

        if (gnss_pre_gate_max_hstd_ > 0.0 && hstd > gnss_pre_gate_max_hstd_) {
            ++gnss_pre_gate_reject_count_;
            if (gnss_pre_gate_reject_count_ <= 10 || (gnss_pre_gate_reject_count_ % 20) == 0) {
                RCLCPP_WARN(get_logger(),
                            "GNSS pre-gate reject: hstd=%.3f > %.3f (used_cov=%s, t=%.3f, reject_count=%zu)", hstd,
                            gnss_pre_gate_max_hstd_, used_covariance ? "true" : "false", sample.gnss.time,
                            gnss_pre_gate_reject_count_);
            }
            return true;
        }

        if (gnss_pre_gate_max_vstd_ > 0.0 && vstd > gnss_pre_gate_max_vstd_) {
            ++gnss_pre_gate_reject_count_;
            if (gnss_pre_gate_reject_count_ <= 10 || (gnss_pre_gate_reject_count_ % 20) == 0) {
                RCLCPP_WARN(get_logger(),
                            "GNSS pre-gate reject: vstd=%.3f > %.3f (used_cov=%s, t=%.3f, reject_count=%zu)", vstd,
                            gnss_pre_gate_max_vstd_, used_covariance ? "true" : "false", sample.gnss.time,
                            gnss_pre_gate_reject_count_);
            }
            return true;
        }

        if (gnss_pre_gate_max_speed_ > 0.0 && have_prev_gnss_for_pregate_) {
            const double dt = sample.gnss.time - prev_gnss_time_for_pregate_;
            if (dt >= std::max(1e-3, gnss_pre_gate_min_dt_)) {
                const Eigen::Vector3d d_ned = Earth::global2local(prev_gnss_blh_for_pregate_, sample.gnss.blh);
                const double speed_xy       = d_ned.head<2>().norm() / dt;
                if (speed_xy > gnss_pre_gate_max_speed_) {
                    ++gnss_pre_gate_reject_count_;
                    if (gnss_pre_gate_reject_count_ <= 10 || (gnss_pre_gate_reject_count_ % 20) == 0) {
                        RCLCPP_WARN(get_logger(),
                                    "GNSS pre-gate reject: jump speed=%.3f m/s > %.3f (dt=%.3f, t=%.3f, reject_count=%zu)",
                                    speed_xy, gnss_pre_gate_max_speed_, dt, sample.gnss.time,
                                    gnss_pre_gate_reject_count_);
                    }
                    return true;
                }
            }
        }

        return false;
    }

    bool tryAutoInitializeState() {
        if (gnss_queue_.empty() || imu_queue_.empty()) {
            return false;
        }

        const double latest_gnss_t = gnss_queue_.back().gnss.time;
        const double latest_imu_t   = imu_queue_.back().imu.time;
        const double gnss_t0        = latest_gnss_t - std::max(0.1, auto_init_gnss_window_sec_);
        const double imu_t0         = latest_imu_t - std::max(0.1, auto_init_imu_window_sec_);

        std::vector<const GnssSample *> gnss_win;
        gnss_win.reserve(gnss_queue_.size());
        for (const auto &s : gnss_queue_) {
            if (s.gnss.time >= gnss_t0) {
                gnss_win.push_back(&s);
            }
        }
        std::vector<const ImuSample *> imu_win;
        if (!auto_init_pos_only_) {
            imu_win.reserve(imu_queue_.size());
            for (const auto &s : imu_queue_) {
                if (s.imu.time >= imu_t0) {
                    imu_win.push_back(&s);
                }
            }
        }

        if (static_cast<int>(gnss_win.size()) < auto_init_min_gnss_samples_ ||
            (!auto_init_pos_only_ && static_cast<int>(imu_win.size()) < auto_init_min_imu_samples_)) {
            return false;
        }

        std::vector<double> lats, lons, alts;
        lats.reserve(gnss_win.size());
        lons.reserve(gnss_win.size());
        alts.reserve(gnss_win.size());
        for (const auto *s : gnss_win) {
            lats.push_back(s->gnss.blh[0]);
            lons.push_back(s->gnss.blh[1]);
            alts.push_back(s->gnss.blh[2]);
        }

        double lat0 = auto_init_use_gnss_median_ ? vecMedian(lats) : std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
        double lon0 = auto_init_use_gnss_median_ ? vecMedian(lons) : std::accumulate(lons.begin(), lons.end(), 0.0) / lons.size();
        double alt0 = auto_init_use_gnss_median_ ? vecMedian(alts) : std::accumulate(alts.begin(), alts.end(), 0.0) / alts.size();

        if (auto_init_reject_gnss_outlier_ && gnss_win.size() >= 5) {
            std::vector<double> de;
            de.reserve(gnss_win.size());
            const Eigen::Vector3d ref_blh(lat0, lon0, alt0);
            for (const auto *s : gnss_win) {
                de.push_back(Earth::global2local(ref_blh, s->gnss.blh).head<2>().norm());
            }
            const double med = vecMedian(de);
            std::vector<double> abs_dev;
            abs_dev.reserve(de.size());
            for (double x : de) {
                abs_dev.push_back(std::abs(x - med));
            }
            const double mad = vecMedian(abs_dev);
            const double sigma_equiv = std::max(1e-6, 1.4826 * mad);
            std::vector<double> lats2, lons2, alts2;
            for (size_t i = 0; i < gnss_win.size(); ++i) {
                if (std::abs(de[i] - med) <= auto_init_gnss_outlier_sigma_ * sigma_equiv) {
                    lats2.push_back(gnss_win[i]->gnss.blh[0]);
                    lons2.push_back(gnss_win[i]->gnss.blh[1]);
                    alts2.push_back(gnss_win[i]->gnss.blh[2]);
                }
            }
            if (static_cast<int>(lats2.size()) >= auto_init_min_gnss_samples_) {
                lat0 = auto_init_use_gnss_median_ ? vecMedian(lats2)
                                                  : std::accumulate(lats2.begin(), lats2.end(), 0.0) / lats2.size();
                lon0 = auto_init_use_gnss_median_ ? vecMedian(lons2)
                                                  : std::accumulate(lons2.begin(), lons2.end(), 0.0) / lons2.size();
                alt0 = auto_init_use_gnss_median_ ? vecMedian(alts2)
                                                  : std::accumulate(alts2.begin(), alts2.end(), 0.0) / alts2.size();
            }
        }

        double yaw         = options_.initstate.euler[0]; // fallback to configured yaw
        double pitch       = options_.initstate.euler[1];
        double roll        = options_.initstate.euler[2];
        bool yaw_from_gnss = false;
        double acc_std_mag = 0.0;
        double gyro_std_mag = 0.0;

        if (!auto_init_pos_only_) {
            // Static IMU leveling in FRD body frame (specific force at rest).
            std::vector<double> axs, ays, azs, gxs, gys, gzs;
            axs.reserve(imu_win.size());
            ays.reserve(imu_win.size());
            azs.reserve(imu_win.size());
            gxs.reserve(imu_win.size());
            gys.reserve(imu_win.size());
            gzs.reserve(imu_win.size());
            for (const auto *s : imu_win) {
                const double dt = std::max(1e-4, s->imu.dt);
                const Eigen::Vector3d acc = s->imu.dvel / dt;
                const Eigen::Vector3d gyr = s->imu.dtheta / dt;
                axs.push_back(acc.x());
                ays.push_back(acc.y());
                azs.push_back(acc.z());
                gxs.push_back(gyr.x());
                gys.push_back(gyr.y());
                gzs.push_back(gyr.z());
            }
            const double ax_mean = std::accumulate(axs.begin(), axs.end(), 0.0) / axs.size();
            const double ay_mean = std::accumulate(ays.begin(), ays.end(), 0.0) / ays.size();
            const double az_mean = std::accumulate(azs.begin(), azs.end(), 0.0) / azs.size();
            const Eigen::Vector3d a_mean(ax_mean, ay_mean, az_mean);
            acc_std_mag =
                std::sqrt(vecStd(axs) * vecStd(axs) + vecStd(ays) * vecStd(ays) + vecStd(azs) * vecStd(azs));
            gyro_std_mag =
                std::sqrt(vecStd(gxs) * vecStd(gxs) + vecStd(gys) * vecStd(gys) + vecStd(gzs) * vecStd(gzs));

            if (auto_init_require_static_rp_ &&
                (acc_std_mag > auto_init_max_acc_std_ || gyro_std_mag > auto_init_max_gyro_std_)) {
                if (!auto_init_wait_log_printed_) {
                    RCLCPP_WARN(get_logger(),
                                "Auto-init waiting for static IMU window: acc_std=%.4f (<=%.4f), gyro_std=%.4f (<=%.4f)",
                                acc_std_mag, auto_init_max_acc_std_, gyro_std_mag, auto_init_max_gyro_std_);
                    auto_init_wait_log_printed_ = true;
                }
                return false;
            }

            const double g = std::max(1e-6, a_mean.norm());
            pitch          = std::asin(std::clamp(a_mean.x() / g, -1.0, 1.0));
            roll           = std::atan2(-a_mean.y(), -a_mean.z());

            if (auto_init_yaw_mode_ == AutoInitYawMode::GNSS_COURSE_OR_CONFIG && gnss_win.size() >= 2) {
                const auto *g0 = gnss_win.front();
                const auto *g1 = gnss_win.back();
                const double dt = g1->gnss.time - g0->gnss.time;
                if (dt > 1e-3) {
                    const Eigen::Vector3d d_ned = Earth::global2local(g0->gnss.blh, g1->gnss.blh);
                    const double speed_xy       = d_ned.head<2>().norm() / dt;
                    if (speed_xy >= auto_init_min_speed_for_yaw_ && d_ned.head<2>().norm() > 1e-3) {
                        yaw = std::atan2(d_ned.y(), d_ned.x()); // heading in NED (east,north)
                        yaw_from_gnss = true;
                    }
                }
            }
        }

        options_.initstate.pos   = Eigen::Vector3d(lat0, lon0, alt0);
        options_.initstate.euler = Eigen::Vector3d(yaw, pitch, roll);
        origin_blh_              = options_.initstate.pos;
        giengine_                = std::make_unique<GIEngine>(options_);

        RCLCPP_INFO(get_logger(),
                    "Auto-init solved: initpos=[%.8f, %.8f, %.3f], initatt[ypr]=[%.2f, %.2f, %.2f] deg%s%s",
                    lat0 * R2D, lon0 * R2D, alt0, yaw * R2D, pitch * R2D, roll * R2D,
                    yaw_from_gnss ? " (yaw from GNSS course)" : " (yaw from config)",
                    auto_init_pos_only_ ? " [pos-only]" : "");
        RCLCPP_INFO(get_logger(), "Auto-init window stats: gnss=%zu imu=%zu acc_std=%.4f gyro_std=%.4f", gnss_win.size(),
                    imu_win.size(), acc_std_mag, gyro_std_mag);
        return true;
    }

    void updateLatestState(const rclcpp::Time &stamp) {
        const NavState nav = giengine_->getNavState();

        Eigen::Vector3d local_ned = Earth::global2local(origin_blh_, nav.pos);
        Eigen::Vector3d vel_ned   = nav.vel;

        Eigen::Vector3d local_out = local_ned;
        Eigen::Vector3d vel_out   = vel_ned;

        Eigen::Matrix3d c_b_n   = Rotation::euler2matrix(nav.euler);
        Eigen::Matrix3d c_b_out = c_b_n;

        if (output_enu_) {
            Eigen::Matrix3d r = nedToEnuMatrix();
            local_out         = nedToEnu(local_ned);
            vel_out           = nedToEnu(vel_ned);
            c_b_out           = r * c_b_n;
        }

        if (odom_orientation_flu_) {
            // KF-GINS body frame is FRD. ROS odom/base_link convention is typically FLU.
            // Convert the published body orientation basis from FRD to FLU while keeping the
            // same physical pose in the navigation frame.
            c_b_out = c_b_out * frdToFluMatrix();
        }

        Eigen::Quaterniond q(c_b_out);

        nav_msgs::msg::Odometry odom;
        odom.header.stamp            = stamp;
        odom.header.frame_id         = frame_id_;
        odom.child_frame_id          = child_frame_id_;
        odom.pose.pose.position.x    = local_out.x();
        odom.pose.pose.position.y    = local_out.y();
        odom.pose.pose.position.z    = local_out.z();
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();

        // Publish-side heading fusion skeleton: blend yaw with single-GNSS course estimate.
        // This improves heading continuity in single-antenna mode without changing internal GIEngine state.
        if (heading_fusion_enable_ && heading_mode_ == HeadingMode::SINGLE_GNSS_COURSE && output_enu_ && odom_orientation_flu_) {
            bool have_course = false;
            double yaw_course = 0.0;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (have_single_gnss_course_heading_) {
                    have_course = true;
                    yaw_course  = latest_single_gnss_course_yaw_enu_;
                }
            }
            if (have_course) {
                Eigen::Quaterniond q_pub(odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
                                         odom.pose.pose.orientation.y, odom.pose.pose.orientation.z);
                Eigen::Vector3d rpy = quatToRpy(q_pub);
                const double dyaw   = wrapAngleRad(yaw_course - rpy.z());
                const double gain   = std::clamp(single_heading_blend_gain_, 0.0, 1.0);
                rpy.z()             = wrapAngleRad(rpy.z() + gain * dyaw);
                Eigen::Quaterniond q_blend = rpyToQuat(rpy.x(), rpy.y(), rpy.z()).normalized();
                odom.pose.pose.orientation.w = q_blend.w();
                odom.pose.pose.orientation.x = q_blend.x();
                odom.pose.pose.orientation.y = q_blend.y();
                odom.pose.pose.orientation.z = q_blend.z();
            }
        }

        odom.twist.twist.linear.x = vel_out.x();
        odom.twist.twist.linear.y = vel_out.y();
        odom.twist.twist.linear.z = vel_out.z();

        const auto out_stamp = selectOutputStamp(stamp);
        if (odom_fused_pub_) {
            nav_msgs::msg::Odometry odom_fused = odom;
            odom_fused.header.stamp            = out_stamp;
            odom_fused_pub_->publish(odom_fused);
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            // Estimate body angular velocity from two consecutive fused attitudes.
            if (have_odom_) {
                const double dt = (out_stamp - last_fused_time_).seconds();
                if (dt > 1e-6) {
                    Eigen::Quaterniond q_prev(last_odom_.pose.pose.orientation.w, last_odom_.pose.pose.orientation.x,
                                              last_odom_.pose.pose.orientation.y, last_odom_.pose.pose.orientation.z);
                    Eigen::Quaterniond q_curr(odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
                                              odom.pose.pose.orientation.y, odom.pose.pose.orientation.z);
                    q_prev.normalize();
                    q_curr.normalize();
                    Eigen::Quaterniond dq = q_prev.conjugate() * q_curr;
                    if (dq.w() < 0.0) {
                        dq.coeffs() *= -1.0;
                    }
                    const Eigen::AngleAxisd aa(dq);
                    last_fused_angular_vel_ = aa.axis() * (aa.angle() / dt);
                } else {
                    last_fused_angular_vel_.setZero();
                }
            } else {
                last_fused_angular_vel_.setZero();
            }

            last_odom_                 = odom;
            last_pose_.header.stamp    = stamp;
            last_pose_.header.frame_id = path_frame_id_;
            last_pose_.pose            = odom.pose.pose;
            last_fused_time_           = out_stamp;
            have_odom_                 = true;
        }

        if (publish_navsat_ && navsat_pub_) {
            sensor_msgs::msg::NavSatFix navsat;
            navsat.header.stamp             = stamp;
            navsat.header.frame_id          = navsat_frame_id_;
            navsat.status.status            = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
            navsat.status.service           = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
            navsat.latitude                 = nav.pos[0] * R2D;
            navsat.longitude                = nav.pos[1] * R2D;
            navsat.altitude                 = nav.pos[2];
            navsat.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_navsat_ = navsat;
            have_navsat_ = true;
        }

        if (publish_nis_ && nis_pub_) {
            const auto nis_seq = giengine_->getLastNISSeq();
            if (nis_seq != last_published_nis_seq_) {
                std_msgs::msg::Float64 msg;
                msg.data = giengine_->getLastNIS();
                nis_pub_->publish(msg);
                last_published_nis_seq_ = nis_seq;
            }
        }

        if (odom_publish_rate_ <= 0.0) {
            publishOdomTimer();
        }
        if (publish_path_ && path_publish_rate_ <= 0.0) {
            publishPathTimer();
        }
    }

    bool getPredictedOdomAt(const rclcpp::Time &target_time, nav_msgs::msg::Odometry &odom_out) {
        nav_msgs::msg::Odometry odom;
        rclcpp::Time fused_time;
        rclcpp::Time input_wall_time;
        Eigen::Vector3d omega_body = Eigen::Vector3d::Zero();
        bool have                  = false;
        bool have_input            = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            have = have_odom_;
            if (have) {
                odom       = last_odom_;
                fused_time = last_fused_time_;
                omega_body = last_fused_angular_vel_;
            }
            have_input      = have_input_;
            input_wall_time = last_input_wall_time_;
        }
        if (!have) {
            return false;
        }
        if (!have_input) {
            return false;
        }
        if ((this->now() - input_wall_time).seconds() > input_stale_timeout_) {
            return false;
        }

        const double dt_pred = (target_time - fused_time).seconds();
        if (dt_pred > 0.0) {
            // Position extrapolation with constant linear velocity.
            odom.pose.pose.position.x += odom.twist.twist.linear.x * dt_pred;
            odom.pose.pose.position.y += odom.twist.twist.linear.y * dt_pred;
            odom.pose.pose.position.z += odom.twist.twist.linear.z * dt_pred;

            // Attitude extrapolation with constant body angular velocity.
            const double omega_norm = omega_body.norm();
            if (omega_norm > 1e-8) {
                Eigen::Quaterniond q(odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
                                     odom.pose.pose.orientation.y, odom.pose.pose.orientation.z);
                q.normalize();
                const double angle = omega_norm * dt_pred;
                Eigen::AngleAxisd aa(angle, omega_body / omega_norm);
                q = q * Eigen::Quaterniond(aa);
                q.normalize();
                odom.pose.pose.orientation.w = q.w();
                odom.pose.pose.orientation.x = q.x();
                odom.pose.pose.orientation.y = q.y();
                odom.pose.pose.orientation.z = q.z();
            }
        }

        odom.header.stamp = target_time;
        odom_out          = odom;
        return true;
    }

    void publishOdomTimer() {
        const auto now_stamp = selectTimerTargetStamp();
        nav_msgs::msg::Odometry odom;
        if (!getPredictedOdomAt(now_stamp, odom)) {
            return;
        }
        odom_pub_->publish(odom);
        if (publish_tf_ && tf_broadcaster_) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header.stamp            = now_stamp;
            tf.header.frame_id         = frame_id_;
            tf.child_frame_id          = child_frame_id_;
            tf.transform.translation.x = odom.pose.pose.position.x;
            tf.transform.translation.y = odom.pose.pose.position.y;
            tf.transform.translation.z = odom.pose.pose.position.z;
            tf.transform.rotation      = odom.pose.pose.orientation;
            tf_broadcaster_->sendTransform(tf);
        }
        if (publish_navsat_ && navsat_publish_rate_ <= 0.0) {
            publishNavsatTimer();
        }
    }

    void publishNavsatTimer() {
        sensor_msgs::msg::NavSatFix navsat;
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            have = have_navsat_;
            if (have) {
                navsat = last_navsat_;
            }
        }
        if (!have || !navsat_pub_) {
            return;
        }
        navsat.header.stamp = selectTimerTargetStamp();
        navsat_pub_->publish(navsat);
    }

    void publishPathTimer() {
        if (!path_pub_) {
            return;
        }
        const auto now_stamp = selectTimerTargetStamp();
        nav_msgs::msg::Odometry odom;
        if (!getPredictedOdomAt(now_stamp, odom)) {
            return;
        }
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp      = now_stamp;
        pose.header.frame_id   = path_frame_id_;
        pose.pose              = odom.pose.pose;
        if (path_incremental_output_) {
            nav_msgs::msg::Path path_incremental;
            path_incremental.header.frame_id = path_frame_id_;
            path_incremental.header.stamp    = now_stamp;
            path_incremental.poses.push_back(pose);
            path_pub_->publish(path_incremental);
        } else {
            path_msg_.header.stamp = now_stamp;
            path_msg_.poses.push_back(pose);
            if (path_max_size_ > 0 && static_cast<int>(path_msg_.poses.size()) > path_max_size_) {
                const auto drop = path_msg_.poses.size() - static_cast<size_t>(path_max_size_);
                path_msg_.poses.erase(path_msg_.poses.begin(), path_msg_.poses.begin() + drop);
            }
            path_pub_->publish(path_msg_);
        }
    }

    rclcpp::Time selectOutputStamp(const rclcpp::Time &measurement_stamp) const {
        return use_wall_time_stamp_ ? this->now() : measurement_stamp;
    }

    rclcpp::Time selectTimerTargetStamp() {
        if (use_wall_time_stamp_) {
            return this->now();
        }
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (have_odom_) {
            return last_fused_time_;
        }
        return rclcpp::Time(0, 0, RCL_SYSTEM_TIME);
    }

private:
    std::string imu_topic_;
    std::string gps_topic_;
    std::string odom_topic_;
    std::string odom_fused_topic_;
    std::string path_topic_;
    std::string navsat_topic_;
    std::string nis_topic_;
    std::string frame_id_;
    std::string path_frame_id_;
    std::string navsat_frame_id_;
    std::string child_frame_id_;
    bool publish_tf_{true};
    bool publish_path_{true};
    bool publish_navsat_{true};
    bool publish_nis_{false};
    int path_max_size_{2000};
    bool path_incremental_output_{false};
    double odom_publish_rate_{0.0};
    double navsat_publish_rate_{-1.0};
    double path_publish_rate_{10.0};
    double input_stale_timeout_{0.5};
    bool output_enu_{true};
    bool imu_in_flu_{true};
    bool odom_orientation_flu_{true};
    double imu_rate_{200.0};
    double max_imu_dt_{0.1};
    int max_queue_size_{2000};
    bool use_navsatfix_covariance_{true};
    std::vector<double> gnss_std_;
    bool gnss_pre_gate_enable_{false};
    double gnss_pre_gate_max_hstd_{-1.0};
    double gnss_pre_gate_max_vstd_{-1.0};
    double gnss_pre_gate_max_speed_{-1.0};
    double gnss_pre_gate_min_dt_{0.2};
    size_t gnss_pre_gate_reject_count_{0};
    bool have_prev_gnss_for_pregate_{false};
    Eigen::Vector3d prev_gnss_blh_for_pregate_{0.0, 0.0, 0.0};
    double prev_gnss_time_for_pregate_{-1.0};
    std::string gnss_update_mode_name_{"xyz"};
    std::string gnss_nis_gate_mode_name_{"xyz"};
    double gnss_time_offset_sec_{0.0};
    double start_time_{0.0};
    double end_time_{-1.0};
    bool use_absolute_time_{false};
    double max_imu_ahead_{0.0};
    bool use_wall_time_stamp_{true};
    std::string filter_scheme_name_{"ESKF"};
    std::string heading_mode_name_{"none"};
    HeadingMode heading_mode_{HeadingMode::NONE};
    bool heading_fusion_enable_{false};
    bool heading_fallback_to_imu_{true};
    double single_heading_min_speed_{1.0};
    double single_heading_std_deg_{5.0};
    double single_heading_blend_gain_{0.2};
    double dual_heading_std_deg_{1.0};
    bool dual_heading_quality_gate_{true};
    bool dual_heading_nis_gate_{true};
    std::string dual_heading_topic_{"/gnss/dual_heading"};
    bool auto_init_enable_{false};
    bool auto_init_pos_only_{false};
    double auto_init_gnss_window_sec_{8.0};
    double auto_init_imu_window_sec_{3.0};
    int auto_init_min_gnss_samples_{5};
    int auto_init_min_imu_samples_{100};
    bool auto_init_use_gnss_median_{true};
    bool auto_init_reject_gnss_outlier_{true};
    double auto_init_gnss_outlier_sigma_{3.5};
    bool auto_init_require_static_rp_{false};
    double auto_init_max_acc_std_{0.5};
    double auto_init_max_gyro_std_{0.1};
    std::string auto_init_yaw_mode_name_{"gnss_course_or_config"};
    AutoInitYawMode auto_init_yaw_mode_{AutoInitYawMode::GNSS_COURSE_OR_CONFIG};
    double auto_init_min_speed_for_yaw_{1.0};
    double gnss_stats_log_period_sec_{5.0};
    bool auto_init_wait_log_printed_{false};
    bool have_prev_gnss_fix_for_course_{false};
    Eigen::Vector3d prev_gnss_blh_for_course_{0.0, 0.0, 0.0};
    double prev_gnss_fix_time_for_course_{-1.0};

    double base_time_{0.0};
    bool base_time_set_{false};
    double latest_gnss_time_{-1.0};

    std::unique_ptr<GIEngine> giengine_;
    GINSOptions options_;
    Eigen::Vector3d origin_blh_{0.0, 0.0, 0.0};

    std::deque<ImuSample> imu_queue_;
    std::deque<GnssSample> gnss_queue_;
    double last_imu_time_{-1.0};
    bool initialized_{false};

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_fused_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr navsat_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nis_pub_;
    nav_msgs::msg::Path path_msg_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    bool publishing_enabled_{false};

    nav_msgs::msg::Odometry last_odom_;
    sensor_msgs::msg::NavSatFix last_navsat_;
    geometry_msgs::msg::PoseStamped last_pose_;
    rclcpp::Time last_fused_time_{0, 0, RCL_SYSTEM_TIME};
    rclcpp::Time last_input_wall_time_{0, 0, RCL_SYSTEM_TIME};
    Eigen::Vector3d last_fused_angular_vel_{Eigen::Vector3d::Zero()};
    bool have_odom_{false};
    bool have_navsat_{false};
    bool have_input_{false};
    uint64_t last_published_nis_seq_{0};
    std::atomic<uint64_t> gnss_rx_count_{0};
    std::atomic<uint64_t> gnss_enqueue_count_{0};
    std::atomic<uint64_t> gnss_to_engine_count_{0};
    std::atomic<uint64_t> gnss_update_count_{0};
    std::atomic<uint64_t> gnss_time_window_drop_count_{0};
    std::atomic<uint64_t> gnss_status_drop_count_{0};
    std::atomic<uint64_t> gnss_pregate_drop_count_{0};
    std::atomic<uint64_t> gnss_stale_before_imu_count_{0};
    uint64_t last_gnss_rx_log_{0};
    uint64_t last_gnss_enq_log_{0};
    uint64_t last_gnss_to_engine_log_{0};
    uint64_t last_gnss_updates_log_{0};
    uint64_t last_gnss_drop_time_log_{0};
    uint64_t last_gnss_drop_status_log_{0};
    uint64_t last_gnss_drop_pregate_log_{0};
    uint64_t last_gnss_stale_log_{0};
    bool have_single_gnss_course_heading_{false};
    double latest_single_gnss_course_yaw_enu_{0.0};
    double latest_single_gnss_course_speed_{0.0};
    rclcpp::Time latest_single_gnss_course_stamp_{0, 0, RCL_SYSTEM_TIME};

    std::mutex state_mutex_;

    rclcpp::TimerBase::SharedPtr odom_timer_;
    rclcpp::TimerBase::SharedPtr navsat_timer_;
    rclcpp::TimerBase::SharedPtr path_timer_;
    rclcpp::TimerBase::SharedPtr gnss_stats_timer_;
    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KfGinsNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
