/*
 * KF-GINS ROS2 node: GNSS/INS integrated navigation
 */

#include <cmath>
#include <deque>
#include <mutex>
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
        frame_id_                 = declare_parameter<std::string>("frame_id", "map");
        path_frame_id_            = declare_parameter<std::string>("path_frame_id", "");
        navsat_frame_id_          = declare_parameter<std::string>("navsat_frame_id", "");
        child_frame_id_           = declare_parameter<std::string>("child_frame_id", "base_link");
        publish_tf_               = declare_parameter<bool>("publish_tf", true);
        publish_path_             = declare_parameter<bool>("publish_path", true);
        publish_navsat_           = declare_parameter<bool>("publish_navsat", true);
        path_max_size_            = declare_parameter<int>("path_max_size", 2000);
        odom_publish_rate_        = declare_parameter<double>("odom_publish_rate", 0.0);
        navsat_publish_rate_      = declare_parameter<double>("navsat_publish_rate", -1.0);
        path_publish_rate_        = declare_parameter<double>("path_publish_rate", 10.0);
        input_stale_timeout_      = declare_parameter<double>("input_stale_timeout", 0.5);
        output_enu_               = declare_parameter<bool>("output_enu", true);
        imu_in_flu_               = declare_parameter<bool>("imu_in_flu", true);
        imu_rate_                 = declare_parameter<double>("imu_rate", 200.0);
        max_imu_dt_               = declare_parameter<double>("max_imu_dt", 0.1);
        max_queue_size_           = declare_parameter<int>("max_queue_size", 2000);
        use_navsatfix_covariance_ = declare_parameter<bool>("use_navsatfix_covariance", true);
        gnss_std_                 = declare_parameter<std::vector<double>>("gnss_std", {1.0, 1.0, 2.0});
        start_time_               = declare_parameter<double>("start_time", 0.0);
        end_time_                 = declare_parameter<double>("end_time", -1.0);
        use_absolute_time_        = declare_parameter<bool>("use_absolute_time", false);
        max_imu_ahead_            = declare_parameter<double>("max_imu_ahead", 0.0);
        use_wall_time_stamp_      = declare_parameter<bool>("use_wall_time_stamp", true);

        if (gnss_std_.size() != 3) {
            RCLCPP_WARN(get_logger(), "Parameter 'gnss_std' must be 3 elements. Using default [1,1,2].");
            gnss_std_ = {1.0, 1.0, 2.0};
        }

        declareKfGinsParams();
        if (!loadOptionsFromParams()) {
            RCLCPP_ERROR(get_logger(), "Failed to load KF-GINS parameters from ROS2 params.");
            throw std::runtime_error("Failed to load params");
        }

        origin_blh_ = options_.initstate.pos;
        giengine_   = std::make_unique<GIEngine>(options_);

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
        if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
            return;
        }

        GnssSample sample;
        sample.stamp       = msg->header.stamp;
        sample.gnss.time   = rclcpp::Time(sample.stamp).seconds();
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

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_input_wall_time_ = this->now();
            have_input_           = true;
        }

        gnss_queue_.push_back(sample);
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

            while (!gnss_queue_.empty() && gnss_queue_.front().gnss.time <= sample.imu.time) {
                giengine_->addGnssData(gnss_queue_.front().gnss);
                gnss_queue_.pop_front();
            }

            giengine_->addImuData(sample.imu);
            giengine_->newImuProcess();
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
        path_msg_.header.stamp = now_stamp;
        path_msg_.poses.push_back(pose);
        if (path_max_size_ > 0 && static_cast<int>(path_msg_.poses.size()) > path_max_size_) {
            const auto drop = path_msg_.poses.size() - static_cast<size_t>(path_max_size_);
            path_msg_.poses.erase(path_msg_.poses.begin(), path_msg_.poses.begin() + drop);
        }
        path_pub_->publish(path_msg_);
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
    std::string frame_id_;
    std::string path_frame_id_;
    std::string navsat_frame_id_;
    std::string child_frame_id_;
    bool publish_tf_{true};
    bool publish_path_{true};
    bool publish_navsat_{true};
    int path_max_size_{2000};
    double odom_publish_rate_{0.0};
    double navsat_publish_rate_{-1.0};
    double path_publish_rate_{10.0};
    double input_stale_timeout_{0.5};
    bool output_enu_{true};
    bool imu_in_flu_{true};
    double imu_rate_{200.0};
    double max_imu_dt_{0.1};
    int max_queue_size_{2000};
    bool use_navsatfix_covariance_{true};
    std::vector<double> gnss_std_;
    double start_time_{0.0};
    double end_time_{-1.0};
    bool use_absolute_time_{false};
    double max_imu_ahead_{0.0};
    bool use_wall_time_stamp_{true};

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

    std::mutex state_mutex_;

    rclcpp::TimerBase::SharedPtr odom_timer_;
    rclcpp::TimerBase::SharedPtr navsat_timer_;
    rclcpp::TimerBase::SharedPtr path_timer_;
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
