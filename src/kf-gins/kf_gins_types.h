/*
 * KF-GINS: An EKF-Based GNSS/INS Integrated Navigation System
 *
 * Copyright (C) 2022 i2Nav Group, Wuhan University
 *
 *     Author : Liqiang Wang
 *    Contact : wlq@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef KF_GINS_TYPES_H
#define KF_GINS_TYPES_H

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "common/angle.h"

enum class FilterScheme : int {
    ESKF = 0,
    UKF = 1,
    SR_UKF = 2,
    ADAPTIVE_UKF = 3,
    ROBUST_UKF = 4,
};

inline const char *filterSchemeName(FilterScheme scheme) {
    switch (scheme) {
    case FilterScheme::ESKF:
        return "ESKF";
    case FilterScheme::UKF:
        return "UKF";
    case FilterScheme::SR_UKF:
        return "SR_UKF";
    case FilterScheme::ADAPTIVE_UKF:
        return "ADAPTIVE_UKF";
    case FilterScheme::ROBUST_UKF:
        return "ROBUST_UKF";
    default:
        return "UNKNOWN";
    }
}

enum class GnssPosMeasMode : int {
    XYZ = 0,
    XY = 1,
};

inline const char *gnssPosMeasModeName(GnssPosMeasMode mode) {
    switch (mode) {
    case GnssPosMeasMode::XYZ:
        return "xyz";
    case GnssPosMeasMode::XY:
        return "xy";
    default:
        return "unknown";
    }
}

typedef struct Attitude {
    Eigen::Quaterniond qbn;
    Eigen::Matrix3d cbn;
    Eigen::Vector3d euler;
} Attitude;

typedef struct PVA {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Attitude att;
} PVA;

typedef struct ImuError {
    Eigen::Vector3d gyrbias;
    Eigen::Vector3d accbias;
    Eigen::Vector3d gyrscale;
    Eigen::Vector3d accscale;
} ImuError;

typedef struct NavState {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d euler;

    ImuError imuerror;
} NavState;

typedef struct ImuNoise {
    Eigen::Vector3d gyr_arw;
    Eigen::Vector3d acc_vrw;
    Eigen::Vector3d gyrbias_std;
    Eigen::Vector3d accbias_std;
    Eigen::Vector3d gyrscale_std;
    Eigen::Vector3d accscale_std;
    double corr_time;
} ImuNoise;

typedef struct GINSOptions {

    // 初始状态和状态标准差
    // initial state and state standard deviation
    NavState initstate;
    NavState initstate_std;

    // IMU噪声参数
    // imu noise parameters
    ImuNoise imunoise;

    // 安装参数
    // install parameters
    Eigen::Vector3d antlever = {0, 0, 0};

    // 滤波方案枚举（当前仅ESKF已实现，其余作为实验扩展入口）
    // filter scheme enum (only ESKF is implemented currently)
    FilterScheme filter_scheme = FilterScheme::ESKF;

    // UKF 参数（用于实验对比）
    // UKF parameters (for experiment comparison)
    double ukf_alpha = 1.0e-3;
    double ukf_beta = 2.0;
    double ukf_kappa = 0.0;

    // GNSS NIS gating (applies to measurement update in current implementation)
    GnssPosMeasMode gnss_update_mode = GnssPosMeasMode::XYZ;
    GnssPosMeasMode gnss_nis_gate_mode = GnssPosMeasMode::XYZ;
    bool gnss_nis_gate_enable = false;
    double gnss_nis_gate_threshold = 11.34; // chi-square 99% for 3D position measurement

    void print_options() {
        std::cout << "---------------KF-GINS Options:---------------" << std::endl;

        // 打印初始状态
        // print initial state
        std::cout << " - Initial State: " << std::endl;
        std::cout << '\t' << "- initial position: ";
        std::cout << std::setprecision(12) << initstate.pos[0] * R2D << "  ";
        std::cout << std::setprecision(12) << initstate.pos[1] * R2D << "  ";
        std::cout << std::setprecision(6) << initstate.pos[2] << " [deg, deg, m] " << std::endl;
        std::cout << '\t' << "- initial velocity: " << initstate.vel.transpose() << " [m/s] " << std::endl;
        std::cout << '\t' << "- initial attitude: " << initstate.euler.transpose() * R2D << " [deg] " << std::endl;
        std::cout << '\t' << "- initial gyrbias : " << initstate.imuerror.gyrbias.transpose() * R2D * 3600
                  << " [deg/h] " << std::endl;
        std::cout << '\t' << "- initial accbias : " << initstate.imuerror.accbias.transpose() * 1e5 << " [mGal] "
                  << std::endl;
        std::cout << '\t' << "- initial gyrscale: " << initstate.imuerror.gyrscale.transpose() * 1e6 << " [ppm] "
                  << std::endl;
        std::cout << '\t' << "- initial accscale: " << initstate.imuerror.accscale.transpose() * 1e6 << " [ppm] "
                  << std::endl;

        // 打印初始状态标准差
        // print initial state STD
        std::cout << " - Initial State STD: " << std::endl;
        std::cout << '\t' << "- initial position std: " << initstate_std.pos.transpose() << " [m] " << std::endl;
        std::cout << '\t' << "- initial velocity std: " << initstate_std.vel.transpose() << " [m/s] " << std::endl;
        std::cout << '\t' << "- initial attitude std: " << initstate_std.euler.transpose() * R2D << " [deg] "
                  << std::endl;
        std::cout << '\t' << "- initial gyrbias std: " << initstate_std.imuerror.gyrbias.transpose() * R2D * 3600
                  << " [deg/h] " << std::endl;
        std::cout << '\t' << "- initial accbias std: " << initstate_std.imuerror.accbias.transpose() * 1e5 << " [mGal] "
                  << std::endl;
        std::cout << '\t' << "- initial gyrscale std: " << initstate_std.imuerror.gyrscale.transpose() * 1e6
                  << " [ppm] " << std::endl;
        std::cout << '\t' << "- initial accscale std: " << initstate_std.imuerror.accscale.transpose() * 1e6
                  << " [ppm] " << std::endl;

        // 打印IMU噪声参数
        // print IMU noise parameters
        std::cout << " - IMU noise: " << std::endl;
        std::cout << '\t' << "- arw: " << imunoise.gyr_arw.transpose() * R2D * 60 << " [deg/sqrt(h)] " << std::endl;
        std::cout << '\t' << "- vrw: " << imunoise.acc_vrw.transpose() * 60 << " [m/s/sqrt(h)] " << std::endl;
        std::cout << '\t' << "- gyrbias  std: " << imunoise.gyrbias_std.transpose() * R2D * 3600 << " [deg/h] "
                  << std::endl;
        std::cout << '\t' << "- accbias  std: " << imunoise.accbias_std.transpose() * 1e5 << " [mGal] " << std::endl;
        std::cout << '\t' << "- gyrscale std: " << imunoise.gyrscale_std.transpose() * 1e6 << " [ppm] " << std::endl;
        std::cout << '\t' << "- accscale std: " << imunoise.accscale_std.transpose() * 1e6 << " [ppm] " << std::endl;
        std::cout << '\t' << "- correlation time: " << imunoise.corr_time / 3600.0 << " [h] " << std::endl;

        // 打印GNSS天线杆臂
        // print GNSS antenna leverarm
        std::cout << " - Antenna leverarm: " << antlever.transpose() << " [m] " << std::endl << std::endl;
        std::cout << " - Filter scheme: " << filterSchemeName(filter_scheme) << std::endl << std::endl;
        std::cout << " - UKF params: alpha=" << ukf_alpha << ", beta=" << ukf_beta << ", kappa=" << ukf_kappa
                  << std::endl
                  << std::endl;
        std::cout << " - GNSS NIS gate: " << (gnss_nis_gate_enable ? "enabled" : "disabled")
                  << ", threshold=" << gnss_nis_gate_threshold
                  << ", update_mode=" << gnssPosMeasModeName(gnss_update_mode)
                  << ", gate_mode=" << gnssPosMeasModeName(gnss_nis_gate_mode) << std::endl
                  << std::endl;
    }

} GINSOptions;

#endif // KF_GINS_TYPES_H
