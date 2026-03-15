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

#include "common/earth.h"
#include "common/rotation.h"

#include "gi_engine.h"
#include "insmech.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

std::string formatGateVector(const Eigen::VectorXd &vec, int dim) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dim; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << std::fixed << std::setprecision(3) << vec(i);
    }
    oss << "]";
    return oss.str();
}

Eigen::VectorXd gateStdFromDiag(const Eigen::MatrixXd &mat, int dim) {
    Eigen::VectorXd out(dim);
    for (int i = 0; i < dim; ++i) {
        const double d = mat(i, i);
        out(i)         = (d >= 0.0) ? std::sqrt(d) : std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

} // namespace

GIEngine::GIEngine(GINSOptions &options) {

    this->options_ = options;
    if (options_.filter_scheme != FilterScheme::ESKF && options_.filter_scheme != FilterScheme::UKF &&
        options_.filter_scheme != FilterScheme::SR_UKF) {
        std::cerr << "[GIEngine] Filter scheme '" << filterSchemeName(options_.filter_scheme)
                  << "' is not implemented yet, fallback to ESKF." << std::endl;
        options_.filter_scheme = FilterScheme::ESKF;
    }
    options_.print_options();
    timestamp_ = 0;

    // 设置协方差矩阵，系统噪声阵和系统误差状态矩阵大小
    // resize covariance matrix, system noise matrix, and system error state matrix
    Cov_.resize(RANK, RANK);
    Qc_.resize(NOISERANK, NOISERANK);
    dx_.resize(RANK, 1);
    Cov_.setZero();
    Qc_.setZero();
    dx_.setZero();

    // 初始化系统噪声阵
    // initialize noise matrix
    auto imunoise                   = options_.imunoise;
    Qc_.block(ARW_ID, ARW_ID, 3, 3) = imunoise.gyr_arw.cwiseProduct(imunoise.gyr_arw).asDiagonal();
    Qc_.block(VRW_ID, VRW_ID, 3, 3) = imunoise.acc_vrw.cwiseProduct(imunoise.acc_vrw).asDiagonal();
    Qc_.block(BGSTD_ID, BGSTD_ID, 3, 3) =
        2 / imunoise.corr_time * imunoise.gyrbias_std.cwiseProduct(imunoise.gyrbias_std).asDiagonal();
    Qc_.block(BASTD_ID, BASTD_ID, 3, 3) =
        2 / imunoise.corr_time * imunoise.accbias_std.cwiseProduct(imunoise.accbias_std).asDiagonal();
    Qc_.block(SGSTD_ID, SGSTD_ID, 3, 3) =
        2 / imunoise.corr_time * imunoise.gyrscale_std.cwiseProduct(imunoise.gyrscale_std).asDiagonal();
    Qc_.block(SASTD_ID, SASTD_ID, 3, 3) =
        2 / imunoise.corr_time * imunoise.accscale_std.cwiseProduct(imunoise.accscale_std).asDiagonal();

    // 设置系统状态(位置、速度、姿态和IMU误差)初值和初始协方差
    // set initial state (position, velocity, attitude and IMU error) and covariance
    initialize(options_.initstate, options_.initstate_std);
}

void GIEngine::initialize(const NavState &initstate, const NavState &initstate_std) {

    // 初始化位置、速度、姿态
    // initialize position, velocity and attitude
    pvacur_.pos       = initstate.pos;
    pvacur_.vel       = initstate.vel;
    pvacur_.att.euler = initstate.euler;
    pvacur_.att.cbn   = Rotation::euler2matrix(pvacur_.att.euler);
    pvacur_.att.qbn   = Rotation::euler2quaternion(pvacur_.att.euler);
    // 初始化IMU误差
    // initialize imu error
    imuerror_ = initstate.imuerror;

    // 给上一时刻状态赋同样的初值
    // set the same value to the previous state
    pvapre_ = pvacur_;

    // 初始化协方差
    // initialize covariance
    ImuError imuerror_std            = initstate_std.imuerror;
    Cov_.block(P_ID, P_ID, 3, 3)     = initstate_std.pos.cwiseProduct(initstate_std.pos).asDiagonal();
    Cov_.block(V_ID, V_ID, 3, 3)     = initstate_std.vel.cwiseProduct(initstate_std.vel).asDiagonal();
    Cov_.block(PHI_ID, PHI_ID, 3, 3) = initstate_std.euler.cwiseProduct(initstate_std.euler).asDiagonal();
    Cov_.block(BG_ID, BG_ID, 3, 3)   = imuerror_std.gyrbias.cwiseProduct(imuerror_std.gyrbias).asDiagonal();
    Cov_.block(BA_ID, BA_ID, 3, 3)   = imuerror_std.accbias.cwiseProduct(imuerror_std.accbias).asDiagonal();
    Cov_.block(SG_ID, SG_ID, 3, 3)   = imuerror_std.gyrscale.cwiseProduct(imuerror_std.gyrscale).asDiagonal();
    Cov_.block(SA_ID, SA_ID, 3, 3)   = imuerror_std.accscale.cwiseProduct(imuerror_std.accscale).asDiagonal();
}

void GIEngine::newImuProcess() {

    // 当前IMU时间作为系统当前状态时间,
    // set current IMU time as the current state time
    timestamp_ = imucur_.time;

    // 如果GNSS有效，则将更新时间设置为GNSS时间
    // set update time as the gnss time if gnssdata is valid
    double updatetime = gnssdata_.isvalid ? gnssdata_.time : -1;

    // 判断是否需要进行GNSS更新
    // determine if we should do GNSS update
    int res = isToUpdate(imupre_.time, imucur_.time, updatetime);

    if (res == 0) {
        // 只传播导航状态
        // only propagate navigation state
        insPropagation(imupre_, imucur_);
    } else if (res == 1) {
        // GNSS数据靠近上一历元，先对上一历元进行GNSS更新
        // gnssdata is near to the previous imudata, we should firstly do gnss update
        gnssUpdate(gnssdata_);
        stateFeedback();

        pvapre_ = pvacur_;
        insPropagation(imupre_, imucur_);
    } else if (res == 2) {
        // GNSS数据靠近当前历元，先对当前IMU进行状态传播
        // gnssdata is near current imudata, we should firstly propagate navigation state
        insPropagation(imupre_, imucur_);
        gnssUpdate(gnssdata_);
        stateFeedback();
    } else {
        // GNSS数据在两个IMU数据之间(不靠近任何一个), 将当前IMU内插到整秒时刻
        // gnssdata is between the two imudata, we interpolate current imudata to gnss time
        IMU midimu;
        imuInterpolate(imupre_, imucur_, updatetime, midimu);
        // NOTE：内插之后采样间隔会变化，严格上不满足INSMech的假设，但影响较小暂时忽略
        // Interpolation changes sampling interval, slightly violating INSMech's assumption (negligible for now)

        // 对前一半IMU进行状态传播
        // propagate navigation state for the first half imudata
        insPropagation(imupre_, midimu);

        // 整秒时刻进行GNSS更新，并反馈系统状态
        // do GNSS position update at the whole second and feedback system states
        gnssUpdate(gnssdata_);
        stateFeedback();

        // 对后一半IMU进行状态传播
        // propagate navigation state for the second half imudata
        pvapre_ = pvacur_;
        insPropagation(midimu, imucur_);
    }

    // 检查协方差矩阵对角线元素
    // check diagonal elements of current covariance matrix
    checkCov();

    // 更新上一时刻的状态和IMU数据
    // update system state and imudata at the previous epoch
    pvapre_ = pvacur_;
    imupre_ = imucur_;
}

void GIEngine::imuCompensate(IMU &imu) {

    // 补偿IMU零偏
    // compensate the imu bias
    imu.dtheta -= imuerror_.gyrbias * imu.dt;
    imu.dvel -= imuerror_.accbias * imu.dt;

    // 补偿IMU比例因子
    // compensate the imu scale
    Eigen::Vector3d gyrscale, accscale;
    gyrscale   = Eigen::Vector3d::Ones() + imuerror_.gyrscale;
    accscale   = Eigen::Vector3d::Ones() + imuerror_.accscale;
    imu.dtheta = imu.dtheta.cwiseProduct(gyrscale.cwiseInverse());
    imu.dvel   = imu.dvel.cwiseProduct(accscale.cwiseInverse());
}

void GIEngine::insPropagation(IMU &imupre, IMU &imucur) {

    // 对当前IMU数据(imucur)补偿误差, 上一IMU数据(imupre)已经补偿过了
    // compensate imu error to 'imucur', 'imupre' has been compensated
    imuCompensate(imucur);
    // IMU状态更新(机械编排算法)
    // update imustate(mechanization)
    INSMech::insMech(pvapre_, pvacur_, imupre, imucur);

    // 系统噪声传播，姿态误差采用phi角误差模型
    // system noise propagate, phi-angle error model for attitude error
    Eigen::MatrixXd Phi, F, Qd, G;

    // 初始化Phi阵(状态转移矩阵)，F阵，Qd阵(传播噪声阵)，G阵(噪声驱动阵)
    // initialize Phi (state transition), F matrix, Qd(propagation noise) and G(noise driven) matrix
    Phi.resizeLike(Cov_);
    F.resizeLike(Cov_);
    Qd.resizeLike(Cov_);
    G.resize(RANK, NOISERANK);
    Phi.setIdentity();
    F.setZero();
    Qd.setZero();
    G.setZero();

    // 使用上一历元状态计算状态转移矩阵
    // compute state transition matrix using the previous state
    Eigen::Vector2d rmrn;
    Eigen::Vector3d wie_n, wen_n;
    double gravity;
    rmrn    = Earth::meridianPrimeVerticalRadius(pvapre_.pos[0]);
    gravity = Earth::gravity(pvapre_.pos);
    wie_n << WGS84_WIE * cos(pvapre_.pos[0]), 0, -WGS84_WIE * sin(pvapre_.pos[0]);
    wen_n << pvapre_.vel[1] / (rmrn[1] + pvapre_.pos[2]), -pvapre_.vel[0] / (rmrn[0] + pvapre_.pos[2]),
        -pvapre_.vel[1] * tan(pvapre_.pos[0]) / (rmrn[1] + pvapre_.pos[2]);

    Eigen::Matrix3d temp;
    Eigen::Vector3d accel, omega;
    double rmh, rnh;

    rmh   = rmrn[0] + pvapre_.pos[2];
    rnh   = rmrn[1] + pvapre_.pos[2];
    accel = imucur.dvel / imucur.dt;
    omega = imucur.dtheta / imucur.dt;

    // 位置误差
    // position error
    temp.setZero();
    temp(0, 0)                = -pvapre_.vel[2] / rmh;
    temp(0, 2)                = pvapre_.vel[0] / rmh;
    temp(1, 0)                = pvapre_.vel[1] * tan(pvapre_.pos[0]) / rnh;
    temp(1, 1)                = -(pvapre_.vel[2] + pvapre_.vel[0] * tan(pvapre_.pos[0])) / rnh;
    temp(1, 2)                = pvapre_.vel[1] / rnh;
    F.block(P_ID, P_ID, 3, 3) = temp;
    F.block(P_ID, V_ID, 3, 3) = Eigen::Matrix3d::Identity();

    // 速度误差
    // velocity error
    temp.setZero();
    temp(0, 0) = -2 * pvapre_.vel[1] * WGS84_WIE * cos(pvapre_.pos[0]) / rmh -
                 pow(pvapre_.vel[1], 2) / rmh / rnh / pow(cos(pvapre_.pos[0]), 2);
    temp(0, 2) = pvapre_.vel[0] * pvapre_.vel[2] / rmh / rmh - pow(pvapre_.vel[1], 2) * tan(pvapre_.pos[0]) / rnh / rnh;
    temp(1, 0) = 2 * WGS84_WIE * (pvapre_.vel[0] * cos(pvapre_.pos[0]) - pvapre_.vel[2] * sin(pvapre_.pos[0])) / rmh +
                 pvapre_.vel[0] * pvapre_.vel[1] / rmh / rnh / pow(cos(pvapre_.pos[0]), 2);
    temp(1, 2) = (pvapre_.vel[1] * pvapre_.vel[2] + pvapre_.vel[0] * pvapre_.vel[1] * tan(pvapre_.pos[0])) / rnh / rnh;
    temp(2, 0) = 2 * WGS84_WIE * pvapre_.vel[1] * sin(pvapre_.pos[0]) / rmh;
    temp(2, 2) = -pow(pvapre_.vel[1], 2) / rnh / rnh - pow(pvapre_.vel[0], 2) / rmh / rmh +
                 2 * gravity / (sqrt(rmrn[0] * rmrn[1]) + pvapre_.pos[2]);
    F.block(V_ID, P_ID, 3, 3) = temp;
    temp.setZero();
    temp(0, 0)                  = pvapre_.vel[2] / rmh;
    temp(0, 1)                  = -2 * (WGS84_WIE * sin(pvapre_.pos[0]) + pvapre_.vel[1] * tan(pvapre_.pos[0]) / rnh);
    temp(0, 2)                  = pvapre_.vel[0] / rmh;
    temp(1, 0)                  = 2 * WGS84_WIE * sin(pvapre_.pos[0]) + pvapre_.vel[1] * tan(pvapre_.pos[0]) / rnh;
    temp(1, 1)                  = (pvapre_.vel[2] + pvapre_.vel[0] * tan(pvapre_.pos[0])) / rnh;
    temp(1, 2)                  = 2 * WGS84_WIE * cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh;
    temp(2, 0)                  = -2 * pvapre_.vel[0] / rmh;
    temp(2, 1)                  = -2 * (WGS84_WIE * cos(pvapre_.pos(0)) + pvapre_.vel[1] / rnh);
    F.block(V_ID, V_ID, 3, 3)   = temp;
    F.block(V_ID, PHI_ID, 3, 3) = Rotation::skewSymmetric(pvapre_.att.cbn * accel);
    F.block(V_ID, BA_ID, 3, 3)  = pvapre_.att.cbn;
    F.block(V_ID, SA_ID, 3, 3)  = pvapre_.att.cbn * (accel.asDiagonal());

    // 姿态误差
    // attitude error
    temp.setZero();
    temp(0, 0) = -WGS84_WIE * sin(pvapre_.pos[0]) / rmh;
    temp(0, 2) = pvapre_.vel[1] / rnh / rnh;
    temp(1, 2) = -pvapre_.vel[0] / rmh / rmh;
    temp(2, 0) = -WGS84_WIE * cos(pvapre_.pos[0]) / rmh - pvapre_.vel[1] / rmh / rnh / pow(cos(pvapre_.pos[0]), 2);
    temp(2, 2) = -pvapre_.vel[1] * tan(pvapre_.pos[0]) / rnh / rnh;
    F.block(PHI_ID, P_ID, 3, 3) = temp;
    temp.setZero();
    temp(0, 1)                    = 1 / rnh;
    temp(1, 0)                    = -1 / rmh;
    temp(2, 1)                    = -tan(pvapre_.pos[0]) / rnh;
    F.block(PHI_ID, V_ID, 3, 3)   = temp;
    F.block(PHI_ID, PHI_ID, 3, 3) = -Rotation::skewSymmetric(wie_n + wen_n);
    F.block(PHI_ID, BG_ID, 3, 3)  = -pvapre_.att.cbn;
    F.block(PHI_ID, SG_ID, 3, 3)  = -pvapre_.att.cbn * (omega.asDiagonal());

    // IMU零偏误差和比例因子误差，建模成一阶高斯-马尔科夫过程
    // imu bias error and scale error, modeled as the first-order Gauss-Markov process
    F.block(BG_ID, BG_ID, 3, 3) = -1 / options_.imunoise.corr_time * Eigen::Matrix3d::Identity();
    F.block(BA_ID, BA_ID, 3, 3) = -1 / options_.imunoise.corr_time * Eigen::Matrix3d::Identity();
    F.block(SG_ID, SG_ID, 3, 3) = -1 / options_.imunoise.corr_time * Eigen::Matrix3d::Identity();
    F.block(SA_ID, SA_ID, 3, 3) = -1 / options_.imunoise.corr_time * Eigen::Matrix3d::Identity();

    // 系统噪声驱动矩阵
    // system noise driven matrix
    G.block(V_ID, VRW_ID, 3, 3)    = pvapre_.att.cbn;
    G.block(PHI_ID, ARW_ID, 3, 3)  = pvapre_.att.cbn;
    G.block(BG_ID, BGSTD_ID, 3, 3) = Eigen::Matrix3d::Identity();
    G.block(BA_ID, BASTD_ID, 3, 3) = Eigen::Matrix3d::Identity();
    G.block(SG_ID, SGSTD_ID, 3, 3) = Eigen::Matrix3d::Identity();
    G.block(SA_ID, SASTD_ID, 3, 3) = Eigen::Matrix3d::Identity();

    // 状态转移矩阵
    // compute the state transition matrix
    Phi.setIdentity();
    Phi = Phi + F * imucur.dt;

    // 计算系统传播噪声
    // compute system propagation noise
    Qd = G * Qc_ * G.transpose() * imucur.dt;
    Qd = (Phi * Qd * Phi.transpose() + Qd) / 2;

    // EKF预测传播系统协方差和系统误差状态
    // do EKF predict to propagate covariance and error state
    filterPredict(Phi, Qd);
}

void GIEngine::gnssUpdate(GNSS &gnssdata) {

    // IMU位置转到GNSS天线相位中心位置
    // convert IMU position to GNSS antenna phase center position
    Eigen::Vector3d antenna_pos;
    Eigen::Matrix3d Dr, Dr_inv;
    Dr_inv      = Earth::DRi(pvacur_.pos);
    Dr          = Earth::DR(pvacur_.pos);
    antenna_pos = pvacur_.pos + Dr_inv * pvacur_.att.cbn * options_.antlever;

    // GNSS位置测量新息
    // compute GNSS position innovation
    const Eigen::Vector3d dz_full = Dr * (antenna_pos - gnssdata.blh);

    // 构造GNSS位置观测矩阵
    // construct GNSS position measurement matrix
    Eigen::MatrixXd H_gnsspos;
    const bool gnss_xy_only = (options_.gnss_update_mode == GnssPosMeasMode::XY);
    const int gnss_meas_dim = gnss_xy_only ? 2 : 3;
    H_gnsspos.resize(gnss_meas_dim, Cov_.rows());
    H_gnsspos.setZero();
    if (gnss_xy_only) {
        H_gnsspos.block(0, P_ID, 2, 2) = Eigen::Matrix2d::Identity();
        H_gnsspos.block(0, PHI_ID, 2, 3) =
            Rotation::skewSymmetric(pvacur_.att.cbn * options_.antlever).topRows(2);
    } else {
        H_gnsspos.block(0, P_ID, 3, 3)   = Eigen::Matrix3d::Identity();
        H_gnsspos.block(0, PHI_ID, 3, 3) = Rotation::skewSymmetric(pvacur_.att.cbn * options_.antlever);
    }

    // 位置观测噪声阵
    // construct measurement noise matrix
    Eigen::MatrixXd R_gnsspos;
    if (gnss_xy_only) {
        R_gnsspos = gnssdata.std.head<2>().cwiseProduct(gnssdata.std.head<2>()).asDiagonal();
    } else {
        R_gnsspos = gnssdata.std.cwiseProduct(gnssdata.std).asDiagonal();
    }

    Eigen::MatrixXd dz;
    if (gnss_xy_only) {
        dz = dz_full.head<2>();
    } else {
        dz = dz_full;
    }

    // EKF更新协方差和误差状态
    // do EKF update to update covariance and error state
    filterUpdate(dz, H_gnsspos, R_gnsspos);

    // GNSS更新之后设置为不可用
    // Set GNSS invalid after update
    gnssdata.isvalid = false;
}

int GIEngine::isToUpdate(double imutime1, double imutime2, double updatetime) const {

    if (abs(imutime1 - updatetime) < TIME_ALIGN_ERR) {
        // 更新时间靠近imutime1
        // updatetime is near to imutime1
        return 1;
    } else if (abs(imutime2 - updatetime) <= TIME_ALIGN_ERR) {
        // 更新时间靠近imutime2
        // updatetime is near to imutime2
        return 2;
    } else if (imutime1 < updatetime && updatetime < imutime2) {
        // 更新时间在imutime1和imutime2之间, 但不靠近任何一个
        // updatetime is between imutime1 and imutime2, but not near to either
        return 3;
    } else {
        // 更新时间不在imutimt1和imutime2之间，且不靠近任何一个
        // updatetime is not bewteen imutime1 and imutime2, and not near to either.
        return 0;
    }
}

void GIEngine::EKFPredict(Eigen::MatrixXd &Phi, Eigen::MatrixXd &Qd) {

    assert(Phi.rows() == Cov_.rows());
    assert(Qd.rows() == Cov_.rows());

    // 传播系统协方差和误差状态
    // propagate system covariance and error state
    Cov_ = Phi * Cov_ * Phi.transpose() + Qd;
    dx_  = Phi * dx_;
}

void GIEngine::filterPredict(Eigen::MatrixXd &Phi, Eigen::MatrixXd &Qd) {

    switch (options_.filter_scheme) {
    case FilterScheme::ESKF:
        EKFPredict(Phi, Qd);
        break;
    case FilterScheme::UKF: {
        Eigen::MatrixXd X, Xp;
        Eigen::VectorXd Wm, Wc;
        if (!buildUkfSigmaPoints(dx_, Cov_, X, Wm, Wc)) {
            std::cerr << "[GIEngine][UKF] buildUkfSigmaPoints failed at t=" << std::setprecision(10) << timestamp_
                      << ", fallback to ESKF predict." << std::endl;
            EKFPredict(Phi, Qd);
            break;
        }
        const int n = dx_.rows();
        const int ns = X.cols();
        Xp.resize(n, ns);
        for (int i = 0; i < ns; ++i) {
            Xp.col(i) = Phi * X.col(i);
        }
        Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < ns; ++i) {
            x_pred += Wm(i) * Xp.col(i);
        }
        Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < ns; ++i) {
            Eigen::VectorXd d = Xp.col(i) - x_pred;
            P_pred += Wc(i) * (d * d.transpose());
        }
        P_pred += Qd;
        regularizeCovariance(P_pred);
        dx_  = x_pred;
        Cov_ = P_pred;
        break;
    }
    case FilterScheme::SR_UKF: {
        Eigen::MatrixXd X, Xp;
        Eigen::VectorXd Wm, Wc;
        if (!buildUkfSigmaPoints(dx_, Cov_, X, Wm, Wc)) {
            std::cerr << "[GIEngine][SR-UKF] buildUkfSigmaPoints failed at t=" << std::setprecision(10) << timestamp_
                      << ", fallback to ESKF predict." << std::endl;
            EKFPredict(Phi, Qd);
            break;
        }
        const int n  = dx_.rows();
        const int ns = X.cols();
        Xp.resize(n, ns);
        for (int i = 0; i < ns; ++i) {
            Xp.col(i) = Phi * X.col(i);
        }
        Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < ns; ++i) {
            x_pred += Wm(i) * Xp.col(i);
        }
        Eigen::MatrixXd D(n, ns);
        for (int i = 0; i < ns; ++i) {
            D.col(i) = Xp.col(i) - x_pred;
        }
        Eigen::MatrixXd P_pred;
        if (!srCovarianceFromSigmaDeviations(D, Wc, Qd, P_pred)) {
            P_pred = Eigen::MatrixXd::Zero(n, n);
            for (int i = 0; i < ns; ++i) {
                P_pred += Wc(i) * (D.col(i) * D.col(i).transpose());
            }
            P_pred += Qd;
        }
        regularizeCovariance(P_pred);
        dx_  = x_pred;
        Cov_ = P_pred;
        break;
    }
    case FilterScheme::ADAPTIVE_UKF:
    case FilterScheme::ROBUST_UKF:
    default:
        EKFPredict(Phi, Qd);
        break;
    }
}

void GIEngine::EKFUpdate(Eigen::MatrixXd &dz, Eigen::MatrixXd &H, Eigen::MatrixXd &R) {

    assert(H.cols() == Cov_.rows());
    assert(dz.rows() == H.rows());
    assert(dz.rows() == R.rows());
    assert(dz.cols() == 1);

    // 计算Kalman增益
    // Compute Kalman Gain
    Eigen::MatrixXd temp = H * Cov_ * H.transpose() + R;
    Eigen::MatrixXd temp_inv = temp.inverse();
    Eigen::MatrixXd K        = Cov_ * H.transpose() * temp_inv;

    // 更新系统误差状态和协方差
    // update system error state and covariance
    Eigen::MatrixXd I;
    I.resizeLike(Cov_);
    I.setIdentity();
    I = I - K * H;
    // 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz;
    // if state feedback is performed after every update, dx_ is always zero before the update
    // the following formula can be simplified as : dx_ = K * dz;
    Eigen::VectorXd innov = dz - H * dx_;
    int gate_dim          = dz.rows();
    last_nis_             = computeNISByGateMode(innov, temp, dz.rows(), gate_dim);
    ++last_nis_seq_;
    if (shouldRejectUpdateByNIS(last_nis_, gate_dim, &innov, &temp, &R)) {
        return;
    }
    dx_  = dx_ + K * innov;
    Cov_ = I * Cov_ * I.transpose() + K * R * K.transpose();
}

void GIEngine::filterUpdate(Eigen::MatrixXd &dz, Eigen::MatrixXd &H, Eigen::MatrixXd &R) {

    switch (options_.filter_scheme) {
    case FilterScheme::ESKF:
        EKFUpdate(dz, H, R);
        break;
    case FilterScheme::UKF: {
        Eigen::MatrixXd X;
        Eigen::VectorXd Wm, Wc;
        if (!buildUkfSigmaPoints(dx_, Cov_, X, Wm, Wc)) {
            std::cerr << "[GIEngine][UKF] buildUkfSigmaPoints failed at t=" << std::setprecision(10) << timestamp_
                      << ", fallback to ESKF update." << std::endl;
            EKFUpdate(dz, H, R);
            break;
        }
        const int n = dx_.rows();
        const int m = dz.rows();
        const int ns = X.cols();

        Eigen::MatrixXd Z(m, ns);
        for (int i = 0; i < ns; ++i) {
            Z.col(i) = H * X.col(i);
        }

        Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(m);
        for (int i = 0; i < ns; ++i) {
            z_pred += Wm(i) * Z.col(i);
        }

        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(m, m);
        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n, m);
        for (int i = 0; i < ns; ++i) {
            Eigen::VectorXd dxs = X.col(i) - dx_;
            Eigen::VectorXd dzs = Z.col(i) - z_pred;
            S += Wc(i) * (dzs * dzs.transpose());
            Pxz += Wc(i) * (dxs * dzs.transpose());
        }
        S += R;
        S = (S + S.transpose()) * 0.5;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            EKFUpdate(dz, H, R);
            break;
        }
        Eigen::MatrixXd K = Pxz * ldlt.solve(Eigen::MatrixXd::Identity(m, m));
        Eigen::VectorXd innov = dz - z_pred;
        int gate_dim          = m;
        last_nis_             = computeNISByGateMode(innov, S, m, gate_dim);
        ++last_nis_seq_;
        if (shouldRejectUpdateByNIS(last_nis_, gate_dim, &innov, &S, &R)) {
            break;
        }
        dx_               = dx_ + K * innov;
        Cov_              = Cov_ - K * S * K.transpose();
        regularizeCovariance(Cov_);
        break;
    }
    case FilterScheme::SR_UKF: {
        Eigen::MatrixXd X;
        Eigen::VectorXd Wm, Wc;
        if (!buildUkfSigmaPoints(dx_, Cov_, X, Wm, Wc)) {
            std::cerr << "[GIEngine][SR-UKF] buildUkfSigmaPoints failed at t=" << std::setprecision(10) << timestamp_
                      << ", fallback to ESKF update." << std::endl;
            EKFUpdate(dz, H, R);
            break;
        }
        const int n  = dx_.rows();
        const int m  = dz.rows();
        const int ns = X.cols();

        Eigen::MatrixXd Z(m, ns);
        for (int i = 0; i < ns; ++i) {
            Z.col(i) = H * X.col(i);
        }

        Eigen::VectorXd z_pred = Eigen::VectorXd::Zero(m);
        for (int i = 0; i < ns; ++i) {
            z_pred += Wm(i) * Z.col(i);
        }

        Eigen::MatrixXd Dz(m, ns), Dx(n, ns);
        for (int i = 0; i < ns; ++i) {
            Dz.col(i) = Z.col(i) - z_pred;
            Dx.col(i) = X.col(i) - dx_;
        }

        Eigen::MatrixXd Szz;
        if (!srCovarianceFromSigmaDeviations(Dz, Wc, R, Szz)) {
            Szz = Eigen::MatrixXd::Zero(m, m);
            for (int i = 0; i < ns; ++i) {
                Szz += Wc(i) * (Dz.col(i) * Dz.col(i).transpose());
            }
            Szz += R;
        }
        regularizeCovariance(Szz);

        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n, m);
        for (int i = 0; i < ns; ++i) {
            Pxz += Wc(i) * (Dx.col(i) * Dz.col(i).transpose());
        }

        Eigen::LDLT<Eigen::MatrixXd> ldlt(Szz);
        if (ldlt.info() != Eigen::Success) {
            EKFUpdate(dz, H, R);
            break;
        }
        Eigen::MatrixXd K = Pxz * ldlt.solve(Eigen::MatrixXd::Identity(m, m));
        Eigen::VectorXd innov = dz - z_pred;
        int gate_dim          = m;
        last_nis_             = computeNISByGateMode(innov, Szz, m, gate_dim);
        ++last_nis_seq_;
        if (shouldRejectUpdateByNIS(last_nis_, gate_dim, &innov, &Szz, &R)) {
            break;
        }
        dx_               = dx_ + K * innov;
        Cov_              = Cov_ - K * Szz * K.transpose();
        regularizeCovariance(Cov_);
        break;
    }
    case FilterScheme::ADAPTIVE_UKF:
    case FilterScheme::ROBUST_UKF:
    default:
        EKFUpdate(dz, H, R);
        break;
    }
}

bool GIEngine::shouldRejectUpdateByNIS(double nis, int meas_dim, const Eigen::VectorXd *innov, const Eigen::MatrixXd *S,
                                       const Eigen::MatrixXd *R) {
    if (!options_.gnss_nis_gate_enable) {
        return false;
    }
    const bool should_log = (nis_reject_count_ < 10) || (((nis_reject_count_ + 1) % 20) == 0);
    if (!std::isfinite(nis)) {
        ++nis_reject_count_;
        std::ostringstream oss;
        oss << "[GIEngine][NIS-GATE] reject non-finite NIS at t=" << std::fixed << std::setprecision(3) << timestamp_
            << " (dim=" << meas_dim << ", reject_count=" << nis_reject_count_ << ")";
        if (should_log && innov != nullptr && innov->rows() >= meas_dim) {
            oss << " innov=" << formatGateVector(innov->head(meas_dim), meas_dim);
        }
        if (should_log && S != nullptr && S->rows() >= meas_dim && S->cols() >= meas_dim) {
            oss << " sqrtS=" << formatGateVector(gateStdFromDiag(S->topLeftCorner(meas_dim, meas_dim), meas_dim), meas_dim);
        }
        if (should_log && R != nullptr && R->rows() >= meas_dim && R->cols() >= meas_dim) {
            oss << " sqrtR=" << formatGateVector(gateStdFromDiag(R->topLeftCorner(meas_dim, meas_dim), meas_dim), meas_dim);
        }
        std::cerr << oss.str() << std::endl;
        return true;
    }
    if (nis <= options_.gnss_nis_gate_threshold) {
        return false;
    }
    ++nis_reject_count_;
    if (should_log) {
        std::ostringstream oss;
        oss << "[GIEngine][NIS-GATE] reject GNSS update at t=" << std::fixed << std::setprecision(3) << timestamp_
            << " nis=" << std::setprecision(6) << nis << " > " << options_.gnss_nis_gate_threshold
            << " (dim=" << meas_dim << ", reject_count=" << nis_reject_count_ << ")";
        if (innov != nullptr && innov->rows() >= meas_dim) {
            const Eigen::VectorXd innov_gate = innov->head(meas_dim);
            oss << " innov=" << formatGateVector(innov_gate, meas_dim)
                << " |innov|=" << std::fixed << std::setprecision(3) << innov_gate.norm();
        }
        if (S != nullptr && S->rows() >= meas_dim && S->cols() >= meas_dim) {
            const Eigen::VectorXd std_s = gateStdFromDiag(S->topLeftCorner(meas_dim, meas_dim), meas_dim);
            oss << " sqrtS=" << formatGateVector(std_s, meas_dim);
            if (innov != nullptr && innov->rows() >= meas_dim) {
                Eigen::VectorXd innov_sigma = innov->head(meas_dim);
                for (int i = 0; i < meas_dim; ++i) {
                    const double sigma = std_s(i);
                    innov_sigma(i)     = (std::isfinite(sigma) && sigma > 1e-9) ? innov_sigma(i) / sigma
                                                                                 : std::numeric_limits<double>::quiet_NaN();
                }
                oss << " innov/sqrtS=" << formatGateVector(innov_sigma, meas_dim);
            }
        }
        if (R != nullptr && R->rows() >= meas_dim && R->cols() >= meas_dim) {
            oss << " sqrtR="
                << formatGateVector(gateStdFromDiag(R->topLeftCorner(meas_dim, meas_dim), meas_dim), meas_dim);
        }
        std::cerr << oss.str() << std::endl;
    }
    return true;
}

double GIEngine::computeNISByGateMode(const Eigen::VectorXd &innov, const Eigen::MatrixXd &S, int meas_dim, int &gate_dim) const {
    gate_dim = meas_dim;
    if (meas_dim <= 0) {
        return std::numeric_limits<double>::infinity();
    }

    if (options_.gnss_nis_gate_mode == GnssPosMeasMode::XY && meas_dim >= 2) {
        gate_dim = 2;
    } else if (options_.gnss_nis_gate_mode == GnssPosMeasMode::XYZ && meas_dim >= 3) {
        gate_dim = 3;
    } else {
        gate_dim = meas_dim;
    }

    if (innov.rows() < gate_dim || S.rows() < gate_dim || S.cols() < gate_dim) {
        return std::numeric_limits<double>::infinity();
    }

    const Eigen::VectorXd innov_gate = innov.head(gate_dim);
    const Eigen::MatrixXd S_gate     = S.topLeftCorner(gate_dim, gate_dim);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S_gate);
    if (ldlt.info() != Eigen::Success) {
        return std::numeric_limits<double>::infinity();
    }
    return (innov_gate.transpose() * ldlt.solve(innov_gate))(0, 0);
}

bool GIEngine::buildUkfSigmaPoints(const Eigen::VectorXd &x, const Eigen::MatrixXd &P, Eigen::MatrixXd &X,
                                   Eigen::VectorXd &Wm, Eigen::VectorXd &Wc) const {

    const int n = x.rows();
    if (P.rows() != n || P.cols() != n || n <= 0) {
        return false;
    }

    const double alpha = options_.ukf_alpha;
    const double beta  = options_.ukf_beta;
    const double kappa = options_.ukf_kappa;
    const double lambda = alpha * alpha * (n + kappa) - n;
    const double c = n + lambda;
    if (c <= 1e-12) {
        return false;
    }

    Eigen::MatrixXd P_reg = (P + P.transpose()) * 0.5;
    Eigen::LLT<Eigen::MatrixXd> llt;
    bool ok = false;
    double jitter = 1e-12;
    for (int k = 0; k < 8; ++k) {
        llt.compute(P_reg);
        if (llt.info() == Eigen::Success) {
            ok = true;
            break;
        }
        P_reg.diagonal().array() += jitter;
        jitter *= 10.0;
    }
    if (!ok) {
        return false;
    }

    Eigen::MatrixXd S = llt.matrixL();
    const int ns      = 2 * n + 1;
    X.resize(n, ns);
    Wm.resize(ns);
    Wc.resize(ns);

    X.col(0) = x;
    const double gamma = std::sqrt(c);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd col = gamma * S.col(i);
        X.col(i + 1)        = x + col;
        X.col(i + 1 + n)    = x - col;
    }

    Wm.setConstant(1.0 / (2.0 * c));
    Wc.setConstant(1.0 / (2.0 * c));
    Wm(0) = lambda / c;
    Wc(0) = lambda / c + (1.0 - alpha * alpha + beta);
    return true;
}

bool GIEngine::srCovarianceFromSigmaDeviations(const Eigen::MatrixXd &D, const Eigen::VectorXd &Wc, const Eigen::MatrixXd &Q,
                                               Eigen::MatrixXd &P) const {

    const int n = D.rows();
    const int ns = D.cols();
    if (Wc.size() != ns || Q.rows() != n || Q.cols() != n) {
        return false;
    }

    // 若出现负权重，严格SR-UKF需要cholupdate/downdate；当前实现回退到普通求和以保证正确性
    for (int i = 0; i < ns; ++i) {
        if (Wc(i) < 0.0) {
            return false;
        }
    }

    Eigen::MatrixXd Qsym = (Q + Q.transpose()) * 0.5;
    Eigen::LLT<Eigen::MatrixXd> qllt;
    bool qok = false;
    double jitter = 1e-12;
    for (int k = 0; k < 8; ++k) {
        qllt.compute(Qsym);
        if (qllt.info() == Eigen::Success) {
            qok = true;
            break;
        }
        Qsym.diagonal().array() += jitter;
        jitter *= 10.0;
    }
    if (!qok) {
        return false;
    }

    int extra_cols = 0;
    for (int i = 0; i < ns; ++i) {
        if (Wc(i) > 0.0) {
            ++extra_cols;
        }
    }
    Eigen::MatrixXd U(n, n + extra_cols);
    U.leftCols(n) = qllt.matrixL();
    int col = n;
    for (int i = 0; i < ns; ++i) {
        if (Wc(i) <= 0.0) {
            continue;
        }
        U.col(col++) = std::sqrt(Wc(i)) * D.col(i);
    }

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(U.transpose());
    Eigen::MatrixXd qr_mat = qr.matrixQR();
    Eigen::MatrixXd R = qr_mat.topLeftCorner(n, n).template triangularView<Eigen::Upper>();
    Eigen::MatrixXd S = R.transpose();
    P = S * S.transpose();
    return true;
}

void GIEngine::regularizeCovariance(Eigen::MatrixXd &P) const {

    P = (P + P.transpose()) * 0.5;
    constexpr double min_diag = 1e-16;
    for (int i = 0; i < P.rows(); ++i) {
        if (P(i, i) < min_diag) {
            P(i, i) = min_diag;
        }
    }
}

void GIEngine::stateFeedback() {

    Eigen::Vector3d vectemp;

    // 位置误差反馈
    // posisiton error feedback
    Eigen::Vector3d delta_r = dx_.block(P_ID, 0, 3, 1);
    Eigen::Matrix3d Dr_inv  = Earth::DRi(pvacur_.pos);
    pvacur_.pos -= Dr_inv * delta_r;

    // 速度误差反馈
    // velocity error feedback
    vectemp = dx_.block(V_ID, 0, 3, 1);
    pvacur_.vel -= vectemp;

    // 姿态误差反馈
    // attitude error feedback
    vectemp                = dx_.block(PHI_ID, 0, 3, 1);
    Eigen::Quaterniond qpn = Rotation::rotvec2quaternion(vectemp);
    pvacur_.att.qbn        = qpn * pvacur_.att.qbn;
    pvacur_.att.cbn        = Rotation::quaternion2matrix(pvacur_.att.qbn);
    pvacur_.att.euler      = Rotation::matrix2euler(pvacur_.att.cbn);

    // IMU零偏误差反馈
    // IMU bias error feedback
    vectemp = dx_.block(BG_ID, 0, 3, 1);
    imuerror_.gyrbias += vectemp;
    vectemp = dx_.block(BA_ID, 0, 3, 1);
    imuerror_.accbias += vectemp;

    // IMU比例因子误差反馈
    // IMU sacle error feedback
    vectemp = dx_.block(SG_ID, 0, 3, 1);
    imuerror_.gyrscale += vectemp;
    vectemp = dx_.block(SA_ID, 0, 3, 1);
    imuerror_.accscale += vectemp;

    // 误差状态反馈到系统状态后,将误差状态清零
    // set 'dx' to zero after feedback error state to system state
    dx_.setZero();
}

NavState GIEngine::getNavState() {

    NavState state;

    state.pos      = pvacur_.pos;
    state.vel      = pvacur_.vel;
    state.euler    = pvacur_.att.euler;
    state.imuerror = imuerror_;

    return state;
}
