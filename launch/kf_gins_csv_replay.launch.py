from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("kf_gins")
    default_params = os.path.join(pkg_share, "config", "params_hangzhou_town_AB_xyz.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to params yaml for replay",
            ),
            DeclareLaunchArgument(
                "node_name",
                default_value="kf_gins_node",
                description="ROS node name used for parameter lookup in params_file",
            ),
            DeclareLaunchArgument(
                "imu_csv",
                default_value="",
                description="Path to exported imu.csv",
            ),
            DeclareLaunchArgument(
                "gnss_csv",
                default_value="",
                description="Path to exported gnss.csv",
            ),
            DeclareLaunchArgument(
                "replay_nav_csv",
                default_value="",
                description="Optional output path for replayed navigation csv",
            ),
            DeclareLaunchArgument(
                "replay_residual_csv",
                default_value="",
                description="Optional output path for replay residual csv",
            ),
            DeclareLaunchArgument(
                "replay_use_first_gnss_init",
                default_value="true",
                description="Initialize replay filter position from first GNSS sample",
            ),
            DeclareLaunchArgument(
                "replay_skip_pregate_rejected",
                default_value="true",
                description="Skip GNSS rows marked as pre-gate rejected",
            ),
            DeclareLaunchArgument(
                "replay_match_max_dt_sec",
                default_value="0.2",
                description="Maximum GNSS/IMU time difference allowed for residual matching",
            ),
            DeclareLaunchArgument(
                "replay_gnss_time_shift_sec",
                default_value="0.0",
                description="Additional GNSS time shift applied when loading gnss.csv",
            ),
            DeclareLaunchArgument(
                "replay_gnss_std_scale_xy",
                default_value="1.0",
                description="Extra XY GNSS std scale applied when loading gnss.csv",
            ),
            DeclareLaunchArgument(
                "replay_gnss_std_scale_z",
                default_value="1.0",
                description="Extra Z GNSS std scale applied when loading gnss.csv",
            ),
            Node(
                package="kf_gins",
                executable="kf_gins_csv_replay",
                name=LaunchConfiguration("node_name"),
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "imu_csv": LaunchConfiguration("imu_csv"),
                        "gnss_csv": LaunchConfiguration("gnss_csv"),
                        "replay_nav_csv": LaunchConfiguration("replay_nav_csv"),
                        "replay_residual_csv": LaunchConfiguration("replay_residual_csv"),
                        "replay_use_first_gnss_init": LaunchConfiguration("replay_use_first_gnss_init"),
                        "replay_skip_pregate_rejected": LaunchConfiguration("replay_skip_pregate_rejected"),
                        "replay_match_max_dt_sec": LaunchConfiguration("replay_match_max_dt_sec"),
                        "replay_gnss_time_shift_sec": LaunchConfiguration("replay_gnss_time_shift_sec"),
                        "replay_gnss_std_scale_xy": LaunchConfiguration("replay_gnss_std_scale_xy"),
                        "replay_gnss_std_scale_z": LaunchConfiguration("replay_gnss_std_scale_z"),
                    },
                ],
            ),
        ]
    )
