from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("kf_gins")
    default_params = os.path.join(pkg_share, "config", "params_dataset.yaml")

    node = Node(
        package="kf_gins",
        executable="kf_gins_node",
        name="kf_gins_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to params.yaml",
            ),
            node,
        ]
    )
