from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('meshify'),
        'config',
        'fast_grid_mesh.yaml',
    ])
    backend = LaunchConfiguration('backend')
    return LaunchDescription([
        DeclareLaunchArgument(
            'backend',
            default_value='cpu',
            description='Mesh backend: cpu, openmp, or cuda',
        ),
        Node(
            package='meshify',
            executable='fast_grid_mesh_node',
            name='fast_grid_mesh_node',
            output='screen',
            parameters=[config, {'backend': backend}],
        ),
    ])
