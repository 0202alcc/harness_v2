from setuptools import find_packages, setup

package_name = "harness_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/harness_system.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Harness maintainers",
    maintainer_email="maintainers@example.invalid",
    description="ROS 2 nodes for the always-on Harness.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "attention_arbiter = harness_nodes.attention_arbiter:main",
            "perception = harness_nodes.perception:main",
            "session_projector = harness_nodes.session_projector:main",
            "inference_action_server = harness_nodes.inference_action_server:main",
            "session_orchestrator = harness_nodes.session_orchestrator:main",
            "ros_web_gateway = harness_nodes.ros_web_gateway:main",
            "lifecycle_manager = harness_nodes.lifecycle_manager:main",
        ],
    },
)
