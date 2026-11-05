from setuptools import find_packages, setup

package_name = 'yolo_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jianmo',
    maintainer_email='as13432216642@163.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_publisher = yolo_ros.image_publisher:main',
            'yolo_detector = yolo_ros.yolo_detector:main',
            'video_publisher = yolo_ros.video_publisher:main',
        ],
    },
)
