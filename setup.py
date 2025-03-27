from setuptools import setup
from glob import glob
import os

package_name = 'auto_shepherd_whistle_teleop'
pkg = package_name

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{pkg}']),
        (f"share/{package_name}/config", glob(os.path.join('config', '*.rviz'))),
        (f"share/{package_name}/config", glob(os.path.join('config', '*.yaml'))),
        (f"share/{package_name}/templates", glob(os.path.join('templates', '*.png'))),
        (f'share/{pkg}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='primordia@live.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'audio_filter.py = auto_shepherd_whistle_teleop.audio_filter:main',
            'pitch_decoder.py = auto_shepherd_whistle_teleop.pitch_decoder:main',
            'template_matcher.py = auto_shepherd_whistle_teleop.template_matcher:main',
        ],
    },
)
