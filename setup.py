from setuptools import setup
from glob import glob
import os

package_name = 'auto_shepherd_whistle_teleop'
pkg = package_name


def package_files(directory):
    paths = []
    for (root, dirs, files) in os.walk(directory):
        for filename in files:
            paths.append(os.path.join(root, filename))
    return paths

template_files = package_files('templates')

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{pkg}']),
        # Install configurations
        (f"share/{pkg}/config", glob(os.path.join('config', '*.rviz'))),
        (f"share/{pkg}/config", glob(os.path.join('config', '*.yaml'))),
        # Install samples
        (f"share/{pkg}/templates/stop", glob(os.path.join('templates/stop', '*.png'))),
        (f"share/{pkg}/templates/forward", glob(os.path.join('templates/forward', '*.png'))),
        (f"share/{pkg}/templates/backward", glob(os.path.join('templates/backward', '*.png'))),
        (f"share/{pkg}/templates/left", glob(os.path.join('templates/left', '*.png'))),
        (f"share/{pkg}/templates/right", glob(os.path.join('templates/right', '*.png'))),
        # Install package
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
