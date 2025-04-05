from setuptools import setup
from glob import glob
import os

package_name = 'auto_shepherd_whistle_teleop'
pkg = package_name


# Dynamically collect all .png files in the templates directory and its subdirectories
data_files = []
for root, _, files in os.walk('templates'):
    png_files = [os.path.join(root, f) for f in files if f.endswith('.png')]
    if png_files:
        # This will preserve the subdirectory structure under share/<pkg>/templates/
        dest = os.path.join('share', package_name, root)
        data_files.append((dest, png_files))

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{pkg}']),
        (f"share/{pkg}/config", glob(os.path.join('config', '*.rviz'))),
        (f"share/{pkg}/config", glob(os.path.join('config', '*.yaml'))),
        (f'share/{pkg}', ['package.xml']),
    ] + data_files,
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
            'template_creator.py = auto_shepherd_whistle_teleop.template_creator:main',
            'template_matcher.py = auto_shepherd_whistle_teleop.template_matcher:main'
        ],
    },
)
