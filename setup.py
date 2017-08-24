from setuptools import setup
import os

PACKAGENAME='lanl'
VERSION='0.0.1'

# define custom build class

with open(os.path.join(PACKAGENAME, '_version.py'), 'w') as f:
    f.write('__version__ = \'{}\''.format(VERSION))


setup(name=PACKAGENAME,
    version=VERSION,
    description='python routines to process simulation data of LA-COMPASS code by S. Li (LANL, Los Alamos, NM)',
    long_description=open(os.path.join(os.path.dirname(__file__),'README.md')).read(),
    url='http://www.til-birnstiel.de',
    author='Til Birnstiel',
    author_email='birnstiel@me.com',
    license='GPLv3',
    packages=[PACKAGENAME],
    include_package_data=True,
    install_requires=[
        'astropy',
        'scipy',
        'numpy',
        'matplotlib'
        ],
    zip_safe=False
    )
