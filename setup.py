"""Setup pylacompass package"""
from setuptools import setup
import os

PACKAGENAME = 'pylacompass'

setup(name=PACKAGENAME,
      version='0.0.0',
      description='python routines to process simulation data of LA-COMPASS code',
      long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
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
          'matplotlib',
          ],
      zip_safe=False
      )
