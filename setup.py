from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='TLTools',
      version='2.1.5',
      description='Tools for acquiring and processing Ultrasonic NDT data',
      author='Timothy Lardner, Jerzy Dziewierz',
      author_email='jerzy.dziewierz@strath.ac.uk',
      url='https://github.com/jerzydziewierz',
      license="Not to be used by anyone",
      packages=['TLTools'],
      install_requires=[
          'pycuda', 'matplotlib', 'numpy', 'viscm', 'scipy', 'pillow', 'pythonnet',
      ],
      long_description='cueTFM part of the cueART',
      package_data={'': ['TFMKernel.cu']},
      classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
    ],
     )
