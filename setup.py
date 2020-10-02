from distutils.core import setup
from os import path
import os

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))

description = 'A computer vision library for 360-degree cameras'

try:
    with open(path.join(this_directory, 'omnicv/README.rst'), "r") as f:
        long_description = f.read()
except:
    long_description = description

version = '1.1.3'

setup(
  name = 'omnicv',
  packages = ['omnicv'],
  version = version,
  license='MIT',        
  description = description,
  long_description=long_description,
  long_description_content_type="text/x-rst",
  author = 'Kaustubh Sadekar, Leena Vachhani, Abhishek Gupta',
  url = 'https://kaustubh-sadekar.github.io/OmniCV-Lib/index.html',
  download_url = 'https://github.com/kaustubh-sadekar/OmniCV-Lib/archive/v_%s.tar.gz'%version,
  keywords = ['Deep Learning', 'Helper functions'],
  package_data={'':['README.rst']},
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',   
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
