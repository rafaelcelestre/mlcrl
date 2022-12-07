# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2021 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

__authors__ = ["R. Celestre, J. Reyes Herrera, M. Sanchez del Rio"]
__license__ = "MIT"
__date__ = "06/04/2021"



from setuptools import setup

setup(name='mlcrl',
    version='0.0.2',
    description='CRLs and more',
    author='R. Celestre, J. Reyes Herrera, M. Sanchez del Rio',
    author_email='rafael.celestre@esrf.fr',
    url='https://github.com/rafaelcelestre/mlcrl/',
    packages=['mlcrl',
        'mlcrl.phasenet',
        ],
    install_requires=[
        'csbdeep', 'oasys-srwpy', 'oasys-barc4ro'
        ],
    package_data={
        },
    setup_requires=[
        'setuptools',
        ],
    )
