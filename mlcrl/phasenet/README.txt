This directory contains the code phasenet: https://github.com/mpicbg-csbd/phasenet

It has been copied locally for better manipulation


Some dependencies installed:

/Applications/Oasys1.2.app/Contents/MacOS/PythonApp -m pip install jupyterlab

/Applications/Oasys1.2.app/Contents/MacOS/PythonApp -m pip install --upgrade pip
/Applications/Oasys1.2.app/Contents/MacOS/PythonApp -m pip install tensorflow

git clone https://github.com/mpicbg-csbd/phasenet
/Applications/Oasys1.2.app/Contents/MacOS/PythonApp -m pip install -e . --no-deps --no-binary :all:
/Applications/Oasys1.2.app/Contents/MacOS/PythonApp -m pip install csbdeep


Some modifications done after:

https://stackoverflow.com/questions/53135439/issue-with-add-method-in-tensorflow-attributeerror-module-tensorflow-python