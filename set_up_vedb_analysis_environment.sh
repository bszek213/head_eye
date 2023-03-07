# Environment setup


# Check for interactive environment
set -e

if [[ $- == *i* ]]
then
	echo 'Interactive mode detected.'
else
	echo 'Not in interactive mode! Please run with `bash -i set_up_vedb_analysis_environment.sh`'
	exit
fi

# Assure g++ is installed
if [ `which g++` ];
then
	echo ">>> g++ found"
else
	echo ">>> installing g++ ..."
	sudo apt-get install g++
fi

export name=vedb_analysis38

# Assure mamba is installed
if hash mamba 2>/dev/null; 
then
	echo ">>> Found mamba installed."
	mamba update mamba
else
	conda install mamba -n base -c conda-forge
	mamba update mamba
fi

# Create initial environment
ENVS=$(conda env list | awk '{print $name}' )
if [[ $ENVS = *"$name"* ]]; 
then
	echo "Found environment $name"
else
	echo "Creating new vedb_analysis environment..."
	mamba env create -f environment_vedb_analysis.yml
fi;

# Activate it
conda activate $name

# Try to do this: (check OS first)
export unamestr=`uname`

# Make source code directory for git libraries
if [ -d ~/Code2 ]
	then
		echo "Code2 directory found."
	else
		echo "Creating ~/Code2 directory."
		mkdir ~/Code2
fi
cd ~/Code2/

# TODO: make this flexible for how git is configured: ssh keys or whatever
# Load list of libraries to install from which repos; should be mostly (all) vedb
# Retrieve and install libraries from git...
# ... for loading files:
echo "========================================="
echo ">>> Installing file_io"
if [ -d file_io ]
then
	:
else
	git clone git@github.com:vedb/file_io.git
fi
cd file_io
python setup.py install
cd ..

# ... for plotting:
echo "========================================="
echo ">>> Installing plot_utils"
if [ -d plot_utils ]
then
	:
else
	git clone git@github.com:piecesofmindlab/plot_utils.git
fi
cd plot_utils
python setup.py install
cd ..

# ... for gaze analysis:
echo "========================================="
echo ">>> Installing vedb-gaze"
if [ -d vedb-gaze ]
then
	:
else
	git clone git@github.com:vedb/vedb-gaze.git
fi
cd vedb-gaze
python setup.py install
cd ..

# ... for database retrieval & storage:
echo "========================================="
echo ">>> Installing vedb-store"
if [ -d vedb-store ]
then
	:
else
	git clone git@github.com:vedb/vedb-store.git
fi
cd vedb-store
python setup.py install
cd ..

# ... helper library for database retrieval & storage:
echo "========================================="
echo ">>> Installing docdb_lite"
if [ -d docdb_lite ]
then
	:
else
	git clone git@github.com:vedb/docdb_lite.git
fi
cd docdb_lite
python setup.py install
cd ..


# ... for odometry processing:
echo "========================================="
echo ">>> Installing odo-py"
if [ -d vedb-odometry ]
then
	:
else
	git clone git@github.com:bszek213/odopy.git
fi
cd odopy
git checkout main
python setup.py install
cd ..

# TEMP: fork of pupil_detectors that will install using anaconda-installed binaries for opencv
echo "========================================="
echo ">>> Installing pupil_detectors"
if [ -d pupil-detectors-ml ]
then
	:
else
	git clone git@github.com:marklescroart/pupil-detectors.git pupil-detectors-ml
fi
cd pupil-detectors-ml
python setup.py install
cd ..

# ... thinplate spline library
echo "========================================="
echo ">>> Installing thinplate"
if [ -d py-thin-plate-spline ]
then
	:
else
	git clone git@github.com:cheind/py-thin-plate-spline.git
fi
cd py-thin-plate-spline
python setup.py install
cd ..

# ... pydra (for pipeline stuff)
echo "========================================"
echo ">>> Installing pydra"
if [ -d pydra ]
then
	:
else
	git clone git@github.com:nipype/pydra.git
fi
cd pydra
#python setup.py install #deprecated according to github page
pip install -e ".[dev]"
cd ..

# ... helper library for database retrieval & storage:
#if [ -d vm_preproc ]
#then
#	:
#else
#	git clone https://github.com/piecesofmindlab/vm_preproc
#fi
#cd vm_preproc/python
#python setup.py install
#cd ../..

cd ~

ipython kernel install --user --name $name
