# Check for interactive environment - took this from Mark. Need this or conda will not activate
set -e
if [[ $- == *i* ]]
then
	echo 'Interactive mode detected.'
else
	echo 'Not in interactive mode! Please run with `bash -i run_scripts.sh /path/to/folder/`'
	exit
fi
echo "conda activate vedb_analysis_dlc"
# Activate it
conda activate vedb_analysis_dlc
conda env list
echo "Data path: $1"
echo "run pylid script"
# Run the pylids script with an argument
python pylid_execute.py "$1"
echo "conda deactivate vedb_analysis_dlc"
# # Deactivate the current environment
conda deactivate
echo "conda activate vedb_analysis_38"
# Activate the 'vedb_analysis38' environment
conda activate vedb_analysis38
conda env list
echo "Run marker detection, head analysis, and validation marker scipt"
# Run the marker and validation script with an argument
python eye_head.py "$1"