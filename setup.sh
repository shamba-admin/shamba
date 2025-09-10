#! /bin/bash

set -e

GREEN='\033[1;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
echo "  ____   _   _     _     __  __  ____     _    "  
echo " / ___| | | | |   / \   |  \/  || __ )   / \   "  
echo " \___ \ | |_| |  / _ \  | |\/| ||  _ \  / _ \  "  
echo "  ___) ||  _  | / ___ \ | |  | || |_) |/ ___ \ "  
echo " |____/ |_| |_|/_/   \_\|_|  |_||____//_/   \_\ "  
echo ""
echo -e "Welcome to the Shamba setup!${NC}"

echo -e "${YELLOW}Do you already have Miniconda for Python 3.7 installed on your computer? [y/n]${NC}"
read miniconda_exists 
if [ $miniconda_exists != "y" ]
then
    echo -e "${RED}Please first install Miniconda for Python 3.7 from the Anaconda official website: https://docs.conda.io/en/latest/miniconda.html"
    echo -e "Once done, please start this script in a 'bash with Anaconda initialized'.${NC}"
    exit
else 
    echo -e "${NC}Please ensure you are running this script in a 'bash with Anaconda initialized' before continuing.${NC}"  
fi

echo -e "${YELLOW}Please enter a name for the shamba virtual environment we are going to create for you${NC}"
read shamba_env_name  

env-config() {
conda config --set channel_priority strict
conda create -y -n $shamba_env_name python=3.7
eval "$(conda shell.bash hook)" #https://github.com/conda/conda/issues/7980
conda activate $shamba_env_name
}

packages-install(){
echo -e "${NC}We are now installing the Shamba dependencies in '$shamba_env_name' environment. This can take several minutes."

conda install -y -c conda-forge basemap;
conda install -y -c conda-forge scipy;
conda install -y -c conda-forge pandas;
conda install -y -c conda-forge pyqt;
conda install -y -c conda-forge numpy;

}
env-config || {

    echo -e "${RED}It looks like you are not running this script in a bash with Anaconda not initialized. Please do so and try again." 
    echo "If it keeps failing, please contact the community here https://framavox.org/shamba/" 
    echo -e "${NC}"
    false
} && packages-install
