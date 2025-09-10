
Write-Host "`n"
Write-Host "  ____   _   _     _     __  __  ____     _    "  -ForegroundColor Green
Write-Host " / ___| | | | |   / \   |  \/  || __ )   / \   "  -ForegroundColor Green
Write-Host " \___ \ | |_| |  / _ \  | |\/| ||  _ \  / _ \  "  -ForegroundColor Green
Write-Host "  ___) ||  _  | / ___ \ | |  | || |_) |/ ___ \ "  -ForegroundColor Green
Write-Host " |____/ |_| |_|/_/   \_\|_|  |_||____//_/   \_\"  -ForegroundColor Green
Write-Host "`n"
Write-Host "Welcome to the Shamba setup!`n"

$miniconda_exists = Read-Host 'Do you already have Miniconda for Python 3.7 installed on your computer? [y/n]'
if ($miniconda_exists -ne 'y') {
    Write-Host "`nPlease first install Miniconda for Python 3.7 from the Anaconda official website: https://docs.conda.io/en/latest/miniconda.html"  -ForegroundColor Green
    Write-Host "Once done, please start this script in a 'Miniconda Powershell Prompt'.`n"  -ForegroundColor Green
    exit
}else {
    Write-Host "Please ensure you are running this script in a 'Miniconda Powershell Prompt' before continuing.`n"  -ForegroundColor Green
}

$shamba_env_name = Read-Host 'Please enter a name for the shamba virtual environment we are going to create for you'

Try
{
conda config --set channel_priority strict #from https://github.com/conda/conda/issues/7626
conda create -y -n $shamba_env_name python=3.7
conda activate $shamba_env_name

Write-Host "`We are now installing the Shamba dependencies in '$shamba_env_name' environment. This can take several minutes."

conda install -y -c conda-forge basemap;
conda install -y -c conda-forge scipy;
conda install -y -c conda-forge pandas;
conda install -y -c conda-forge pyqt;
conda install -y -c conda-forge numpy;
}
Catch [System.Management.Automation.CommandNotFoundException]
{
    Write-Host "It looks like you are not running this script in a Miniconda Powershell Prompt. Please do so and try again." -ForegroundColor Red
    Write-Host "If it keeps failing, please contact the community here https://framavox.org/shamba/" -ForegroundColor Red
}