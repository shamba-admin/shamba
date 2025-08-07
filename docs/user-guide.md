# SHAMBA guides

## Introduction

Shamba can be used as:

* a command line application
* a GUI application

>Disclaimer : The two applications do not cover the same capabilities. The command line version offers more features.

## Setup your environment

### Pre requisite

1. unzip the soil data file located here `./shamba/rasters/soil/hwsd.7z`.
2. install Miniconda on your computer from this webpage [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

**Ensure to download the Python 3.10.16 version.**

### Windows setup

1. Open Anaconda Powershell prompt from the Windows Start menu.
2. From the prompt, navigate to the shamba source code folder.
3. Execute the `./setup.ps1` script.
4. Follow the script instructions.

### Linux based system setup

1. Open a terminal.
2. From the prompt, navigate to the shamba source code folder.
3. Execute the `./setup.sh` script.
4. Follow the script instructions.
5. Because of an [issue in Miniconda](https://github.com/conda/conda/issues/7980) you will have to activate created environment using the folowing command: `conda activate your-environment-name`

### Anaconda tips

#### Activate an environment from the Anaconda prompt

Type the following command with `myenv` being your environment name.

`conda activate myenv`

If you have more needs [please refer to Anaconda web site documentation here.](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Use Shamba

### Shamba Command line application

To start the command line application, run the `python ./shamba/shamba_command_line_.py` Python script from Anaconda prompt with the appropriate environment set.

### Shamba Graphical User Interface application

To start the application, run the `python shamba.pyw` Python script from Anaconda prompt with the appropriate environment set.

When SHAMBA is launched, it will open a window containing a disclaimer about the SHAMBA tool. You must first accept this disclaimer before being able to use the tool. After accepting the terms, you will arrive at the main interface for shamba. There are four tabs: when starting any new SHAMBA project, the information in the “General” tab must first be filled out. Baselines and interventions can then be specified in the “Baselines” and “Interventions” tabs respectively. The “Mitigation Estimates” tab then allows you to plot the yearly greenhouse gas emissions for a baseline-intervention pair and calculates the mitigation estimates for that intervention.

### Debuging, for developers

>The following applies to Windows using Miniconda

To debug and implement Shamba we actually use the open source editor Microsoft Visual Studio Code. If you want to use it also and having the Miniconda environments support, we recommand you to install the following extensions in vscode:

* Python
* Anaconda Extension Pack

Then to enable a smooth debugging experience:

1- start the Miniconda Powershell command line

2- navigate to the shamba source code folder

3- start vscode from the command line: ```%VSCODE_PATH%/code .```

4- choose the appropriate Python interpreter in vscode (bottom left in the status bar)
You should choose something like ```Python 3.10.16 64-bit ('myenv':conda)```

5- open the ```shamba/shamba_command_line_.py``` file in the editor

6- then hit F5 key of your keyboard or go in "Run -> Start Debugging" menu

For any issue please go to vscode user manual website : <https://code.visualstudio.com/docs/python/python-tutorial>

### General Project Information

General project information (such as location and climate) must first be specified before the rest of the SHAMBA tool can be used. In the “General” tab, click the “Enter general project information” button. A new window labelled “General Information” will appear, which contains several screens with information to enter regarding your project.

The screens can be navigated using the “Next” and “Back” buttons at the bottom of the window, and clicking the “Help” button will bring up a pop­up window with more information about the current screen. It is important to note that the screens must all be completed in order for the information to be saved. That is, you will lose all information entered if you close the window via the close button (the x in the top corner) instead of the “Save” button on the last screen. You are also advised to review all entered information before clicking “Save” since, once saved, the information cannot be edited (only overwritten).

Once the information has been saved, the program will return to the main interface. In the “General” tab you will see a text summary of the information that was entered in the other window. If the information was not entered correctly, it can be overwritten by simply pressing the “Enter general project information button” again and going through the process again. Otherwise, you are now free to move on to the “Baselines” and “Interventions” tabs.

Since this general information is used to do calculations on the baseline scenarios and interventions, it cannot be changed after creating any baselines or interventions. It is thus crucial to ensure that this information is accurate before moving on to the steps outlined in Section 2.3.

### Baselines and Interventions

In the “Baselines” and “Interventions” tabs of the main interface, you can add information regarding the baseline scenarios and interventions in your project. To do this, click on the “Add new baseline” or “Add new intervention” button.

After entering the name of the baseline/intervention you wish to create, you will then be taken to a new window similar to the “General information” window. After providing all the required information and saving it, you will then see a text summary of the baseline/intervention in the appropriate tab of the main interface.

Multiple baselines and interventions can be created in the same manner, and the drop­down lists at the top­left of the “Baselines” and “Interventions” tabs allow you to see details about each baseline/intervention that has been created.

### Mitigation Estimates

Once at least one baseline and one intervention have been created, you can then use the “Mitigation Estimates” tab to plot the greenhouse gas emissions/removals associated with each, and see the net impact (mitigation estimates) associated with the intervention. Simply select which baseline and intervention you would like to plot from the drop-down lists, then click the “Plot mitigation estimates” button. This can be repeated for any baseline-intervention combination, and you can also continue to define new baselines and interventions; any new additions will be automatically displayed in the drop­down lists.

### Miscellaneous

#### Saving the Project

You can save your SHAMBA project (all raw data and calculations) to .csv files by pressing “Save” in the file menu at the top-left of the screen (or with ctrl+s on the keyboard). If this is the first save for the project, you will be prompted to enter a name/location for the save folder. If you have previously saved this project, the tool will automatically add any new information to the folder you selected already - you can specify a new save folder by using the “Save As” option.

#### Loading an Existing Project

Unfortunately, the SHAMBA tool cannot load an existing project (yet). Saved project data can be imported into Excel, Python, R, and many other tools, however, since it is all in .csv format.
