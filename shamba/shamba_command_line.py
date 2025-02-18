#!/usr/bin/python
"""
### TERMS AND CONDITIONS ###
This software is provided under the University of Edinburgh's Open Technology By
downloading this software you accept the University of Edinburgh's Open Technology
terms and conditions.

These can be viewed here: http://www.research-innovation.ed.ac.uk/Opportunities/small-
holder-agriculture-mitigation-benefit-assessment-tool

### INSTRUCTIONS ###
Below are NINE steps to help you run this script.

This script provides a method for estimating GHG impacts for single or multiple
interventions. It has been developed as part of the Plan Vivo approved approach
See here for further details:
https://shambatool.files.wordpress.com/2013/10/shamba-methodology-v9-plan-vivo-approved-approach.pdf

This script runs SHAMBA v1.1 in conjunction with the '_input.csv' sheet
generated from the Excel spreadsheet 'SHAMBA_input_output_template_v1.1'. For
the full instructions, first see this Excel spreadsheet in the
'plan_vivo_approach_excel_templates' folder first .

The script is currently set up as an example to run with the example
'WL_input.csv' generated from the Excel spreadsheet 'example_SHAMBA_input_output_uganda_tech_spec'.
found in the 'plan_vivo_approach_excel_templates' folder.

In the script below, comments with a double hash (##) are
INSTRUCTIONS and require you to do something to the subsequent code.

Comments marked with a single hash (#) are just notation to describe
the subsequent code and shouldn't usually require any action (unless
you want to develop the script to run differently)

If you have feedback on the model code, please feel free to change the code
and send the script with a short explanation to shamba.model@gmail.com
"""

import os
import sys

import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np

from model.common import csv_handler, io_handler

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_dir))

import model.command_line.climate as Climate
from model.command_line.soil_params import SoilParams

from model.command_line.crop_params import CropParams
from model.command_line.crop_model import CropModel

from model.command_line.tree_params import TreeParams
from model.command_line.tree_growth import TreeGrowth
from model.command_line.tree_model import TreeModel

from model.command_line.litter import LitterModel
from model.command_line.soil_model import InverseRothC, ForwardRothC
from model import configuration
from model.command_line import emit


def setup_project_directory(project_name):
    """
    Set up a new project directory with the required input files.

    Args:
    project_name (str): The name of the new project directory.

    Returns:
    str: Path to the newly created project directory.
    """

    # New project directory
    project_dir = os.path.join(configuration.PROJECT_DIR, project_name)

    # Input directory within the project directory
    input_dir = os.path.join(project_dir, "input")

    # Create the project and input directories
    os.makedirs(input_dir, exist_ok=True)

    # List of files to copy
    files_to_copy = [
        "crop_ipcc_baseline.csv",
        "crop_ipcc.csv",
        "climate.csv",
        "soil-info.csv",
        "WL_input.csv",
    ]

    # Source directory (using an existing project as an example)
    source_dir = os.path.join(
        configuration.PROJECT_DIR, "examples", "UG_TS_2016", "input"
    )

    # Copy each file
    for file in files_to_copy:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(input_dir, file)
        shutil.copy2(source_file, dest_file)
        print(f"Copied {file} to {dest_file}")

    print(f"Project setup complete. New project directory: {project_dir}")
    return project_dir


def main(n, arguments):
    project_name = arguments["project-name"]

    # Create a new project directory
    setup_project_directory(project_name)

    # Setup the project directory constants
    configuration.SAVE_DIR = os.path.join(configuration.PROJECT_DIR, project_name)

    # specifiying input and output files
    configuration.INPUT_DIR = os.path.join(configuration.SAVE_DIR, "input")
    configuration.OUTPUT_DIR = os.path.join(configuration.SAVE_DIR, "output")

    input_csv = arguments["input-file-name"]

    # ----------
    # getting input data
    # ----------

    ## creating dictionary of input data from input.csv
    data = list(csv.reader(open(configuration.INPUT_DIR + "/" + input_csv)))
    keys = data[0]
    values = data[n + 1]
    val = dict(list(zip(keys, values)))

    # terms in coded below preceded by val are values being pulled in from dictionary
    # created above. Converting to float or interger as needed for each
    # key

    ## getting plot anlaysis number to name output

    st = int(val["analysis_no"])

    # ----------
    # location information
    # ----------
    loc = (float(val["lat"]), float(val["lon"]))
    climate = Climate.from_location(loc)

    # ----------
    # project length
    # ----------
    # YEARS = length of tree data. ACCT = years in accounting period
    configuration.N_YEARS = int(val["yrs_proj"])
    configuration.N_ACCT = int(val["yrs_acct"])

    # ----------
    # soil equilibrium solve
    # ----------
    soil = SoilParams.from_location(loc)
    invRoth = InverseRothC(soil, climate)

    # ----------
    # tree model
    # ----------

    """
    If nitrogen allocations, carbon, root/shoot and/or wood density attributes
    differ between tree cohorts, add a new row specifying these tree parametres
    to the the tree_defaults.csv at shamba/default_input folder and make sure the 
    '_input.csv' file correctly attributes each tree cohort to the relevant 
    parametres under 'trees in baseline' and 'trees in project'
    """

    # linking tree cohort parameteres
    tree_par_base = TreeParams.from_species_index(int(val["species_base"]))
    tree_par1 = TreeParams.from_species_index(int(val["species1"]))
    tree_par2 = TreeParams.from_species_index(int(val["species2"]))
    tree_par3 = TreeParams.from_species_index(int(val["species3"]))

    # linking tree growth
    spp = int(val["species_base"])
    if spp == 1:
        growth_base = TreeGrowth.from_csv1(tree_par_base, n, filename=input_csv)
    elif spp == 2:
        growth_base = TreeGrowth.from_csv2(tree_par_base, n, filename=input_csv)
    else:
        growth_base = TreeGrowth.from_csv3(tree_par_base, n, filename=input_csv)

    spp = int(val["species1"])
    if spp == 1:
        growth1 = TreeGrowth.from_csv1(tree_par1, n, filename=input_csv)
    elif spp == 2:
        growth1 = TreeGrowth.from_csv2(tree_par1, n, filename=input_csv)
    else:
        growth1 = TreeGrowth.from_csv3(tree_par1, n, filename=input_csv)

    spp = int(val["species2"])
    if spp == 1:
        growth2 = TreeGrowth.from_csv1(tree_par2, n, filename=input_csv)
    elif spp == 2:
        growth2 = TreeGrowth.from_csv2(tree_par2, n, filename=input_csv)
    else:
        growth2 = TreeGrowth.from_csv3(tree_par2, n, filename=input_csv)

    spp = int(val["species3"])
    if spp == 1:
        growth3 = TreeGrowth.from_csv1(tree_par3, n, filename=input_csv)
    elif spp == 2:
        growth3 = TreeGrowth.from_csv2(tree_par3, n, filename=input_csv)
    else:
        growth3 = TreeGrowth.from_csv3(tree_par3, n, filename=input_csv)

    # specify thinning regime and fraction left in field (lif)
    # baseline thinning regime
    # (add line of thinning[yr] = % thinned for each event)
    thinning_base = np.zeros(configuration.N_YEARS + 1)
    thinning_base[int(val["thin_base_yr1"])] = float(val["thin_base_pc1"])
    thinning_base[int(val["thin_base_yr2"])] = float(val["thin_base_pc2"])

    # project thinning regime
    # (add need line of thinning[yr] = % thinned for each event)
    thinning_proj = np.zeros(configuration.N_YEARS + 1)
    thinning_proj[int(val["thin_proj_yr1"])] = float(val["thin_proj_pc1"])
    thinning_proj[int(val["thin_proj_yr2"])] = float(val["thin_proj_pc2"])
    thinning_proj[int(val["thin_proj_yr3"])] = float(val["thin_proj_pc3"])
    thinning_proj[int(val["thin_proj_yr4"])] = float(val["thin_proj_pc4"])

    # baseline fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_proj.
    thin_frac_lif_base = np.array(
        [1, float(val["thin_base_br"]), float(val["thin_base_st"]), 1, 1]
    )

    # project fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_proj.
    thin_frac_lif_proj = np.array(
        [1, float(val["thin_proj_br"]), float(val["thin_proj_st"]), 1, 1]
    )

    # specify mortality regime and fraction left in field (lif)

    # baseline yearly mortality
    mort_base = np.array((configuration.N_YEARS + 1) * [float(val["base_mort"])])

    # project yearly mortality
    mort_proj = np.array((configuration.N_YEARS + 1) * [float(val["proj_mort"])])

    # baseline fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_proj.
    mort_frac_lif_base = np.array(
        [1, float(val["mort_base_br"]), float(val["mort_base_st"]), 1, 1]
    )

    # project fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_proj.
    mort_frac_lif_proj = np.array(
        [1, float(val["mort_proj_br"]), float(val["mort_proj_st"]), 1, 1]
    )

    # run tree model

    # trees planted in baseline (standDens must be at least 1)
    tree_base = TreeModel.from_defaults(
        tree_params=tree_par1,
        tree_growth=growth_base,
        yearPlanted=0,
        standDens=int(val["base_plant_dens"]),
        thin=thinning_base,
        thinFrac=thin_frac_lif_base,
        mort=mort_base,
        mortFrac=mort_frac_lif_base,
    )

    # trees planted in project
    tree_proj1 = TreeModel.from_defaults(
        tree_params=tree_par1,
        tree_growth=growth1,
        yearPlanted=int(val["proj_plant_yr1"]),
        standDens=int(val["proj_plant_dens1"]),
        thin=thinning_proj,
        thinFrac=thin_frac_lif_proj,
        mort=mort_proj,
        mortFrac=mort_frac_lif_proj,
    )

    tree_proj2 = TreeModel.from_defaults(
        tree_params=tree_par2,
        tree_growth=growth2,
        yearPlanted=int(val["proj_plant_yr2"]),
        standDens=int(val["proj_plant_dens2"]),
        thin=thinning_proj,
        thinFrac=thin_frac_lif_proj,
        mort=mort_proj,
        mortFrac=mort_frac_lif_proj,
    )

    tree_proj3 = TreeModel.from_defaults(
        tree_params=tree_par3,
        tree_growth=growth3,
        yearPlanted=int(val["proj_plant_yr3"]),
        standDens=int(val["proj_plant_dens3"]),
        thin=thinning_proj,
        thinFrac=thin_frac_lif_proj,
        mort=mort_proj,
        mortFrac=mort_frac_lif_proj,
    )

    # ----------
    # fire model
    # ----------
    # return interval of fire, [::2] = 1 is return interval of two years
    base_fire_interval = int(val["fire_int_base"])
    if base_fire_interval == 0:
        fire_base = np.zeros(configuration.N_YEARS)
    else:
        fire_base = np.zeros(configuration.N_YEARS)
        fire_base[::base_fire_interval] = int(val["fire_pres_base"])

    proj_fire_interval = int(val["fire_int_proj"])
    if proj_fire_interval == 0:
        fire_proj = np.zeros(configuration.N_YEARS)
    else:
        fire_proj = np.zeros(configuration.N_YEARS)
        fire_proj[::proj_fire_interval] = int(val["fire_pres_proj"])

    # ----------
    # litter model
    # ----------

    # baseline external organic inputs
    l_base = LitterModel.from_defaults(
        litterFreq=int(val["base_lit_int"]), litterQty=float(val["base_lit_qty"])
    )

    # baseline synthetic fertiliser additions
    sf_base = LitterModel.synthetic_fert(
        freq=int(val["base_sf_int"]),
        qty=float(val["base_sf_qty"]),
        nitrogen=float(val["base_sf_n"]),
    )

    # Project external organic inputs
    l_proj = LitterModel.from_defaults(
        litterFreq=int(val["proj_lit_int"]), litterQty=float(val["proj_lit_qty"])
    )

    # Project synthetic fertiliser additions
    sf_proj = LitterModel.synthetic_fert(
        freq=int(val["proj_sf_int"]),
        qty=float(val["proj_sf_qty"]),
        nitrogen=float(val["proj_sf_n"]),
    )

    # ----------
    # Crop model
    # ----------
    # Baseline specify crop, yield, and % left in field in csv file
    cropPar = csv_handler.read_csv(input_csv)
    cropPar = np.atleast_2d(cropPar)
    crop_base = []  # list for crop objects
    crop_par_base = []

    spp = int(val["crop_base_spp1"])
    harvYield = np.zeros(configuration.N_YEARS)
    harvYield[int(val["crop_base_start1"]) : int(val["crop_base_end1"])] = float(
        val["crop_base_yd1"]
    )
    harvFrac = float(val["crop_base_left1"])

    ci = CropParams.from_species_index(spp)
    c = CropModel(ci, harvYield, harvFrac)
    crop_base.append(c)
    crop_par_base.append(ci)

    spp = int(val["crop_base_spp2"])
    harvYield = np.zeros(configuration.N_YEARS)
    harvYield[int(val["crop_base_start2"]) : int(val["crop_base_end2"])] = float(
        val["crop_base_yd2"]
    )
    harvFrac = float(val["crop_base_left2"])

    ci = CropParams.from_species_index(spp)
    c = CropModel(ci, harvYield, harvFrac)
    crop_base.append(c)
    crop_par_base.append(ci)

    spp = int(val["crop_base_spp3"])
    harvYield = np.zeros(configuration.N_YEARS)
    harvYield[int(val["crop_base_start3"]) : int(val["crop_base_end3"])] = float(
        val["crop_base_yd3"]
    )
    harvFrac = float(val["crop_base_left3"])

    ci = CropParams.from_species_index(spp)
    c = CropModel(ci, harvYield, harvFrac)
    crop_base.append(c)
    crop_par_base.append(ci)

    # Project specify crop, yield, and % left in field in csv file
    cropPar = csv_handler.read_csv(input_csv)
    cropPar = np.atleast_2d(cropPar)
    crop_proj = []
    crop_par_proj = []

    spp = int(val["crop_proj_spp1"])
    harvYield = np.zeros(configuration.N_YEARS)
    harvYield[int(val["crop_proj_start1"]) : int(val["crop_proj_end1"])] = float(
        val["crop_proj_yd1"]
    )
    harvFrac = float(val["crop_proj_left1"])

    ci = CropParams.from_species_index(spp)
    c = CropModel(ci, harvYield, harvFrac)
    crop_proj.append(c)
    crop_par_proj.append(ci)

    spp = int(val["crop_proj_spp2"])
    harvYield = np.zeros(configuration.N_YEARS)
    harvYield[int(val["crop_proj_start2"]) : int(val["crop_proj_end2"])] = float(
        val["crop_proj_yd2"]
    )
    harvFrac = float(val["crop_proj_left2"])

    ci = CropParams.from_species_index(spp)
    c = CropModel(ci, harvYield, harvFrac)
    crop_proj.append(c)
    crop_par_proj.append(ci)

    spp = int(val["crop_proj_spp3"])
    harvYield = np.zeros(configuration.N_YEARS)
    harvYield[int(val["crop_proj_start3"]) : int(val["crop_proj_end3"])] = float(
        val["crop_proj_yd3"]
    )
    harvFrac = float(val["crop_proj_left3"])

    ci = CropParams.from_species_index(spp)
    c = CropModel(ci, harvYield, harvFrac)
    crop_proj.append(c)
    crop_par_proj.append(ci)

    # soil cover for baseline
    cover_base = np.zeros(12)
    cover_base[int(val["base_cvr_mth_st"]) : int(val["base_cvr_mth_en"])] = int(
        val["base_cvr_pres"]
    )

    # soil cover for project
    cover_proj = np.zeros(12)
    cover_proj[int(val["proj_cvr_mth_st"]) : int(val["proj_cvr_mth_en"])] = int(
        val["proj_cvr_pres"]
    )

    # Solve to y=0
    forRoth = ForwardRothC(
        soil,
        climate,
        cover_base,
        Ci=invRoth.eqC,
        crop=crop_base,
        fire=fire_base,
        solveToValue=True,
    )

    # Soil carbon for baseline and project
    roth_base = ForwardRothC(
        soil,
        climate,
        cover_base,
        Ci=forRoth.SOC[-1],
        crop=crop_base,
        tree=[tree_base],
        litter=[l_base],
        fire=fire_base,
    )

    roth_proj = ForwardRothC(
        soil,
        climate,
        cover_proj,
        Ci=forRoth.SOC[-1],
        crop=crop_proj,
        tree=[tree_proj1, tree_proj2, tree_proj3],
        litter=[l_proj],
        fire=fire_proj,
    )

    # Emissions stuff
    emit_base = emit.Emission(
        forRothC=roth_base,
        crop=crop_base,
        tree=[tree_base],
        litter=[l_base],
        fert=[sf_base],
        fire=fire_base,
    )
    emit_proj = emit.Emission(
        forRothC=roth_proj,
        crop=crop_proj,
        tree=[tree_proj1, tree_proj2, tree_proj3],
        litter=[l_proj],
        fert=[sf_proj],
        fire=fire_proj,
    )

    # ----------
    # Printing outputs
    # ----------

    # print stuff
    print("location: ", loc)
    Climate.print_to_stdout(climate)
    soil.print_()
    growth1.print_()
    growth2.print_()
    growth3.print_()
    tree_proj1.print_biomass()
    tree_proj1.print_balance()
    tree_proj2.print_biomass()
    tree_proj2.print_balance()
    tree_proj3.print_biomass()
    tree_proj3.print_balance()
    forRoth.print_()
    roth_base.print_()
    roth_proj.print_()

    # crop emissions print
    print("\n\nCROP EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    crop_base_emit = emit.Emission(crop=crop_base, fire=fire_base)
    crop_proj_emit = emit.Emission(crop=crop_proj, fire=fire_proj)

    crop_diff = crop_proj_emit.emissions - crop_base_emit.emissions
    for i in range(len(crop_base_emit.emissions)):
        print(crop_base_emit.emissions[i], crop_proj_emit.emissions[i], crop_diff[i])

    print("\nTotal crop difference: ", sum(crop_diff), " t CO2 ha^-1")

    print("Average crop difference: ", np.mean(crop_diff))

    # fert emissions print
    print("\n\nFERTILISER EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    fert_base_emit = emit.Emission(fert=[sf_base])
    fert_proj_emit = emit.Emission(fert=[sf_proj])

    fert_diff = fert_proj_emit.emissions - fert_base_emit.emissions
    for i in range(len(fert_base_emit.emissions)):
        print(fert_base_emit.emissions[i], fert_proj_emit.emissions[i], fert_diff[i])

    print("\nTotal fertiliser difference: ", sum(fert_diff), " t CO2 ha^-1")

    print("Average fertiliser difference: ", np.mean(fert_diff))

    # litter emissions print
    print("\n\nLITTER EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    lit_base_emit = emit.Emission(litter=[l_base], fire=fire_base)
    lit_proj_emit = emit.Emission(litter=[l_proj], fire=fire_proj)

    lit_diff = lit_proj_emit.emissions - lit_base_emit.emissions
    for i in range(len(lit_base_emit.emissions)):
        print(lit_base_emit.emissions[i], lit_proj_emit.emissions[i], lit_diff[i])

    print("\nTotal Litter difference: ", sum(lit_diff), " t CO2 ha^-1")

    print("Average Litter difference: ", np.mean(lit_diff))

    # fire emissions print
    print("\n\nFIRE EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    fire_base_emit = emit.Emission(fire=fire_base)
    fire_proj_emit = emit.Emission(fire=fire_proj)

    fire_diff = fire_proj_emit.emissions - fire_base_emit.emissions
    for i in range(len(fire_base_emit.emissions)):
        print(fire_base_emit.emissions[i], fire_proj_emit.emissions[i], fire_diff[i])

    print("\nTotal Fire difference: ", sum(fire_diff), " t CO2 ha^-1")

    print("Average Fire difference: ", np.mean(fire_diff))

    # tree emissions print
    print("\n\nTREE EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    tree_base_emit = emit.Emission(tree=[tree_base], fire=fire_base)
    tree_proj_emit = emit.Emission(
        tree=[tree_proj1, tree_proj2, tree_proj3], fire=fire_proj
    )

    tree_diff = tree_proj_emit.emissions - tree_base_emit.emissions
    for i in range(len(tree_base_emit.emissions)):
        print(tree_base_emit.emissions[i], tree_proj_emit.emissions[i], tree_diff[i])

    print("\nTotal tree difference: ", sum(tree_diff), " t CO2 ha^-1")

    print("Average tree difference: ", np.mean(tree_diff))

    # soil diff emissions print
    print("\n\nSOIL EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    soil_base_emit = emit_base.emissions - (
        crop_base_emit.emissions
        + fert_base_emit.emissions
        + lit_base_emit.emissions
        + fire_base_emit.emissions
        + tree_base_emit.emissions
    )
    soil_proj_emit = emit_proj.emissions - (
        crop_proj_emit.emissions
        + fert_proj_emit.emissions
        + lit_proj_emit.emissions
        + fire_proj_emit.emissions
        + tree_proj_emit.emissions
    )

    soil_diff = soil_proj_emit - soil_base_emit
    for i in range(len(emit_base.emissions)):
        print(soil_base_emit[i], soil_proj_emit[i], soil_diff[i])

    print("\nTotal Soil difference: ", sum(soil_diff), " t CO2 ha^-1")

    print("Average Soil difference: ", np.mean(soil_diff))

    # total emissions print
    print("\n\nTOTAL EMISSIONS (t CO2)")
    print("=================\n")
    print("baseline    project")

    emit_diff = emit_proj.emissions - emit_base.emissions
    for i in range(len(emit_base.emissions)):
        print(emit_base.emissions[i], emit_proj.emissions[i], emit_diff[i])

    print("\nTotal difference: ", sum(emit_diff), " t CO2 ha^-1")

    print("Average difference: ", np.mean(emit_diff))

    # summary of GHG pools
    print("\n\nSUMMARY OF EMISSIONS (t CO2)")
    print("over " + str(val["yrs_acct"]) + " years")
    print("=================\n")
    print("baseline    project")

    print("\nTotal crop difference: ", sum(crop_diff), " t CO2 ha^-1")
    print("\nTotal fertiliser difference: ", sum(fert_diff), " t CO2 ha^-1")
    print("\nTotal litter difference: ", sum(lit_diff), " t CO2 ha^-1")
    print("\nTotal fire difference: ", sum(fire_diff), " t CO2 ha^-1")
    print("\nTotal tree difference: ", sum(tree_diff), " t CO2 ha^-1")
    print("\nTotal Soil difference: ", sum(soil_diff), " t CO2 ha^-1")

    print("\nTotal difference: ", sum(emit_diff), " t CO2 ha^-1")

    # Save stuff

    # starting plot output number
    st = 1

    dir = configuration.OUTPUT_DIR + "_" + mod_run + "\plot_" + str(n + st)

    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    plot_name = dir + "\plot_" + str(n + st)

    Climate.save(climate, plot_name + "_climate.csv")

    soil.save_(plot_name + "_soil.csv")
    growth1.save_(plot_name + "_growth1.csv")
    growth2.save_(plot_name + "_growth2.csv")
    growth3.save_(plot_name + "_growth3.csv")
    tree_proj1.save_(plot_name + "_tree_proj1.csv")
    tree_proj2.save_(plot_name + "_tree_proj2.csv")
    tree_proj3.save_(plot_name + "_tree_proj3.csv")

    i = 1
    for i in range(len(crop_base)):
        crop_base[i].save_(plot_name + "_crop_model_base_" + str(i) + ".csv")

        crop_par_base[i].save_(plot_name + "_crop_params_base_" + str(i) + ".csv")

        crop_proj[i].save_(plot_name + "_crop_model_proj_" + str(i) + ".csv")

        crop_par_proj[i].save_(plot_name + "_crop_params_proj_" + str(i) + ".csv")

    invRoth.save_(plot_name + "_invRoth.csv")
    forRoth.save_(plot_name + "_forRoth.csv")

    roth_base.save_(plot_name + "_soil_model_base.csv")
    roth_proj.save_(plot_name + "_soil_model_proj.csv")
    emit_proj.save_(emit_base, emit_proj, plot_name + "_emit_proj.csv")

    with open(
        configuration.OUTPUT_DIR
        + "_"
        + mod_run
        + "\plot_"
        + str(n + st)
        + "\plot_"
        + str(n + st)
        + "_emissions_all_pools_per_year.csv",
        "w+",
    ) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [
                "emit_base",
                "emit_proj",
                "emit_diff",
                "soil_base",
                "soil_proj",
                "soil_diff",
                "tree_base",
                "tree_proj",
                "tree_diff",
                "fire_base",
                "fire_proj",
                "fire_diff",
                "lit_base",
                "lit_proj",
                "lit_diff",
                "fert_base",
                "fert_proj",
                "fert_diff",
                "crop_base",
                "crop_proj",
                "crop_diff",
            ]
        )
        for i in range(len(emit_base.emissions)):
            writer.writerow(
                [
                    emit_base.emissions[i],
                    emit_proj.emissions[i],
                    emit_diff[i],
                    soil_base_emit[i],
                    soil_proj_emit[i],
                    soil_diff[i],
                    tree_base_emit.emissions[i],
                    tree_proj_emit.emissions[i],
                    tree_diff[i],
                    fire_base_emit.emissions[i],
                    fire_proj_emit.emissions[i],
                    fire_diff[i],
                    lit_base_emit.emissions[i],
                    lit_proj_emit.emissions[i],
                    lit_diff[i],
                    fert_base_emit.emissions[i],
                    fert_proj_emit.emissions[i],
                    fert_diff[i],
                    crop_base_emit.emissions[i],
                    crop_proj_emit.emissions[i],
                    crop_diff[i],
                ]
            )

    # Plot stuff
    growth1.plot_(saveName=plot_name + "_growthFits.png")
    plt.close()

    tree_proj1.plot_biomass(saveName=plot_name + "_biomassPools.png")
    plt.close()

    tree_proj1.plot_balance(saveName=plot_name + "_massBalance.png")
    plt.close()

    tree_proj2.plot_biomass(saveName=plot_name + "_biomassPools.png")
    plt.close()

    tree_proj2.plot_balance(saveName=plot_name + "_massBalance.png")
    plt.close()

    tree_proj3.plot_biomass(saveName=plot_name + "_biomassPools.png")
    plt.close()

    tree_proj3.plot_balance(saveName=plot_name + "_massBalance.png")
    plt.close()

    forRoth.plot_(legendStr="initialisation")

    roth_base.plot_(legendStr="baseline")

    roth_proj.plot_(legendStr="project", saveName=plot_name + "_soilModel.png")
    plt.close()

    emit_base.plot_(legendStr="baseline")

    emit_proj.plot_(legendStr="project")

    emit.Emission.ax.plot(emit_diff, label="difference")

    emit.Emission.ax.legend(loc="best")

    plt.savefig(os.path.join(configuration.OUTPUT_DIR, plot_name + "_emissions.png"))
    plt.close()

    emit_proj.save_(emit_base, emit_proj=emit_proj, file=plot_name + "_emissions.csv")

    return (
        sum(crop_diff),
        sum(fert_diff),
        sum(lit_diff),
        sum(fire_diff),
        sum(tree_diff),
        sum(soil_diff),
        sum(emit_diff),
    )


if __name__ == "__main__":
    """
    ## STEP ## (OPTIONAL)
    If you only want to run a selection of rows (i.e. scenarios) from the '_input.csv'
    file, specify the number_of_rows value here. If you want to run all rows place a # in front
    of the line of code below to deactivate them.
    """

    number_of_rows = 1
    # Get command line arguments
    arguments = io_handler.get_arguments_interactively()

    """
    ## STEP 8 ##
    Specify in the code below the title that will be attached to all of your
    output folders and files
    """
    mod_run = arguments["output-title"]

    emit_output_data = []
    for n in range(number_of_rows):
        emit_output_data.append(main(n, arguments))

    """
    ## STEP 9 ##
    After you have completed all of the above steps, run the code.
    
    Results for each line of the _input.csv file will (i.e. each model run) 
    will appear in a subfolder in the shamba/projects/"project"/output_/. 
    
    All results are in tCO2e.
    
    The .csv output files provided detailed information about the parameteres 
    (params), baseline emissions (base) and intervention emissions (proj) for
    each GHG sink or source. The .csv summarise '_emissions_all_pools_per_year.csv' this information.
    You can analyse this information further using Excel.
    """
