"""Run the baseline/intervention model from the gui."""

import os
import sys

from model.common import csv_handler

try:
    from io import StringIO  # # for Python 2
except ImportError:
    from io import StringIO  ## for Python 3

import logging as log
import shutil

import numpy as np
from PyQt5 import QtCore, QtGui

import model.litter as LitterModel
import model.soil_params as SoilParams
import model.tree_params as TreeParams
from model import configuration, emit
from model.crop_model import CropModel
from model.crop_params import CropParams
from model.tree_growth import TreeGrowth
from model.tree_model import TreeModel


class ProjectModel(object):
    """Run the part of the model corresponding to
    a baseline or intervention - with info from the project dialog."""

    def __init__(self, ui, gen_params, name, proj_index, isBaseline=False):
        """
        Args:
            ui: project_dialog instance
            params: model.General instance
            name: name of project given by user
            proj_index: index of project in the stackedWidget
                        (+1 for 1-indexing)

        """
        self.ui = ui
        self.name = str(name)
        self.proj_index = proj_index
        self.total_emissions = None

        self.isBaseline = isBaseline

        self.soil = Soil(self.ui, gen_params)

        self.crops = []
        for i in range(self.ui.cropNum.value()):
            self.crops.append(Crop(self.ui, i))

        self.trees = []
        for i in range(self.ui.treeNum.value()):
            self.trees.append(Tree(self.ui, i))

        self.litter = []
        for i in range(self.ui.litterNum.value()):
            self.litter.append(Litter(self.ui, i))

        self.fert = []
        for i in range(self.ui.fertNum.value()):
            self.fert.append(Fertiliser(self.ui, i))

        self.save_description()

    def save_data(self):
        """Save info for this baseline/intervention.
        Called in the shamba.pyw file as a slot."""
        # make directory
        if self.isBaseline:
            proj_dir = os.path.join(configuration.OUTPUT_DIR, "baselines", self.name)
        else:
            proj_dir = os.path.join(
                configuration.OUTPUT_DIR, "interventions", self.name
            )
        if not os.path.exists(proj_dir):
            os.makedirs(proj_dir)

        # general description (what's printed out in the box of the gui)
        if self.isBaseline:
            filename = os.path.join(
                configuration.OUTPUT_DIR, "baselines", self.name, "baseline_info.txt"
            )
        else:
            filename = os.path.join(
                configuration.OUTPUT_DIR,
                "interventions",
                self.name,
                "intervention_info.txt",
            )
        with open(filename, "w+") as fout:
            fout.write(self.description)

        # save soil info
        self.soil.soil_params.save_(os.path.join(proj_dir, "soil_params.csv"))
        try:  # see if soil input file exists
            dest_file = os.path.basename(self.soil.soilFilename).split(".csv")[0]
            dest_file += "_" + self.name + ".csv"
            dest_file = os.path.join(configuration.INPUT_DIR, dest_file)
            shutil.copyfile(self.soil.soilFilename, dest_file)
        except AttributeError:  # no soil input file
            pass

        # crop info
        for i, crop in enumerate(self.crops):
            if len(self.crops) == 1:
                params_basename = "crop_params.csv"
                model_basename = "crop_model.csv"
            else:
                params_basename = "crop_params_" + str(i + 1) + ".csv"
                model_basename = "crop_model_" + str(i + 1) + ".csv"
            crop.params.save_(os.path.join(proj_dir, params_basename))
            crop.model.save_(os.path.join(proj_dir, model_basename))

        # tree info
        for i, tree in enumerate(self.trees):
            if len(self.trees) == 1:
                params_basename = "tree_params.csv"
                growth_basename = "tree_growth.csv"
                model_basename = "tree_model.csv"
            else:
                params_basename = "tree_params_" + str(i + 1) + ".csv"
                growth_basename = "tree_growth_" + str(i + 1) + ".csv"
                model_basename = "tree_model_" + str(i + 1) + ".csv"
            tree.params.save_(os.path.join(proj_dir, params_basename))
            tree.growth.save_(os.path.join(proj_dir, growth_basename))
            tree.model.save_(os.path.join(proj_dir, model_basename))

            try:  # growth file
                dest_file = os.path.basename(tree.growthFilename).split(".csv")[0]
                if len(self.trees) == 1:
                    dest_file += "_" + self.name + ".csv"
                else:
                    dest_file += "_" + self.name + "_" + str(i + 1) + ".csv"
                dest_file = os.path.join(configuration.INPUT_DIR, dest_file)
                shutil.copyfile(tree.growthFilename, dest_file)
            except AttributeError:  # no growth file
                print("attribute errr")

        # litter and fert info
        for i, litt in enumerate(self.litter):
            if len(self.litter) == 1:
                model_basename = "litter_model.csv"
            else:
                model_basename = "litter_model" + str(i + 1) + ".csv"
            litt.model.save_(os.path.join(proj_dir, model_basename))

        for i, fert in enumerate(self.fert):
            if len(self.fert) == 1:
                model_basename = "fert_model.csv"
            else:
                model_basename = "fert_model" + str(i + 1) + ".csv"
            fert.model.save_(os.path.join(proj_dir, model_basename))

    def save_description(self):
        """Print the baseline/intervention data to the description str"""

        # redirect stdout to StringIO object
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        if self.isBaseline:
            print("NEW BASELINE")
            print("Baseline name:\n\t%s" % self.name)
        else:
            print("NEW INTERVENTION")
            print("Intervention name:\n\t%s" % self.name)

        print("\nSOIL PARAMETERS")
        print("Soil parameters loaded from:")
        if self.ui.soilFromHWSD.isChecked():
            print("\tHWSD data")
        elif self.ui.soilFromCsv.isChecked():
            print("\t%s" % self.soil.soilFilename)
        else:
            print("\tcustom data")
        self.soil.soil_params.print_()

        print("\nSOIL MANAGEMENT")
        print("Soil cover (1 covered, 0 bare):\n\t", self.soil.cover)
        print("Fire:")
        if self.ui.fireNever.isChecked():
            print("\tNever")
        elif self.ui.fireInterval.isChecked():
            print("\tEvery %d years" % self.ui.fireFreq.value())
        else:
            print("\tIn years: %s" % str(self.ui.fireCustomLine.text()))

        print("\nCROP MANAGEMENT")
        if self.ui.cropNum.value() == 0:
            print("*** No crop management ***")
        else:
            for i in range(self.ui.cropNum.value()):
                print("\nCROP %d" % (i + 1))
                print(("Crop type:\n\t%s" % (str(self.ui.crop_type[i].currentText()))))
                print(
                    ("Crop yield:\n\t%.1f t DM / ha" % (self.ui.crop_yield[i].value()))
                )
                print(
                    (
                        "Residues left in field after harvest:\n\t%d %%"
                        % (self.ui.crop_leftinfield[i].value())
                    )
                )
                print(
                    (
                        "Off-farm residues burned:\n\t%s"
                        % (str(self.ui.crop_burnyes[i].isChecked()))
                    )
                )

        print("\nTREE MANAGEMENT")
        if self.ui.treeNum.value() == 0:
            print("*** No tree management ***")
        else:
            for i in range(self.ui.treeNum.value()):
                print("\nTREE %d" % (i + 1))
                print(("Tree species:\n\t%s" % (str(self.ui.tree_name[i].text()))))
                print(("Tree type:\n\t%s" % (str(self.ui.tree_type[i].currentText()))))
                print(
                    (
                        "Tree stocking density:\n\t%d / ha"
                        % (self.ui.tree_standdens[i].value())
                    )
                )
                print(
                    ("Year planted:\n\tyear %d" % (self.ui.tree_yearplanted[i].value()))
                )

                print("Thinning:")
                if self.ui.thin_neverbut[i].isChecked():
                    print("\tNever")
                elif self.ui.thin_intervalbut[i].isChecked():
                    print("\tEvery %d years" % self.ui.thin_freq[i].value())
                else:
                    print(("\tIn years: %s" % (str(self.ui.thin_customline[i].text()))))
                if not self.ui.thin_neverbut[i].isChecked():
                    print(
                        (
                            "Trees thinned when thinning occurs:\n\t%d %%"
                            % (self.ui.thin_amount[i].value())
                        )
                    )
                    print("Amount of thinned mass left in field:")
                    print(("\tStems:\n\t\t%d %%" % (self.ui.thin_stemsleft[i].value())))
                    print(
                        (
                            "\tBranches:\n\t\t%d %%"
                            % (self.ui.thin_branchesleft[i].value())
                        )
                    )

                print(
                    (
                        "Mortality rate:\n\t %d %% per year"
                        % (self.ui.mort_amount[i].value())
                    )
                )
                if self.ui.mort_amount[i].value() != 0:
                    print("Amount of dead mass left in field:")
                    print(("\tStems:\n\t\t%d %%" % (self.ui.mort_stemsleft[i].value())))
                    print(
                        (
                            "\tBranches:\n\t\t%d %%"
                            % (self.ui.mort_branchesleft[i].value())
                        )
                    )

                print("Tree growth loaded from:")
                if self.ui.growth_csvbut[i].isChecked():
                    print("\t%s" % self.trees[i].growthFilename)
                else:
                    print("\tcustom data")
                print(
                    ("Allometric:\n\t%s" % (str(self.ui.growth_allom[i].currentText())))
                )
                self.trees[i].growth.print_()

        print("\nEXTERNAL ORGANIC INPUTS")
        if self.ui.litterNum.value() == 0:
            print("*** No external organic inputs ***")
        else:
            for i in range(self.ui.litterNum.value()):
                print("\nEOI %d" % (i + 1))
                print(
                    (
                        "Input quantity:\n\t%.1f t DM / ha"
                        % (self.ui.litter_input[i].value())
                    )
                )
                print("Input frequency:")
                if self.ui.litter_intervalbut[i].isChecked():
                    print("\tEvery %d years" % self.ui.litter_freq[i].value())
                else:
                    print(
                        ("\tIn years: %s" % (str(self.ui.litter_customline[i].text())))
                    )

        print("\nSYNTHETIC FERTILISER")
        if self.ui.fertNum.value() == 0:
            print("*** No fertiliser ***")
        else:
            for i in range(self.ui.fertNum.value()):
                print("\nFERTILISER %d" % (i + 1))
                print(
                    (
                        "Input quantity:\n\t%.1f t DM / ha"
                        % (self.ui.fert_input[i].value())
                    )
                )
                print(
                    (
                        "Nitrogen content:\n\t%.1f %% N"
                        % (self.ui.fert_nitrogen[i].value())
                    )
                )
                print("Input frequency:")
                if self.ui.fert_intervalbut[i].isChecked():
                    print("\tEvery %d years" % self.ui.fert_freq[i].value())
                else:
                    print(("\tIn years: %s" % (str(self.ui.fert_customline[i].text()))))

        # save the description from the StringIO object
        sys.stdout = old_stdout
        self.description = mystdout.getvalue()
        mystdout.close()


class Soil(object):
    """Object for the soil and fire params."""

    def __init__(self, ui, gen_params):
        self.ui = ui
        self.location = gen_params.location
        self.climate = gen_params.climate

        # Soil data
        self.soil_params = self.get_soil_params()

        # soil management
        self.cover, self.fire = self.get_soil_mgmt_params()
        emit.FIRE = self.fire

    def get_soil_params(self):
        """Read soil params from the project dialog."""
        if self.ui.soilFromHWSD.isChecked():
            soil = SoilParams.from_location(self.location)
        elif self.ui.soilFromCsv.isChecked():
            # name of file. save in cfg.INP_FILES for future copying
            # to the project folder
            src_filename = str(self.ui.soilCsvBrowseBox.text())
            soil = SoilParams.from_csv(src_filename)
            self.soilFilename = src_filename  # for printing purposes
        elif self.ui.soilFromCustom.isChecked():
            soil = SoilParams.from_csv(self._print_soil_from_ui())
        else:
            log.error("Soil data input method not selected")

        return soil

    def _print_soil_from_ui(self):
        """Print the 'custom soil data' (if entered) to a csv."""
        Cy0 = self.ui.Cy0Input.value()
        clay = self.ui.clayInput.value()
        data = np.array([Cy0, clay])

        filepath = os.path.join(configuration.INPUT_DIR, "soil.csv")
        csv_handler.print_csv(filepath, data, col_names=["Cy0", "clay"])

        return filepath

    def get_soil_mgmt_params(self):
        """Read soil cover and fire info."""
        cover = self._get_cover_from_ui()

        fire = np.zeros(configuration.N_YEARS)  # N_YEARS now defined
        if self.ui.fireNever.isChecked():
            pass  # already 0
        elif self.ui.fireInterval.isChecked():
            years = list(range(0, configuration.N_YEARS, self.ui.fireFreq.value()))
            years = years[1:]
            fire[years] = 1  # shift
        elif self.ui.fireCustom.isChecked():
            years = parse_regex(str(self.ui.fireCustomLine.text()))
            years = years[years > 0]  # cut off any <=0 in the input
            years = years[years <= configuration.N_YEARS]  # cut off > cfg.N_YEARS
            fire[years - 1] = 1

        return cover, fire

    def _get_cover_from_ui(self):
        """Read soil cover from the 12 checkboxes in the ui."""
        months = [
            self.ui.janCover,
            self.ui.febCover,
            self.ui.marCover,
            self.ui.aprCover,
            self.ui.mayCover,
            self.ui.junCover,
            self.ui.julCover,
            self.ui.augCover,
            self.ui.sepCover,
            self.ui.octCover,
            self.ui.novCover,
            self.ui.decCover,
        ]
        cover = np.zeros(len(months))
        for i in range(len(months)):
            if months[i].isChecked():
                cover[i] = 1
            # else it stays 0

        return cover


class Crop(object):
    """Object for crop params and model."""

    def __init__(self, ui, crop_num):
        """
        Args:
            ui: projectDialog
            crop_num: crop number for this instance of Crop
                      -> determines which widgets to read the crop info from
                        (can have up to 10 crops)

        """

        self.ui = ui
        self.crop_num = crop_num
        self.params = self.get_crop_params()

        # Run model
        yield_ = self.ui.crop_yield[self.crop_num].value()
        left = self.ui.crop_leftinfield[self.crop_num].value() * 0.01
        self.model = CropModel(self.params, yield_, left)

        # residues burned off-farm or not
        if self.ui.crop_burnno[self.crop_num].isChecked():
            self.burn_res = False
        else:
            self.burn_res = True

    def get_crop_params(self):
        """Read crop params from project dialog."""
        crop_type = self.ui.crop_type[self.crop_num]
        index = crop_type.currentIndex()
        if index == crop_type.count() - 1:
            crop_par = self._get_custom_crop_params()
        else:
            spp = str(crop_type.currentText())
            crop_par = CropParams.from_species_name(spp)

        return crop_par

    def _get_custom_crop_params(self):
        pass


class Tree(object):
    """Object for tree params and model."""

    def __init__(self, ui, tree_num):
        """
        Args:
            ui: projectDialog instance
            tree_num: the tree number for this Tree instance
                      -> determines which widgets to read tree info from
                      (can have up to 10 trees)
        """
        self.ui = ui
        self.tree_num = tree_num

        self.params = self.get_tree_params()
        self.thin, self.thin_frac = self.get_tree_thinning()
        self.mort, self.mort_frac = self.get_tree_mortality()
        self.growth = self.get_tree_growth(self.params)

        # Run model
        initialSD = self.ui.tree_standdens[self.tree_num].value()
        yp = self.ui.tree_yearplanted[self.tree_num].value()
        self.model = TreeModel.from_defaults(
            self.params,
            self.growth,
            yearPlanted=yp,
            standDens=initialSD,
            thin=self.thin,
            thinFrac=self.thin_frac,
            mort=self.mort,
            mortFrac=self.mort_frac,
        )

    def get_tree_params(self):
        """Read tree params from project dialog."""

        tree_type = self.ui.tree_type[self.tree_num]

        index = tree_type.currentIndex()
        if index == tree_type.count() - 1:
            tree_par = self._get_custom_tree_params()
        else:
            spp = str(tree_type.currentText())
            tree_par = TreeParams.from_species_name(spp)

        return tree_par

    def _get_custom_tree_params(self):
        pass

    def get_tree_thinning(self):
        """Read tree thinning info from project dialog for this tree_num"""
        if self.ui.thin_neverbut[self.tree_num].isChecked():
            # Use default values (namely, no thinning)
            thin = None
            thin_frac = None
        else:
            thin = np.zeros(configuration.N_YEARS + 1)
            thin_amt = self.ui.thin_amount[self.tree_num].value() * 0.01
            thin_frac = np.array(
                [
                    1,
                    self.ui.thin_branchesleft[self.tree_num].value() * 0.01,
                    self.ui.thin_stemsleft[self.tree_num].value() * 0.01,
                    1,  # root always 1
                    1,
                ]
            )

            if self.ui.thin_intervalbut[self.tree_num].isChecked():
                yp = self.ui.tree_yearplanted[self.tree_num].value()
                freq = self.ui.thin_freq[self.tree_num].value()
                years = list(range(yp - 1, configuration.N_YEARS + 1, freq))
                years = years[1:]
                thin[years] = thin_amt
            elif self.ui.thin_custombut[self.tree_num].isChecked():
                years = parse_regex(str(self.ui.thin_customline[self.tree_num].text()))
                years = years[years > 0]  # cut off <=0
                years = years[
                    years <= configuration.N_YEARS + 1
                ]  # cut off > cfg.N_YEAR+1
                thin[years - 1] = thin_amt

        return thin, thin_frac

    def get_tree_mortality(self):
        """Read tree death info from project dialog for this tree_num."""
        if self.ui.mort_amount[self.tree_num].value() == 0:
            mort = None
            mort_frac = None
        else:

            mort_amt = self.ui.mort_amount[self.tree_num].value() * 0.01
            mort = np.array((configuration.N_YEARS + 1) * [mort_amt])

            mort_frac = np.array(
                [
                    1,
                    self.ui.mort_branchesleft[self.tree_num].value() * 0.01,
                    self.ui.mort_stemsleft[self.tree_num].value() * 0.01,
                    1,  # roots always 1
                    1,
                ]
            )

        return mort, mort_frac

    def get_tree_growth(self, params):
        """Read growth info from ui for this tree."""
        if self.ui.growth_csvbut[self.tree_num].isChecked():
            allom_type = self.ui.growth_allom[self.tree_num]
            index = allom_type.currentIndex()
            if index == allom_type.count() - 1:
                allometric = self._get_custom_allom()
            else:
                allometric = str(allom_type.currentText())

            # store the file so when proj is saved, it can be copied to
            # the 'input' folder
            src_file = str(self.ui.growth_csvline[self.tree_num].text())
            self.growthFilename = src_file

            # run the growth model
            growth = TreeGrowth.from_csv(params, filename=src_file, allom=allometric)

        elif self.ui.growth_custombut[self.tree_num].isChecked():
            growth = self._get_custom_tree_growth()

        return growth

    def get_custom_tree_growth():
        pass


class Litter(object):
    """Object for litter model."""

    def __init__(self, ui, litter_num):
        """Calculation of litter inputs.

        Args:
            ui: project_dialog ui
            litter_num: number for this instance of Litter
                        (up to 2 litter models for each project)
                        - determines which widgets to read info from
        """
        self.ui = ui
        self.litter_num = litter_num
        self.model = self.get_litter()

    def get_litter(self):
        """Read litter info from ui."""
        if self.ui.litter_intervalbut[self.litter_num].isChecked():
            # qty in kg - convert to tonnes here
            qty = 0.001 * self.ui.litter_input[self.litter_num].value()
            freq = self.ui.litter_freq[self.litter_num].value()
            model = LitterModel.from_defaults(freq, qty)
        elif self.ui.litter_custombut[self.litter_num].isChecked():
            # convert qty to tonnes
            qty = 0.001 * self.ui.litter_input[self.litter_num].value()
            years = parse_regex(str(self.ui.litter_customline[self.litter_num].text()))
            years = years[years > 0]  # cut off <=0
            years = years[years <= configuration.N_YEARS]  # cut off > N_YEARS
            litter = np.zeros(configuration.N_YEARS)
            litter[years - 1] = qty  # convert to 0-indexing
            model = LitterModel.from_defaults(0, 0, litterVector=litter)

        return model


class Fertiliser(object):
    """Object for fertiliser model."""

    def __init__(self, ui, fert_num):
        """Record fertiliser inputs.

        Args:
            ui: project_dialog ui
            fert_num: number for this instance of Fertiliser
                      - determines which widgets to read info from
        """
        self.ui = ui
        self.fert_num = fert_num
        self.model = self.get_fert()

    def get_fert(self):
        """Read fert info from gui."""
        if self.ui.fert_intervalbut[self.fert_num].isChecked():
            # convert mass to tonnes
            mass = 0.001 * self.ui.fert_input[self.fert_num].value()
            nitrogen = self.ui.fert_nitrogen[self.fert_num].value() * 0.01
            freq = self.ui.fert_freq[self.fert_num].value()
            model = LitterModel.synthetic_fert(freq, mass, nitrogen)
        elif self.ui.fert_custombut[self.fert_num].isChecked():
            mass = 0.001 * self.ui.fert_input[self.fert_num].value()
            nitrogen = self.ui.fert_nitrogen[self.fert_num].value() * 0.01

            years = parse_regex(str(self.ui.fert_customline[self.fert_num].text()))
            years = years[years > 0]
            years = years[years <= configuration.N_YEARS]
            fertVec = np.zeros(configuration.N_YEARS)
            fertVec[years - 1] = mass
            model = LitterModel.synthetic_fert(0, 0, nitrogen, vector=fertVec)

        return model


def parse_regex(s):
    """Parse a string with a comma-separated list of numbers/ranges.
    Return the numbers as a sorted array.

    e.g. "1,3-6,10" returns np.array([1,3,4,5,6,10])
         "10,2,9-7" returns np.array([2,7,8,9,10])

    """
    s = s.split(",")
    data = []
    for i in s:
        if "-" in i:
            j = i.split("-")
            j[0] = int(j[0])
            j[1] = int(j[1])
            for k in range(min(j), max(j) + 1):
                if k:  # not empty string
                    data.append(int(k))
        else:
            if i:  # not empty string
                data.append(int(i))

    return np.array(sorted(list(set(data))))
