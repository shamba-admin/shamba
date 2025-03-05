"""Run the emissions model from the gui"""

import os

import numpy as np

import model.command_line.emit as Emit
from model import configuration
from model.command_line.soil_model import ForwardRothC, InverseRothC


class Emissions(object):

    def __init__(self, ui, gen_params, baseline, intervention):
        """Initialise the forwardRothC calculations for give baseline
        and interventions. Called when "plot" button pressed.

        Args:
            ui: main_ui instance
            gen_params: instance of General class
            baseline: instance of Project class for baseline activities
            interventions: intervention instances of Project

        """
        self.ui = ui
        self.gen_params = gen_params
        self.baseline = baseline
        self.intervention = intervention

        # Inverse soil model
        self.eqRoth = InverseRothC(
            self.baseline.soil.soil_params, self.gen_params.climate
        )

        self._make_lists()
        self.run_soil_model()  # forward soil model
        self.run_emit_model()  # emission model

        # self.save_data()

    def _make_lists(self):
        # Lists of crop, tree and models
        self.crop_models_base = []
        self.crop_models_inter = []
        self.crop_burn_res_base = []
        self.crop_burn_res_inter = []
        self.tree_models_base = []
        self.tree_models_inter = []
        self.litter_models_base = []
        self.litter_models_inter = []
        self.fert_models_base = []
        self.fert_models_inter = []

        for c in self.baseline.crops:
            self.crop_models_base.append(c.model)
            self.crop_burn_res_base.append(c.burn_res)
        for c in self.intervention.crops:
            self.crop_models_inter.append(c.model)
            self.crop_burn_res_inter.append(c.burn_res)
        for t in self.baseline.trees:
            self.tree_models_base.append(t.model)
        for t in self.intervention.trees:
            self.tree_models_inter.append(t.model)
        for li in self.baseline.litter:
            self.litter_models_base.append(li.model)
        for li in self.intervention.litter:
            self.litter_models_inter.append(li.model)
        for f in self.baseline.fert:
            self.fert_models_base.append(f.model)
        for f in self.intervention.fert:
            self.fert_models_inter.append(f.model)

        # see if residues are burned off-farm

    def run_soil_model(self):

        # run baseline to y=0 first
        self.y0roth = ForwardRothC(
            self.baseline.soil.soil_params,
            self.gen_params.climate,
            self.baseline.soil.cover,
            Ci=self.eqRoth.eqC,
            crop=self.crop_models_base,
            tree=self.tree_models_base,
            litter=self.litter_models_base,
            solveToValue=True,
        )
        self.roth_base = ForwardRothC(
            self.intervention.soil.soil_params,
            self.gen_params.climate,
            self.baseline.soil.cover,
            Ci=self.y0roth.SOC[-1],
            crop=self.crop_models_base,
            tree=self.tree_models_base,
            litter=self.litter_models_base,
        )
        self.roth_inter = ForwardRothC(
            self.intervention.soil.soil_params,
            self.gen_params.climate,
            self.intervention.soil.cover,
            Ci=self.y0roth.SOC[-1],
            crop=self.crop_models_inter,
            tree=self.tree_models_inter,
            litter=self.litter_models_inter,
        )

    def run_emit_model(self):
        self.emit_base = Emit.create(
            no_of_years=configuration.N_YEARS,
            forRothC=self.roth_base,
            crop=self.crop_models_base,
            tree=self.tree_models_base,
            litter=self.litter_models_base,
            fert=self.fert_models_base,
            burnOff=self.crop_burn_res_base,
        )
        self.emit_inter = Emit.create(
            no_of_years=configuration.N_YEARS,
            forRothC=self.roth_inter,
            crop=self.crop_models_inter,
            tree=self.tree_models_inter,
            litter=self.litter_models_inter,
            fert=self.fert_models_inter,
            burnOff=self.crop_burn_res_inter,
        )

        self.baseline.total_emissions = np.sum(self.emit_base)
        self.intervention.total_emissions = np.sum(self.emit_inter)

    def save_data(self):
        base_dir = os.path.join(
            configuration.OUTPUT_DIR, "baselines", self.baseline.name
        )
        inter_dir = os.path.join(
            configuration.OUTPUT_DIR, "interventions", self.intervention.name
        )

        self.eqRoth.save_(os.path.join(base_dir, "soil_model_inverse.csv"))
        self.y0roth.save_(os.path.join(base_dir, "soil_model_to_y0.csv"))
        self.roth_base.save_(os.path.join(base_dir, "soil_model.csv"))
        self.roth_inter.save_(os.path.join(inter_dir, "soil_model.csv"))
        # TODO: is this a bug? Why are we using self.emit_base twice?
        # self.emit_base.save_(
        #     self.emit_base, file=os.path.join(base_dir, "emissions.csv")
        # )
        # self.emit_inter.save_(
        #     self.emit_inter, file=os.path.join(inter_dir, "emissions.csv")
        # )
        Emit.save(
            self.emit_base, self.emit_base, file=os.path.join(base_dir, "emissions.csv")
        )
        Emit.save(
            self.emit_inter,
            self.emit_inter,
            file=os.path.join(inter_dir, "emissions.csv"),
        )

        self.save_total_emissions()

    def save_total_emissions(self):
        # append to the file with all the emissions
        try:
            # read existing file
            with open(
                os.path.join(configuration.OUTPUT_DIR, "emissions_total.csv"), "r"
            ) as f:
                col_names = f.readline()
                col_names = col_names[:-1]  # remove newline char
                values = f.readline()
                values = values[:-1]
        except IOError:
            # file doesn't exist, so use empty string to append
            col_names = ""
            values = ""

        if self.baseline.name not in col_names.split(","):
            col_names += "," + self.baseline.name
            tot_emit = np.sum(self.emit_base)
            values += ",%.5f" % tot_emit

        if self.intervention.name not in col_names.split(","):
            col_names += "," + self.intervention.name
            tot_emit = np.sum(self.emit_inter)
            values += ",%.5f" % tot_emit

        if col_names.startswith(","):
            col_names = col_names[1:]
        if values.startswith(","):
            values = values[1:]

        with open(
            os.path.join(configuration.OUTPUT_DIR, "emissions_total.csv"), "w"
        ) as f:
            f.write(col_names + "\n" + values + "\n")
