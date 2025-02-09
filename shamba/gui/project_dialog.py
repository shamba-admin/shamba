#!/usr/bin/python


"""Contains class for setting up the project dialog."""


import os
from shamba.model import configuration

from PyQt5 import QtCore, QtGui, QtWidgets
from shamba.gui.translate_ import translate_ as _
from shamba.gui.designer.project_dialog_ui import Ui_project
from shamba.gui.page_change import NextBackButtons
from shamba.gui.dialog_setup import GenericDialog, GenericPage


# Rich-text for the title page - set to baseline text by default, but set
# text programmatically to these if !isBaseline

INTERVENTION_TITLE_TEXT = _(
        "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">"
        "p, li { white-space: pre-wrap; }"
        "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">NEW INTERVENTION</span></p></body></html>"
)

INTERVENTION_SUBTITLE_TEXT = _(
        "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">"
        "p, li { white-space: pre-wrap; }"
        "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">The following screens will prompt you to enter </span></p>"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">information about the farming activities that make </span></p>"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">up the intervention</span></p></body></html>"      
)

INTERVENTION_END_TITLE_TEXT = _(
       "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">"
        "p, li { white-space: pre-wrap; }"
        "</style></head><body style=\" font-family:'Sans'; font-size:10pt; font-weight:400; font-style:normal;\">"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">END OF NEW INTERVENTION</span></p></body></html>"
) 

class ProjectDialog(GenericDialog, Ui_project):
    
    """
    Set up the Baseline/Intervention dialog that pops up when
    "add new baseline/intervention" button is pressed
    
    Set up things like toggling of pages/items, toComplete, 
    button groups, etc, for each page in the dialog stackedWidget.

    Extends the GenericDialog and Ui_project classes.
    """

    def __init__(self, isBaseline=True, parent=None):
        super(ProjectDialog, self).__init__(parent)
        self.setupUi(self)  # bare ui from designer
        

        # change some labels if it's not baseline
        self.isBaseline = isBaseline
        if not self.isBaseline:
            self.title_label.setText(INTERVENTION_TITLE_TEXT)
            self.subtitle_label.setText(INTERVENTION_SUBTITLE_TEXT)
            self.end_title_label.setText(INTERVENTION_END_TITLE_TEXT)
            self.setWindowTitle(_("Define Intervention"))    

        self.buttons = NextBackButtons(
                self.nextButton, self.backButton, self.stackedWidget)

        # Make list of all the pages in the stackedWidget
        sw = self.stackedWidget
        sw.setCurrentIndex(0)
        for w in [sw.widget(i) for i in range(sw.count())]:
            self.pageList.append(w) 
       
        self._setup_mgmtInfo_page()
        self._setup_soil_page()
        self._setup_soilMgmt_page()
        self._setup_cropMgmt_pages()
        self._setup_treeMgmt_pages()
        self._setup_treeThin_pages()
        self._setup_treeMort_pages()
        self._setup_treeGrowth_pages()
        self._setup_litter_pages()
        self._setup_fert_pages()

        for p in self.pageList:
            for par in self.parList:
                if p not in par:
                    par[p] = []
            GenericPage(
                    p, sw, self.buttons,
                    toComplete=self.toComplete[p],
                    toCompleteConditional=self.toCompleteConditional[p],
                    toToggle=self.toToggle[p],
                    buttonsToShowPage=self.buttonsToShowPage[p],
                    buttonsToHidePage=self.buttonsToHidePage[p],
                    spinboxesToShowPage=self.spinboxesToShowPage[p],
                    buttonGroups=self.buttonGroups[p],
                    comboBoxes=self.comBox[p]
            )
        
        # toggle the mort amount box a couple times so that
        # the fields on that page are shown correctly
        pages = [
                self.findChild(QtWidgets.QWidget, "treeMortPage_"+str(i))
                for i in range(self.treeNum.maximum())
        ]
        for i, p in enumerate(pages):
            mort_rate_box = p.findChild(QtWidgets.QSpinBox, "treesDead_"+str(i))
            mort_rate_box.setValue(0)
            mort_rate_box.setValue(1)

        # help dialog
        self.helpButton.clicked.connect(
                lambda: self.show_help(sw.currentWidget())
        )

        # rearrange every time page is finished so that it is right
        self.buttons.finishedPage.connect(
                lambda: self.sort_pages(sw, self.pageList)
        )
         
    def _setup_mgmtInfo_page(self):
        p = self.infoPage

        if self.isBaseline:
            type = "baseline scenario"
        else:
            type = "intervention"
        self.help_text[p] = _((
                "Use this screen to enter details of the number of species "
                "planted and external inputs added in the " + type + ". "
                "\n\nA maximum of 10 crop species and 10 tree species can "
                "be entered for each " + type + ". "
                "\n\nOrganic inputs are plant or animal material that is "
                "added to the field. Different types of organic input "
                "include leaf litter, mulch and manure. "
                "Each " + type + " can include a maximum of 2 types of "
                "organic inputs."
                "\n\nSynthetic fertilisers include any inorganic material "
                "added to the soil to improve plant growth. Different types "
                "of synthetic fertiliser are those with different "
                "proportions of the macronutrients - Nitrogen (N), "
                "Phosphorus (P) and Potassium (K). A maximum of 2 types of "
                "synthetic fertiliser can be included in each " + type + ". "
        ))

    def _setup_soil_page(self):
        p = self.soilPage
        
        self.help_text[p] = _((
                "SHAMBA default soil data is from the Harmonised World "
                "Soil Database. Using the default data is recommended "
                "for most users, unless high quality soil data are "
                "available from a local survey."
                "\n\nIf you choose to load soil data, the file must be in "
                ".csv format and match the format of the sample soil file "
                "(found in shamba/sample_input_files/soil-info.csv). "
                "\n\nIf you choose to directly enter data, enter the carbon "
                "and clay content of the soil in the boxes that appear. "
        ))

        self.buttonGroups[p] = [
                [self.soilFromHWSD, self.soilFromCsv, self.soilFromCustom]]
        self.toToggle[p] = {
                self.soilFromCustom: [self.Cy0Input, self.clayInput,
                                      self.soilCy0Label, self.soilClayLabel,
                                      self.soilCy0Unit, self.soilClayUnit],
                self.soilFromCsv: [self.soilCsvLabel, self.soilCsvBrowseBox,
                                   self.soilCsvBrowseButton]
        }
        self.toCompleteConditional[p] = [
            (self.soilCsvBrowseBox, type(self.soilCsvBrowseBox),
                self.soilFromCsv),
        ]
        # File browser
        self.soilCsvBrowseButton.clicked.connect(
                lambda: self.browse_for_file(self.soilCsvBrowseBox)) 

    def _setup_soilMgmt_page(self):
        p = self.soilMgmtPage
        self.help_text[p] = _((
                "Tick the boxes for the months (if any) when soil is "
                "covered or partially covered by crops, crop residues or "
                "other vegetation. Leave the box empty if the soil is left "
                "bare for that month."
                "\n\nIf the field if periodically burned to clear vegetation "
                "or crop residues, or by naturally occurring fires, enter "
                "the frequency of burning. Or if burning is irregular, enter "
                "the years when burning is expected to occur."
        ))

        self.coverFields = [
                self.janCover, self.febCover, self.marCover,
                self.aprCover, self.mayCover, self.junCover,
                self.julCover, self.augCover, self.sepCover,
                self.octCover, self.novCover, self.decCover,
        ]
        self.buttonGroups[p] = [
                [self.fireNever, self.fireInterval, self.fireCustom],
        ]

        self.toToggle[p] = {
                self.fireCustom: [self.fireCustomText,
                                       self.fireCustomLine]
        }
        self.toCompleteConditional[p] = [
                (self.fireCustomLine, type(self.fireCustomLine),
                 self.fireCustom)
        ]
        
        # Validator for the line edit which needs a list/range of years
        reg_exp = "^([0-9]+(-[0-9]+)?)(,([0-9]+(-[0-9]+)?))*$"
        val = QtGui.QRegExpValidator(QtCore.QRegExp(reg_exp), self)
        self.fireCustomLine.setValidator(val)
        
        # setup what the "select all" box does
        self.allCover.stateChanged.connect(self._toggle_cover_fields)

    def _toggle_cover_fields(self, checkState):
        # if selectAll was changed from notSelected to selected,
        # check all the fields
        if checkState == QtCore.Qt.Checked:
            for f in self.coverFields:
                f.setChecked(True)
        if checkState == QtCore.Qt.Unchecked:
            for f in self.coverFields:
                f.setChecked(False)

    def _setup_cropMgmt_pages(self):
        
        pages = []
        pages = [
                self.findChild(QtWidgets.QWidget, "cropMgmtPage_"+str(i))
                for i in range(self.cropNum.maximum())
        ]       
        # store lists of widgets
        self.crop_type, self.crop_yield, self.crop_leftinfield = [], [], []
        self.crop_burnno, self.crop_burnyes  = [], []

        for i, p in enumerate(pages):
            
            # help text to show
            self.help_text[p] = _((
                    "For each planted crop, select the species from the drop "
                    "down menu or, if the specific species is not listed, "
                    "choose the closest generic type. Then enter the "
                    "typical yield in tonnes of dry matter and the "
                    "percentage of crop residues that are left in the field "
                    "after harvest. If crop residues are removed from the "
                    "field, select a button to indicate whether they are "
                    "burned."
            ))

            # make widget lists
            self.crop_type.append(
                    p.findChild(QtWidgets.QComboBox, "cropType_"+str(i))
            )
            self.crop_yield.append(
                    p.findChild(QtWidgets.QDoubleSpinBox, "cropYield_"+str(i))
            )
            self.crop_leftinfield.append(
                    p.findChild(QtWidgets.QSpinBox, "leftInField_"+str(i))
            )
            self.crop_burnyes.append(
                    p.findChild(QtWidgets.QRadioButton, "cropBurnYes_"+str(i))
            )
            self.crop_burnno.append(
                    p.findChild(QtWidgets.QRadioButton, "cropBurnNo_"+str(i))
            )            


            # complete and combobox
            self.toComplete[p] = [
                    (self.crop_type[i],type(self.crop_type[i]))]
            self.comBox[p] = {self.crop_type[i]: [0,1,2,19,20,29,30,31]}

            # toggle
            toggle_lst = [
                    p.findChild(QtWidgets.QLabel, "cropCsvLabel_"+str(i)),
                    p.findChild(QtWidgets.QLabel, "cropInputType_"+str(i)),
                    p.findChild(QtWidgets.QLineEdit, "cropCsvBrowseBox_"+str(i)),
                    p.findChild(
                            QtWidgets.QPushButton, "cropCsvBrowseButton_"+str(i)),
                    p.findChild(
                            QtWidgets.QRadioButton, "cropCustomButton_"+str(i)),
                    p.findChild(QtWidgets.QRadioButton, "cropCsvButton_"+str(i)),
            ]
            self.toToggle[p] = {self.crop_type[i]: toggle_lst}
            
            # show/hide page
            self.spinboxesToShowPage[p] = [(self.cropNum, i+1)]
            
            # button groups
            self.buttonGroups[p] = [
                    [self.crop_burnyes[i], self.crop_burnno[i]]]

    def _setup_treeMgmt_pages(self):
        
        pages = [
                self.findChild(QtWidgets.QWidget, "treeMgmtPage_"+str(i))
                for i in range(self.treeNum.maximum())
        ]
        self.tree_name,self.tree_type = [],[]
        self.tree_standdens,self.tree_yearplanted  = [],[]

        for i, p in enumerate(pages):
            
            self.help_text[p] = _((
                    "For each tree species planted, specify the type of "
                    "species, planting density and year when planting is "
                    "carried out (i.e. year 0 if trees are planted at "
                    "the start of the project period)."
            ))

            # list of widgets
            self.tree_name.append(
                    p.findChild(QtWidgets.QLineEdit, "speciesName_"+str(i))
            )
            self.tree_type.append(
                    p.findChild(QtWidgets.QComboBox, "treeType_"+str(i))
            )
            self.tree_standdens.append(
                    p.findChild(QtWidgets.QSpinBox, "standDens_"+str(i))
            )
            self.tree_yearplanted.append(
                    p.findChild(QtWidgets.QSpinBox, "yearPlanted_"+str(i))
            )
            
            self.toComplete[p] = [
                    (self.tree_type[i], type(self.tree_type[i]))
            ]
            self.comBox[p] = {self.tree_type[i]: [0,1,2,4,5,6]}

            # toggle
            toggle_lst = [
                    p.findChild(QtWidgets.QLabel, "treeInputType_"+str(i)),
                    p.findChild(QtWidgets.QRadioButton, "treeCsvButton_"+str(i)),
                    p.findChild(
                            QtWidgets.QRadioButton, "treeCustomButton_"+str(i)),
                    p.findChild(QtWidgets.QLabel, "treeCsvLabel_"+str(i)),
                    p.findChild(QtWidgets.QLineEdit, "treeCsvBrowseBox_"+str(i)),
                    p.findChild(
                            QtWidgets.QPushButton, "treeCsvBrowseButton_"+str(i)),
            ]
            self.toToggle[p] = {self.tree_type[i]: toggle_lst}

            # show/hide page
            self.spinboxesToShowPage[p] = [(self.treeNum, i+1)]
            
    def _setup_treeThin_pages(self):
       
        pages = [
                self.findChild(QtWidgets.QWidget, "treeThinPage_"+str(i))
                for i in range(self.treeNum.maximum())
        ]
        self.thin_neverbut, self.thin_intervalbut = [], []
        self.thin_custombut, self.thin_customline = [], []
        self.thin_freq, self.thin_amount = [], []
        self.thin_stemsleft, self.thin_branchesleft = [],[]

        for i, p in enumerate(pages):
            
            self.help_text[p] = _((
                    "For each tree species planted, specify if and when "
                    "thinning or harvest is planned. If trees will be "
                    "thinned or harvested, specify the percentage of trees "
                    "that will be felled during each thinning or harvest "
                    "event, and the percentage of stem and branch biomass "
                    "that will be left in the field."
            ))

            self.thin_custombut.append(
                    p.findChild(QtWidgets.QRadioButton, "thinCustom_"+str(i))
            )
            self.thin_intervalbut.append(
                    p.findChild(QtWidgets.QRadioButton, "thinInterval_"+str(i))
            )
            self.thin_neverbut.append(
                    p.findChild(QtWidgets.QRadioButton, "thinNever_"+str(i))
            )
            self.thin_customline.append(
                    p.findChild(QtWidgets.QLineEdit, "thinCustomLine_"+str(i))
            )
            self.thin_freq.append(
                    p.findChild(QtWidgets.QSpinBox, "thinFreq_"+str(i))
            )
            self.thin_amount.append(
                    p.findChild(QtWidgets.QSpinBox, "treesThinned_"+str(i))
            )
            self.thin_stemsleft.append(
                    p.findChild(QtWidgets.QSpinBox, "thinnedStemsLeft_"+str(i))
            )
            self.thin_branchesleft.append(
                    p.findChild(QtWidgets.QSpinBox,"thinnedBranchesLeft_"+str(i)),
            )
            
            self.toCompleteConditional[p] = [
                    (self.thin_customline[i], 
                     type(self.thin_customline[i]), 
                     self.thin_custombut[i])
            ]

            # show/hide page
            self.spinboxesToShowPage[p] = [(self.treeNum, i+1)]

            # button groups
            never_but = p.findChild(QtWidgets.QRadioButton,
                                "thinNever_"+str(i))
            interval_but = p.findChild(QtWidgets.QRadioButton,
                                   "thinInterval_"+str(i))
            self.buttonGroups[p] = [
                    [self.thin_neverbut[i], self.thin_intervalbut[i], 
                     self.thin_custombut[i]]
            ]
        
            # toggle
            toggle_lst = [
                    self.thin_amount[i],
                    p.findChild(QtWidgets.QLabel, "treesThinnedText_"+str(i)),
                    p.findChild(QtWidgets.QLabel, "treesThinnedLabel_"+str(i)),
                    p.findChild(QtWidgets.QLabel, "thinnedLeftText_"+str(i)),
                    self.thin_stemsleft[i],
                    p.findChild(QtWidgets.QLabel, "thinnedStemsLeftText_"+str(i)),
                    p.findChild(
                            QtWidgets.QLabel, "thinnedStemsLeftLabel_"+str(i)),
                    self.thin_branchesleft[i],
                    p.findChild(
                            QtWidgets.QLabel, "thinnedBranchesLeftText_"+str(i)),
                    p.findChild(
                            QtWidgets.QLabel,"thinnedBranchesLeftLabel_"+str(i)),
            ]
            toggle_lst_custom = [
                    self.thin_customline[i],
                    p.findChild(QtWidgets.QLabel,
                                "thinCustomText_"+str(i)),
            ]
            
            self.toToggle[p] = {
                    self.thin_intervalbut[i]: toggle_lst,
                    self.thin_custombut[i]: toggle_lst + toggle_lst_custom
            }
            
            # Validator for the line edits which need a list/range of years
            reg_exp = "^([0-9]+(-[0-9]+)?)(,([0-9]+(-[0-9]+)?))*$"
            val = QtGui.QRegExpValidator(QtCore.QRegExp(reg_exp), self)
            self.thin_customline[i].setValidator(val)

    def _setup_treeMort_pages(self):
        
        pages = [
                self.findChild(QtWidgets.QWidget, "treeMortPage_"+str(i))
                for i in range(self.treeNum.maximum())
        ]
        self.mort_amount = []
        self.mort_stemsleft, self.mort_branchesleft = [], []
        
        for i,p in enumerate(pages):
            
            self.help_text[p] = _((
                    "For each tree species planted, specify the expected "
                    "mortality rate (excluding felled trees), and the "
                    "percentage of dead stem and branch biomass removed "
                    "from the field."
            ))

            self.mort_amount.append(
                    p.findChild(QtWidgets.QSpinBox, "treesDead_"+str(i))
            )
            self.mort_stemsleft.append(
                    p.findChild(QtWidgets.QSpinBox, "deadStemsLeft_"+str(i))
            )
            self.mort_branchesleft.append(
                    p.findChild(QtWidgets.QSpinBox, "deadBranchesLeft_"+str(i))
            )

            # show/hide page
            self.spinboxesToShowPage[p] = [(self.treeNum, i+1)]
        
            # toggle
            toggle_lst = [
                    p.findChild(QtWidgets.QLabel,
                                "deadLeftText_"+str(i)),
                    self.mort_stemsleft[i],
                    p.findChild(QtWidgets.QLabel,
                                "deadStemsLeftText_"+str(i)),
                    p.findChild(QtWidgets.QLabel,
                                "deadStemsLeftLabel_"+str(i)),
                    self.mort_branchesleft[i],
                    p.findChild(QtWidgets.QLabel,
                                "deadBranchesLeftText_"+str(i)),
                    p.findChild(QtWidgets.QLabel,
                                "deadBranchesLeftLabel_"+str(i)),
            ]
            self.toToggle[p] = {self.mort_amount[i]: toggle_lst}

    def _setup_treeGrowth_pages(self):
       
        pages = [
                self.findChild(QtWidgets.QWidget, "treeGrowthPage_"+str(i))
                for i in range(self.treeNum.maximum())
        ]
        self.growth_csvbut, self.growth_custombut = [], []
        self.growth_csvline, self.growth_allom = [], []
        
        for i,p in enumerate(pages):
            
            self.help_text[p] = _((
                    "For each tree species planted, load a growth data "
                    "file and choose an appropriate allometric model for "
                    "estimating total tree biomass. Choose the most "
                    "appropriate model for the project area from the list "
                    "of models provided."
                    "\n\nThe growth file must be in .csv format and match "
                    "the format of the sample growth file (found in "
                    "shamba/sample_input_files/growth.csv)."
            ))

            self.growth_allom.append(
                    p.findChild(QtWidgets.QComboBox, "allom_"+str(i))
            )
            self.growth_csvbut.append(
                    p.findChild(QtWidgets.QRadioButton,"growthCsvButton_"+str(i))
            )
            self.growth_custombut.append(
                    p.findChild(QtWidgets.QRadioButton,
                                "growthCustomButton_"+str(i))
            )
            self.growth_csvline.append(
                    p.findChild(QtWidgets.QLineEdit, 
                                "growthCsvBrowseBox_"+str(i))
            )

            # start page stuff
            self.toComplete[p] = [
                    (self.growth_allom[i], type(self.growth_allom[i]))
            ]
                    
            self.comBox[p] = {self.growth_allom[i]: [0,1,2,7,8,12,13,14]} 
            self.spinboxesToShowPage[p] = [(self.treeNum, i+1)]
            self.buttonGroups[p] = [
                    [self.growth_csvbut[i], self.growth_custombut[i]]
            ]
        
            # complete conditional
            self.toCompleteConditional[p] = [
                    (self.growth_csvline[i], 
                     type(self.growth_csvline[i]), 
                     self.growth_csvbut[i])
            ]

            # toggle
            toggle_lst = [
                    self.growth_csvline[i],
                    p.findChild(QtWidgets.QPushButton,
                                "growthCsvBrowseButton_"+str(i)),
                    p.findChild(QtWidgets.QLabel,
                                "growthCsvLabel_"+str(i))
            
            ]
            self.toToggle[p] = {self.growth_csvbut[i]: toggle_lst}
            
        # file browser stuff needs to be done explicitly since 
        # the slot is called at runtime (when csv_browse_but == button 4)
        self.growthCsvBrowseButton_0.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_0)
        )
        self.growthCsvBrowseButton_1.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_1)
        )
        self.growthCsvBrowseButton_2.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_2)
        )
        self.growthCsvBrowseButton_3.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_3)
        )
        self.growthCsvBrowseButton_4.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_4)
        )
        self.growthCsvBrowseButton_5.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_5)
        )
        self.growthCsvBrowseButton_6.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_6)
        )
        self.growthCsvBrowseButton_7.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_7)
        )
        self.growthCsvBrowseButton_8.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_8)
        )
        self.growthCsvBrowseButton_9.clicked.connect(
                lambda: self.browse_for_file(self.growthCsvBrowseBox_9)
        )
    def _setup_litter_pages(self):

        pages = [
                self.findChild(QtWidgets.QWidget, "litterPage_"+str(i)) 
                for i in range(self.litterNum.maximum())
        ]
        self.litter_input = []
        self.litter_freq = []
        self.litter_customline = []
        self.litter_intervalbut = []
        self.litter_custombut = []

        for i,p in enumerate(pages):
            self.help_text[p] = _((
                    "If organic inputs such as litter, mulch or manure that "
                    "originate outside the field are added to the field, "
                    "specify how often these inputs are added, and the "
                    "approximate dry weight of each application."
                    "\n\nIf no organic inputs are added, go back to the "
                    "first screen and set the number of kinds of external "
                    "organic inputs added to zero."
            ))

            # populate lists of widgets
            self.litter_input.append(
                    p.findChild(QtWidgets.QDoubleSpinBox,
                                "litterInput_"+str(i))
            )
            self.litter_freq.append(
                    p.findChild(QtWidgets.QSpinBox,
                                "litterFreq_"+str(i))
            )
            self.litter_customline.append(
                    p.findChild(QtWidgets.QLineEdit,
                                "litterCustomLine_"+str(i))
            )
            self.litter_intervalbut.append(
                    p.findChild(QtWidgets.QRadioButton,
                                "litterInterval_"+str(i))
            )
            self.litter_custombut.append(
                    p.findChild(QtWidgets.QRadioButton,
                                "litterCustom_"+str(i))
            )

            # complete (conditional)
            self.toCompleteConditional[p] = [
                    (self.litter_customline[i], 
                     type(self.litter_customline[i]), 
                     self.litter_custombut[i])
            ]
            
            self.spinboxesToShowPage[p] = [(self.litterNum, i+1)]
            self.buttonGroups[p] = [
                    [self.litter_custombut[i], self.litter_intervalbut[i]]]
                
            # to toggle
            toggle_lst = [
                    self.litter_customline[i],
                    p.findChild(QtWidgets.QLabel, 
                                "litterCustomText_"+str(i)),
            ]
            self.toToggle[p] = {self.litter_custombut[i]: toggle_lst}
            
            # Validator for the line edits which need a list/range of years
            reg_exp = "^([0-9]+(-[0-9]+)?)(,([0-9]+(-[0-9]+)?))*$"
            val = QtGui.QRegExpValidator(QtCore.QRegExp(reg_exp), self)
            self.litter_customline[i].setValidator(val)

    def _setup_fert_pages(self):
        
        pages = [
                self.findChild(QtWidgets.QWidget, "fertPage_"+str(i))
                for i in range(self.fertNum.maximum())
        ]
        self.fert_input = []
        self.fert_nitrogen = []
        self.fert_freq = []
        self.fert_customline = []
        self.fert_intervalbut = []
        self.fert_custombut = []

        for i,p in enumerate(pages):
            self.help_text[p] = _((
                    "If synthetic fertilisers are added to the field, "
                    "specify how often these are added, the approximate "
                    "amount added for each application, and the nitrogen "
                    "content of the fertiliser."
                    "\n\nIf the fertiliser used has an NPK rating on the "
                    "packaging (e.g. 16-4-8), the first number in the "
                    "sequence is the percentage nitrogen (e.g. 16% in this "
                    "example)."
                    "\n\nIf no synthetic fertilisers are added, go back to "
                    "the first screen and set the number of kinds of "
                    "synthetic fertiliser added to zero."
            ))
            # populate the widget lists
            self.fert_input.append(
                    p.findChild(QtWidgets.QDoubleSpinBox,
                                "fertInput_"+str(i))
            )
            self.fert_nitrogen.append(
                    p.findChild(QtWidgets.QSpinBox,
                                "fertNitrogen_"+str(i))
            )
            self.fert_freq.append(
                    p.findChild(QtWidgets.QSpinBox,
                                "fertFreq_"+str(i))
            )
            self.fert_customline.append(
                    p.findChild(QtWidgets.QLineEdit, 
                                "fertCustomLine_"+str(i))
            )
            self.fert_custombut.append(
                    p.findChild(QtWidgets.QRadioButton,
                                "fertCustom_"+str(i))
            )
            self.fert_intervalbut.append(
                    p.findChild(QtWidgets.QRadioButton,
                                "fertInterval_"+str(i))
            )
            
            # to complete
            self.toCompleteConditional[p] = [
                    (self.fert_customline[i], 
                     type(self.fert_customline[i]), 
                     self.fert_custombut[i])
            ]
             
            # show/hide page
            self.spinboxesToShowPage[p] = [(self.fertNum, i+1)]

            # button groups
            self.buttonGroups[p] = [
                    [self.fert_intervalbut[i], self.fert_custombut[i]]]
            
            # to toggle
            toggle_lst = [
                    self.fert_customline[i],
                    p.findChild(QtWidgets.QLabel,
                                "fertCustomText_"+str(i)),
            ]
            self.toToggle[p] = {self.fert_custombut[i]: toggle_lst}

            # Validator for the line edits which need a list/range of years
            reg_exp = "^([0-9]+(-[0-9]+)?)(,([0-9]+(-[0-9]+)?))*$"
            val = QtGui.QRegExpValidator(QtCore.QRegExp(reg_exp), self)
            self.fert_customline[i].setValidator(val)    


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = ProjectDialog()
    sys.exit(ui.exec_())
