# Versions

## SHAMBA 1.1

bug fixes and enhancements

MAJOR ADDITIONS:

1. Expanded command line file shamba_command_line.py to run with several tree cohorts and produce summaries of information for typical use of SHAMBA for developing Plan Vivo projects to `shamba_command_line.py` file through a dictionary

2. Created an Excel based document 'SHAMBA input output template v1.xlsx' to implement Plan Vivo SHAMBA methodlogy. The document is in questionnaire style and accepts all background information, data inputs and references for a Plan Vivo style anlaysis. It generates as csv input file to link to the SHAMBA command line, manually accepts and graphs outputs from SHAMBA. It provides the supporting documentation needed to get a SHAMBA estimate approved by Plan Vivo.

3. Included example of Excel sheet and new command line file 'example_SHAMBA_input_output_uganda_tech_spec.xslx', and added and updated user guide and SHAMBA installation instructions. Also added Spanish version of installation instructions

BUG FIXES:

1. Created seperate python files for command line and GUI versions. File titles for the command line version finish in 'cl' This was to stop development of command line version intefering with the static GUI model files.

2. Fixed bug in emit_cl.py and soil_model_cl.py files where fire could not differ between baseline and intervention

3. Fixed bug in litter_cl.py where it would not accept a zero value for organic or synthetic fertiliser inputs (i.e. where they were not used)

4. Altered tree_growth_cl.py file to use dictionary to link to growth rates from 'SHAMBA input output template v1.xlsx' instead of seperate growth.csv input file

5. Fixed bug in tree_model_cl.py where it would not allow for baselines or scenarios with no trees

6. fixed labelling on some output graphs and csv files
