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


## SHAMBA 1.2

default parameter value updates, bug fixes and enhancements


GENERAL IMPROVEMENTS:

1. Python updated to V3.10
2. Old non-functional GUI removed, proof of concept alternative added. This is not ready for use and is not reflected in v1.2 documentation.
3. Old raster climate and soil data removed, in favour of API.
4. Improvements to naming for clarity, e.g. cohorts vs species vs trees, stand density vs wood density.
5. Command line input is interactive
6. Environment uses docker and poetry to minimise install complications for users
7. Basic data validation added- initially for percentages
8. Typechecking and data conversions using Marshmallow
9. Added pytest for testing, plus a few basic tests
10. Tidy up code, including changing camelCase to snake_case, additional comments for clarity
11. Structure to allow choice of soil models added. This is not ready for use and is not reflected in v1.2 documentation.
12. Remove limit of 3 cohorts of trees. Note that baseline scenarios are still only allowed one cohort.

DATA & DEFAULT UPDATES:

1. Climate data: was CRU-TS 3.10, now OpenMeteo API request for daily ERA5 data
2. Soil data: was HWSD, now SoilGrids 2.0 via API
3. Global Warming Potential values: were IPCC Second Assessment Report (1995), now IPCC Sixth Assessment Report (2021)
4. Combustion factors, crop parameters, emission factors, volatilisation factors: were IPCC (2006), now	IPCC (2019)
5. Crop C content: was literature (Johnson et al., 2006; Latshaw & Miller, 1924), now IPCC (2019)


BUG FIXES:

1. In tree_growth.py: tree growth derivative functions were inconsistent in the inputs they expected. This has been standardised to expect aboveground biomass.
2. In tree_growth.py: Chave Dry allometric parameters included a typo, that has been fixed.
3. In emit.py: tree inputs reduced from fire used crop combustion factor, now fixed to use the tree combustion factor.
4. In emit.py: nitrogen_emit did not account for fire impacts on input amounts, this is now applies.
5. In calculate_emissions.py: whether residues off-field are burned is now set by user in the input template.
6. Some inconsistencies in how the input data questionnaire was set out have been updated.