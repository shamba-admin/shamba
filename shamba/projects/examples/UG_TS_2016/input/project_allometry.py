# File contains example allometric functions not included inside SHAMBA.
# These are written by the user. 
# Ensure that functions return aboveground biomass in kg C, and that
# a dictionary called "allometric" maps allometric keys to functions

def swietenia_macrophylla(dbh, tree_params):
    abg = 0.903*((dbh**2)*(0.6488*dbh + 1.7084))**0.684
    return abg * tree_params.carbon

gmelina_data = {
                    "dbh": [0, 25, 50, 62.5, 75, 87.5],
                    "height": [0, 6.5, 7, 7.4, 7.5, 7.6]
}

def gmelina_arborea(dbh, tree_params):
    try:
        index = gmelina_data['dbh'].index(dbh)
        height = gmelina_data['height'][index]
        abg = 0.06*(dbh**2 * height)**0.88
        return abg * tree_params.carbon
    except ValueError:
        # Handle case where exact dbh is not found in input data
        raise ValueError(f"DBH value {dbh} not found in gmelina_data")


allometric = {
    "swietenia_macrophylla": swietenia_macrophylla,
    "gmelina_arborea": gmelina_arborea,
}