# Power grid of Chameleon on IHP SG13G2: LibreLane's default PDN configuration, plus the
# connection of the SRAM macros to the grid.
#
# The power pins of the IHP SRAM macros (VDD!, VDDARRAY!, VSS!) are vertical Metal4 stripes
# across the full macro height. The default macro grid only connects TopMetal1 to TopMetal2,
# so the horizontal TopMetal2 straps that cross the macros are connected down to the Metal4
# stripes here. PDN_MACRO_CONNECTIONS in config.yaml ties the pins to the supply nets.

source $::env(SCRIPTS_DIR)/openroad/common/pdn_cfg.tcl

add_pdn_connect \
    -grid macro \
    -layers "Metal4 $::env(PDN_HORIZONTAL_LAYER)"
