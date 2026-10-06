all: configure

configure: gen_config_memory gen_pointers

gen_config_memory:
	python3 deps/asic-cells/src/spi_interface/generate_config_memory.py \
		--path_to_json src/config_memory.json

gen_pointers:
	python3 deps/asic-cells/src/spi_interface/generate_pointers.py \
		--path_to_json src/pointers.json

# Liberty files of the IHP SG13G2 SRAM macros in the slew convention of the
# standard cells, for ihp_sg13g2/config.yaml (see ihp_sg13g2/patch_sram_lib.py)
ihp_sram_libs:
	python3 ihp_sg13g2/patch_sram_lib.py
