all: configure

configure: gen_config_memory gen_pointers

gen_config_memory:
	python3 deps/asic-cells/src/spi_interface/generate_config_memory.py \
		--path_to_json src/config_memory.json

gen_pointers:
	python3 deps/asic-cells/src/spi_interface/generate_pointers.py \
		--path_to_json src/pointers.json
