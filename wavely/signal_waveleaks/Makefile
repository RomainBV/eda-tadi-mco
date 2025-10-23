# requirements : python3.7-venv, python3.9-venv, python3.11-venv
PYTHON_VERSIONS = 3.7 3.9 3.11
WHEEL_DIR = dist
PACKAGE_NAME = wavely.signal

.PHONY: all clean build wheels

all: clean wheels

clean:
	rm -rf build/ $(WHEEL_DIR)/ *.egg-info

wheels: clean
	@for ver in $(PYTHON_VERSIONS); do \
		pybin=$$(which python$$ver 2>/dev/null); \
		if [ -z "$$pybin" ]; then \
			echo "Python $$ver not found!"; exit 1; \
		fi; \
		venv_dir=$$(mktemp -d); \
		$$pybin -m venv $$venv_dir; \
		. $$venv_dir/bin/activate; \
		pip install wheel pyc_wheel build; \
		python -m build -n --wheel --outdir $(WHEEL_DIR); \
		package_name=${PACKAGE_NAME}; \
		if [ $$ver != "3.7" ]; then \
			package_name=$$(echo $(PACKAGE_NAME) | tr '.' '_'); \
		fi; \
		package_version=$$(python -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)"); \
		wheel_name="$${package_name}-$${package_version}-py3-none-any.whl"; \
		python -m pyc_wheel $(WHEEL_DIR)/$${wheel_name}; \
		ver_no_dot=$$(echo $$ver | tr -d '.'); \
		full_name="$${package_name}-$${package_version}-cp$${ver_no_dot}-none-any.whl"; \
		mv $(WHEEL_DIR)/$${wheel_name} $(WHEEL_DIR)/$${full_name}; \
		deactivate; \
		rm -rf $$venv_dir; \
	done
