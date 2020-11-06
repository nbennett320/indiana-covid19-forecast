update-model:
	rm -r ._temp_ -v
	python3 scripts/update_dataset.py
install-model-dependencies:
	pip3 install numpy pandas tensorflow requests bs4 selenium chromedriver-install
clean:
	rm -r ._temp_ -v