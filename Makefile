update-model:
	python3 scripts/model.py --county All --days 7 --train-dir ./train/ --output-dir ./frontend/src/data/ --verbose --update-datasets
	python3 scripts/model.py --county Indiana --days 7 --train-dir ./train/ --output-dir ./frontend/src/data/ --verbose --update-datasets
	cd ./frontend
	yarn deploy
	cd ../
	rm -r ./public
	mv ./frontend/build ./public
	git add *
	git commit -m "updating model and datasets (auto generated commit)"
install-model-dependencies:
	pip3 install numpy pandas tensorflow requests bs4 selenium chromedriver-install
clean:
	rm -r ._temp_ -v
