update-model:
	python3 scripts/model.py --county All --days 7 --train-dir ./train/ --output-dir ./frontend/src/data/ --verbose --update-datasets
	python3 scripts/model.py --county Indiana --days 7 --train-dir ./train/ --output-dir ./frontend/src/data/ --verbose --update-datasets
	git add *
	git commit -m "updating model and datasets (auto generated commit)"
update-frontend:
	python3 scripts/update_frontend.py
	git add *
	git commit -m "rebuild and deploy frontend (auto generated commit)"
update:
	python3 scripts/model.py --county All --days 7 --train-dir ./train/ --output-dir ./frontend/src/data/ --verbose --update-datasets
	python3 scripts/model.py --county Indiana --days 7 --train-dir ./train/ --output-dir ./frontend/src/data/ --verbose --update-datasets
	git add *
	git commit -m "updating model and datasets (auto generated commit)"
	python3 scripts/update_frontend.py
	git add *
	git commit -m "rebuild and deploy frontend (auto generated commit)"
	git push origin main
install-model-dependencies:
	pip3 install numpy pandas tensorflow requests bs4 selenium chromedriver-install
clean:
	rm -r ._temp_ -v
