.PHONY: all install test notebooks

all:
	@echo "Add things to this Makefile"

install:
	pip install -r requirements.txt

test:
	py.test 
	
# Convert Jupyter Notebooks to HTML
notebooks:
	jupyter nbconvert --execute --to=html --FilesWriter.build_directory=./notebooks ./notebooks/*.ipynb
