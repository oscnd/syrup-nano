.PHONY: link
link:
	go build -buildmode=c-shared -o nano.so .

.PHONY: clean
clean:
	jupyter nbconvert --clear-output --ClearMetadataPreprocessor.enabled=True --inplace */*.ipynb