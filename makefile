.PHONY: link
link:
	go build -buildmode=c-shared -o nano.so .