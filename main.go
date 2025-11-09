package main

import "C"

func main() {}

//export encode
func encode(text *C.char) *C.char {
	result := C.GoString(text)
	return C.CString(result + "!")
}
