package main

import "C"
import "go.scnd.dev/open/syrup/nano/lib/tokenizer"

func main() {}

//export encode
func encode(text *C.char) *C.char {
	result := C.GoString(text)
	tokenizer.Encode(result)
	return C.CString("Hello")
}
