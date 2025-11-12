package main

import "C"

import (
	"encoding/json"

	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/service/constructor"
	"go.scnd.dev/open/syrup/nano/lib/service/tokenizer"
	"go.uber.org/fx"
)

type App struct {
	Config      *config.Config
	Pogreb      *pogreb.Pogreb
	Constructor constructor.Server
	Tokenizer   tokenizer.Server
	Shutdown    func()
}

var app *App

func main() {}

//export load
func load() {
	go fx.New(
		fx.Provide(
			config.Init,
			pogreb.Init,
			constructor.Serve,
			tokenizer.Serve,
		),
		fx.Invoke(
			func(shutdowner fx.Shutdowner, config *config.Config, pogreb *pogreb.Pogreb, constructor constructor.Server, tokenizer tokenizer.Server) {
				app = &App{
					Config:      config,
					Pogreb:      pogreb,
					Tokenizer:   tokenizer,
					Constructor: constructor,
					Shutdown: func() {
						_ = shutdowner.Shutdown()
					},
				}
			},
		),
	).Run()
}

//export encode
func encode(text *C.char) *C.char {
	wordPairs := app.Tokenizer.ProcessLine(C.GoString(text))
	bytes, err := json.Marshal(wordPairs)
	if err != nil {
		return C.CString(err.Error())
	}
	return C.CString(string(bytes))
}

//export decode
func decode(token C.ulonglong) *C.char {
	result := app.Tokenizer.Decode(uint64(token))
	return C.CString(result)
}

//export clear
func clear() {
	app.Constructor.Clear()
}

//export get_num
func get_num() C.ulonglong {
	return C.ulonglong(app.Constructor.GetNum())
}

//export construct_word_special
func construct_word_special(pattern *C.char) {
	app.Constructor.ConstructWordSpecial(C.GoString(pattern))
}

//export construct_from_glob
func construct_from_glob(pattern *C.char) {
	app.Constructor.ConstructFromGlob(C.GoString(pattern))
}

//export construct_content
func construct_content(filename *C.char, content *C.char) {
	app.Constructor.ConstructContent(C.GoString(filename), C.GoString(content))
}
