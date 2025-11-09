package main

import "C"

import (
	"encoding/json"

	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/service/tokenizer"
	"go.uber.org/fx"
)

type App struct {
	Config    *config.Config
	Pogreb    *pogreb.Pogreb
	Tokenizer tokenizer.Server
	Shutdown  func()
}

var app *App

func main() {}

//export load
func load() {
	go fx.New(
		fx.Provide(
			config.Init,
			pogreb.Init,
			tokenizer.Serve,
		),
		fx.Invoke(
			func(shutdowner fx.Shutdowner, config *config.Config, pogreb *pogreb.Pogreb, tokenizer tokenizer.Server) {
				app = &App{
					Config:    config,
					Pogreb:    pogreb,
					Tokenizer: tokenizer,
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
