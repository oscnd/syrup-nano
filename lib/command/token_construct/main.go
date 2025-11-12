package main

import (
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/service/constructor"
	"go.uber.org/fx"
)

func main() {
	// * clear pogreb database
	ClearPogreb(config.Init())

	// * main fx application
	fx.New(
		fx.Provide(
			config.Init,
			pogreb.Init,
			constructor.Serve,
		),
		fx.Invoke(
			invoke,
		),
	).Run()
}

func invoke(shutdowner fx.Shutdowner, constructor constructor.Server) {
	constructor.ConstructWordSpecial("dataset/tokenizer/word_*.jsonl")
	_ = shutdowner.Shutdown()
}
