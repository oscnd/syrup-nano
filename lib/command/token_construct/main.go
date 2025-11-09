package main

import (
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
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
		),
		fx.Invoke(
			invoke,
		),
	).Run()
}

func invoke(config *config.Config, pogreb *pogreb.Pogreb) {
	no := uint64(0)
	ConstructWordSpecial(pogreb, &no)
	ConstructWordCode(pogreb, &no)
}
