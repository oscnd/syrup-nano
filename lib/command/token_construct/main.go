package main

import (
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.uber.org/fx"
)

func main() {
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

}
