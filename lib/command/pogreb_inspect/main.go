package main

import (
	"errors"
	"log"

	pogreb2 "github.com/akrylysov/pogreb"
	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/util"
	"go.uber.org/fx"
)

func main() { // * main fx application
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

func invoke(pogreb *pogreb.Pogreb) {
	it := pogreb.WordMapper.Items()
	for {
		key, val, err := it.Next()
		if errors.Is(err, pogreb2.ErrIterationDone) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%s %s", gut.Base62(util.BytesToUint64(val)), string(key))
	}
}
