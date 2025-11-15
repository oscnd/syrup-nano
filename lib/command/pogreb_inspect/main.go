package main

import (
	"errors"
	"log"
	"slices"

	pogreb2 "github.com/akrylysov/pogreb"
	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/fxo"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/util"
	"go.uber.org/fx"
)

func main() { // * main fx application
	fx.New(
		fxo.Option(),
		fx.Provide(
			config.Init,
			pogreb.Init,
		),
		fx.Invoke(
			invoke,
		),
	).Run()
}

func invoke(shutdowner fx.Shutdowner, pogreb *pogreb.Pogreb) {
	it := pogreb.WordMapper.Items()
	keys := make([]string, 0)
	for {
		key, val, err := it.Next()
		if errors.Is(err, pogreb2.ErrIterationDone) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%11d %s %11d %s\n", util.BytesToUint64(val[8:16]), gut.Base62(util.BytesToUint64(val[8:16])), util.BytesToUint64(val[16:24]), string(key))
		keys = append(keys, string(key))
	}

	slices.Sort(keys)

	for _, key := range keys {
		log.Println(key)
	}

	_ = shutdowner.Shutdown()
}
