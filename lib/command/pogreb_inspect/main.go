package main

import (
	"errors"
	"log"
	"slices"

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
		log.Printf("%s %s", gut.Base62(util.BytesToUint64(val)), string(key))
		keys = append(keys, string(key))
	}

	slices.Sort(keys)

	for _, key := range keys {
		log.Println(key)
	}

	_ = shutdowner.Shutdown()
}
