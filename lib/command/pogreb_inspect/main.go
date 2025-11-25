package main

import (
	"errors"
	"fmt"
	"log"
	"slices"

	pogreb2 "github.com/akrylysov/pogreb"
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
	max := uint64(0)
	for {
		key, val, err := it.Next()
		if errors.Is(err, pogreb2.ErrIterationDone) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		special, no, count := util.MapperPayloadExtract(val)
		fmt.Printf("%t %11d %11d %s\n", special, no, count, key)
		keys = append(keys, string(key))
		if no > max {
			max = no
		}
	}

	slices.Sort(keys)

	for _, key := range keys {
		log.Println(key)
	}

	it = pogreb.TokenMapper.Items()
	for {
		key, val, err := it.Next()
		if errors.Is(err, pogreb2.ErrIterationDone) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		fmt.Printf("%d: %s\n", util.BytesToUint64(key), string(val))
	}

	log.Println("max:", max)

	_ = shutdowner.Shutdown()
}
