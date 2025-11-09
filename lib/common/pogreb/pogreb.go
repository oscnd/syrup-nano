package pogreb

import (
	"context"

	"github.com/akrylysov/pogreb"
	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.uber.org/fx"
)

type Pogreb struct {
	WordMapper  *pogreb.DB
	TokenMapper *pogreb.DB
}

func Init(lifecycle fx.Lifecycle, config *config.Config) *Pogreb {
	p := new(Pogreb)

	var err error
	p.WordMapper, err = pogreb.Open(*config.PogrebWordMapper, &pogreb.Options{
		BackgroundSyncInterval:       0,
		BackgroundCompactionInterval: 0,
		FileSystem:                   nil,
	})
	if err != nil {
		gut.Fatal("unable to open pogreb", err)
	}

	p.TokenMapper, err = pogreb.Open(*config.PogrebTokenMapper, &pogreb.Options{
		BackgroundSyncInterval:       0,
		BackgroundCompactionInterval: 0,
		FileSystem:                   nil,
	})
	if err != nil {
		gut.Fatal("unable to open pogreb", err)
	}

	lifecycle.Append(fx.Hook{
		OnStart: func(context context.Context) error {
			return nil
		},
		OnStop: func(context context.Context) error {
			p.WordMapper.Close()
			p.TokenMapper.Close()
			return nil
		},
	})

	return p
}
