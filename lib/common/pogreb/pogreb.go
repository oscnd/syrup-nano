package pogreb

import (
	"github.com/akrylysov/pogreb"
	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
)

type Pogreb struct {
	WordMapper  *pogreb.DB
	TokenMapper *pogreb.DB
}

func Init(config *config.Config) *Pogreb {
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

	return p
}
