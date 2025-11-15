package pogreb

import (
	"context"
	"fmt"

	"github.com/akrylysov/pogreb"
	"github.com/akrylysov/pogreb/fs"
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

	fileSystem := fs.OSMMap
	if !*config.PogrebWritable {
		fileSystem = &FileSystem{
			OSMMap: fs.OSMMap,
			Mem:    fs.Mem,
		}
	}

	var err error
	p.WordMapper, err = pogreb.Open(*config.PogrebWordMapper, &pogreb.Options{
		BackgroundSyncInterval:       0,
		BackgroundCompactionInterval: 0,
		FileSystem:                   fileSystem,
	})
	if err != nil {
		gut.Fatal("unable to open pogreb", err)
	}

	p.TokenMapper, err = pogreb.Open(*config.PogrebTokenMapper, &pogreb.Options{
		BackgroundSyncInterval:       0,
		BackgroundCompactionInterval: 0,
		FileSystem:                   fileSystem,
	})
	if err != nil {
		gut.Fatal("unable to open pogreb", err)
	}

	lifecycle.Append(fx.Hook{
		OnStart: func(context context.Context) error {
			return nil
		},
		OnStop: func(context context.Context) error {
			// close word mapper
			result, err := p.WordMapper.Compact()
			if err != nil {
				fmt.Printf("error word mapper compaction: %v\n", err)
			} else {
				fmt.Printf("word mapper compaction: %v\n", result)
			}
			_ = p.WordMapper.Close()

			// close token mapper
			result, err = p.TokenMapper.Compact()
			if err != nil {
				fmt.Printf("error token mapper compaction: %v\n", err)
			} else {
				fmt.Printf("token mapper compaction: %v\n", result)
			}
			_ = p.TokenMapper.Close()
			return nil
		},
	})

	return p
}
