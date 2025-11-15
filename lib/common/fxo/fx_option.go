package fxo

import (
	"time"

	"go.uber.org/fx"
)

func Option() fx.Option {
	return fx.StopTimeout(24 * 60 * 60 * time.Second)
}
