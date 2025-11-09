package main

import (
	"os"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
)

func ClearPogreb(config *config.Config) {
	if err := os.RemoveAll(*config.PogrebWordMapper); err != nil {
		if !os.IsNotExist(err) {
			gut.Fatal("unable to remove word mapper pogreb", err)
		}
	}

	if err := os.RemoveAll(*config.PogrebTokenMapper); err != nil {
		if !os.IsNotExist(err) {
			gut.Fatal("unable to remove token mapper pogreb", err)
		}
	}
}
