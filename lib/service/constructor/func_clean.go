package constructor

import (
	"os"

	"github.com/bsthun/gut"
)

func (r *Service) Clear() {
	if err := os.RemoveAll(*r.config.PogrebWordMapper); err != nil {
		if !os.IsNotExist(err) {
			gut.Fatal("unable to remove word mapper pogreb", err)
		}
	}

	if err := os.RemoveAll(*r.config.PogrebTokenMapper); err != nil {
		if !os.IsNotExist(err) {
			gut.Fatal("unable to remove token mapper pogreb", err)
		}
	}

	// * reset token no counter
	r.no = 0
}
