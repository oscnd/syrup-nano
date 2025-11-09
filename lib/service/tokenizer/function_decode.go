package tokenizer

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) Decode(token uint64) string {
	// * pogreb token mapper get
	value, err := r.pogreb.TokenMapper.Get(util.Uint64ToBytes(token))
	if err != nil {
		fmt.Printf("error getting token %d: %v\n", token, err)
		return ""
	}
	if value == nil {
		fmt.Printf("token %d not found\n", token)
		return ""
	}
	return string(value)
}
