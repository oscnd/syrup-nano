package constructor

import (
	"errors"
	"fmt"

	pogreb2 "github.com/akrylysov/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) InitializeNo() {
	// Find maximum token number by iterating through word mapper
	iter := r.pogreb.WordMapper.Items()

	maxToken := uint64(0)
	for {
		key, value, err := iter.Next()
		if errors.Is(err, pogreb2.ErrIterationDone) {
			break
		}
		if err != nil {
			break
		}
		if key == nil || value == nil {
			continue
		}

		// Extract token number using MapperPayloadExtract
		_, tokenNo, _ := util.MapperPayloadExtract(value)
		if tokenNo > maxToken {
			maxToken = tokenNo
		}
	}

	r.No = maxToken
	if maxToken > 0 {
		fmt.Printf("initialized constructor service with max token number: %d\n", maxToken)
	}
}

func (r *Service) GetNum() uint64 {
	return r.No
}
