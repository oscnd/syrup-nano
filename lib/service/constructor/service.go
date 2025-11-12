package constructor

import (
	"errors"
	"fmt"

	pogreb2 "github.com/akrylysov/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

type Server interface {
	Clear()
	ConstructWordSpecial(pattern string)
	ConstructFromGlob(pattern string)
	ConstructContent(filename string, content string)
	GetNo() uint64
}

type Service struct {
	config            *config.Config
	pogreb            *pogreb.Pogreb
	WordSpecialLookup map[rune][]string
	no                uint64
}

func Serve(
	config *config.Config,
	pogreb *pogreb.Pogreb,
) Server {
	r := &Service{
		config:            config,
		pogreb:            pogreb,
		WordSpecialLookup: make(map[rune][]string),
		no:                0,
	}

	// Initialize no by finding maximum token number from pogreb
	r.initializeNo()

	return r
}

func (r *Service) initializeNo() {
	// Find maximum token number by iterating through token mapper
	iter := r.pogreb.TokenMapper.Items()

	maxToken := uint64(0)
	for {
		key, _, err := iter.Next()
		if errors.Is(err, pogreb2.ErrIterationDone) {
			break // iterator exhausted
		}
		if err != nil {
			break
		}
		if key == nil {
			break
		}

		// Convert key bytes to uint64 token number
		if len(key) == 8 {
			token := util.BytesToUint64(key)
			if token > maxToken {
				maxToken = token
			}
		}
	}

	r.no = maxToken
	if maxToken > 0 {
		fmt.Printf("initialized constructor service with max token number: %d\n", maxToken)
	}
}

func (r *Service) GetNo() uint64 {
	return r.no
}
