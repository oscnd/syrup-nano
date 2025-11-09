package tokenizer

import (
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

type Server interface {
	LoadWordModifier()
	LoadWordSpecial()
	ProcessLine(line string) []*tuple.WordPair
	ProcessWord(word string) *tuple.WordPair
}

type Service struct {
	config            *config.Config
	pogreb            *pogreb.Pogreb
	WordSpecialLookup map[string][]string
	WordSpecialToken  map[string]uint64
}

func Serve(
	config *config.Config,
	pogreb *pogreb.Pogreb,
) Server {
	s := &Service{
		config:            config,
		pogreb:            pogreb,
		WordSpecialLookup: make(map[string][]string),
		WordSpecialToken:  make(map[string]uint64),
	}

	s.LoadWordModifier()
	s.LoadWordSpecial()

	return s
}
