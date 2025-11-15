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
	ProcessWord(word string) []*tuple.WordPair
	Decode(token uint64) string
}

type Service struct {
	config            *config.Config
	pogreb            *pogreb.Pogreb
	WordSpecialLookup map[rune][]*tuple.SpecialWord
	WordSpecialToken  map[string]uint64
}

func Serve(
	config *config.Config,
	pogreb *pogreb.Pogreb,
) Server {
	s := &Service{
		config:            config,
		pogreb:            pogreb,
		WordSpecialLookup: make(map[rune][]*tuple.SpecialWord),
		WordSpecialToken:  make(map[string]uint64),
	}

	s.LoadWordModifier()
	s.LoadWordSpecial()

	return s
}
