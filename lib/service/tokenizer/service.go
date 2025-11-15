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
	WordSpecialLookup map[rune][]string
	WordSpecialToken  map[string]uint64
	WordRootLookup    map[rune][]*tuple.CompoundWord
}

func Serve(
	config *config.Config,
	pogreb *pogreb.Pogreb,
) Server {
	s := &Service{
		config:            config,
		pogreb:            pogreb,
		WordSpecialLookup: make(map[rune][]string),
		WordSpecialToken:  make(map[string]uint64),
		WordRootLookup:    make(map[rune][]*tuple.CompoundWord),
	}

	s.LoadWordModifier()
	s.LoadWordSpecial()

	return s
}
