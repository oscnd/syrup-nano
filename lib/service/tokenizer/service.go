package tokenizer

import (
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

type Server interface {
	LoadWordModifier()
	LoadWord()
	ProcessLine(line string) []*tuple.WordPair
	ProcessWord(word string) []*tuple.WordPair
	Decode(token uint64) string
}

type Service struct {
	config            *config.Config
	pogreb            *pogreb.Pogreb
	WordLookup        map[rune][]*tuple.SpecialWord
	WordSpecialLookup map[rune][]*tuple.SpecialWord
	WordToken         map[string]uint64
}

func Serve(
	config *config.Config,
	pogreb *pogreb.Pogreb,
) Server {
	s := &Service{
		config:            config,
		pogreb:            pogreb,
		WordLookup:        make(map[rune][]*tuple.SpecialWord),
		WordSpecialLookup: make(map[rune][]*tuple.SpecialWord),
		WordToken:         make(map[string]uint64),
	}

	s.LoadWordModifier()
	s.LoadWord()

	return s
}
