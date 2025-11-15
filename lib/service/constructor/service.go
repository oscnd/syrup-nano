package constructor

import (
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

type Server interface {
	ConstructWordSpecial(pattern string)
	ConstructWordRoot(pattern string)
	ConstructFromGlob(pattern string)
	ConstructFromFile(filename string)
	ConstructContent(filename string, content string)
	GetNum() uint64
}

type Service struct {
	config            *config.Config
	pogreb            *pogreb.Pogreb
	WordSpecialLookup map[rune][]string
	WordRootLookup    map[rune][]*tuple.CompoundWord
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
		WordRootLookup:    make(map[rune][]*tuple.CompoundWord),
		no:                0,
	}

	// * resume token number initialization
	r.InitializeNo()

	return r
}
