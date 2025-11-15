package constructor

import (
	"time"

	"github.com/bsthun/gut"
	"go.scnd.dev/open/syrup/nano/lib/common/config"
	"go.scnd.dev/open/syrup/nano/lib/common/pogreb"
	"go.scnd.dev/open/syrup/nano/lib/type/tuple"
)

type Server interface {
	ConstructWordSpecial(pattern string)
	ConstructFromGlob(pattern string)
	ConstructFromFile(filename string)
	ConstructContent(filename string, content string)
	GetNum() uint64
}

type Service struct {
	config            *config.Config
	pogreb            *pogreb.Pogreb
	WordSpecialLookup map[rune][]*tuple.SpecialWord
	No                uint64
	LastLogged        *time.Time
}

func Serve(
	config *config.Config,
	pogreb *pogreb.Pogreb,
) Server {
	r := &Service{
		config:            config,
		pogreb:            pogreb,
		WordSpecialLookup: make(map[rune][]*tuple.SpecialWord),
		No:                0,
		LastLogged:        gut.Ptr(time.Now()),
	}

	// * resume token number initialization
	r.InitializeNo()

	// * construct word modifiers
	r.ConstructWordModifier()

	return r
}
