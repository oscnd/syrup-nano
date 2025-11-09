package enum

type WordModifierType string

const (
	WordModifierNextCamel  WordModifierType = "#nextCamel#"
	WordModifierNextUpper  WordModifierType = "#nextUpper#"
	WordModifierNextPlural WordModifierType = "#nextPlural#"
	WordModifierNextEd     WordModifierType = "#nextEd#"
	WordModifierNextEr     WordModifierType = "#nextEr#"
	WordModifierNextIng    WordModifierType = "#nextIng#"
	WordModifierNextLy     WordModifierType = "#nextLy#"
)

var WordModifier = map[WordModifierType]uint64{
	WordModifierNextCamel:  0,
	WordModifierNextUpper:  0,
	WordModifierNextPlural: 0,
	WordModifierNextEd:     0,
	WordModifierNextEr:     0,
	WordModifierNextIng:    0,
	WordModifierNextLy:     0,
}
