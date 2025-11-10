package enum

type WordModifierType string

const (
	WordModifierNextCamel WordModifierType = "#nextCamel#"
	WordModifierNextUpper WordModifierType = "#nextUpper#"
)

var WordModifier = map[WordModifierType]uint64{
	WordModifierNextCamel: 0,
	WordModifierNextUpper: 0,
}
