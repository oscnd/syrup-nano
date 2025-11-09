package tokenizer

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) LoadWordModifier() {
	// loop through modifiers
	for key := range enum.WordModifier {
		modifierName := string(key)
		value, err := r.pogreb.WordMapper.Get([]byte(modifierName))
		if err != nil || value == nil {
			fmt.Printf("modifier %s not found in pogreb\n", modifierName)
			continue
		}

		tokenNo, _ := util.MapperPayloadExtract(value)
		enum.WordModifier[key] = tokenNo
		fmt.Printf("loaded modifier %s with tokenNo: %d\n", modifierName, tokenNo)
	}
}
