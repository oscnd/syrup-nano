package tokenizer

import (
	"fmt"

	"go.scnd.dev/open/syrup/nano/lib/type/enum"
	"go.scnd.dev/open/syrup/nano/lib/util"
)

func (r *Service) LoadWordModifier() {
	// loop through word modifiers
	for key := range enum.WordModifier {
		modifierName := string(key)
		value, err := r.pogreb.WordMapper.Get([]byte(modifierName))
		if err != nil || value == nil {
			fmt.Printf("modifier %s not found in pogreb\n", modifierName)
			continue
		}

		_, tokenNo, _ := util.MapperPayloadExtract(value)
		enum.WordModifier[key] = tokenNo
		fmt.Printf("loaded modifier %s with tokenNo: %d\n", modifierName, tokenNo)
	}
	// loop through word sections
	for key := range enum.WordSection {
		modifierName := string(key)
		value, err := r.pogreb.WordMapper.Get([]byte(modifierName))
		if err != nil || value == nil {
			fmt.Printf("section %s not found in pogreb\n", modifierName)
			continue
		}

		_, tokenNo, _ := util.MapperPayloadExtract(value)
		enum.WordSection[key] = tokenNo
		fmt.Printf("loaded section %s with tokenNo: %d\n", modifierName, tokenNo)
	}

	// loop through word suffix
	for key := range enum.WordSuffix {
		suffixName := string(key)
		value, err := r.pogreb.WordMapper.Get([]byte(suffixName))
		if err != nil || value == nil {
			fmt.Printf("suffix modifier %s not found in pogreb\n", suffixName)
			continue
		}

		_, tokenNo, _ := util.MapperPayloadExtract(value)
		enum.WordSuffix[key].TokenNo = tokenNo
		fmt.Printf("loaded suffix modifier %s with tokenNo: %d\n", suffixName, tokenNo)
	}
}
