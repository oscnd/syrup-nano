package constructor

import "go.scnd.dev/open/syrup/nano/lib/type/enum"

func (r *Service) ConstructWordModifier() {
	// * process word modifier
	for key := range enum.WordModifier {
		r.ProcessWord(string(key))
		enum.WordModifier[key] = r.no
	}

	// * process word suffix
	for key := range enum.WordSuffix {
		r.ProcessWord(string(key))
		enum.WordSuffix[key].TokenNo = r.no
	}
}
