package constructor

import "go.scnd.dev/open/syrup/nano/lib/type/enum"

func (r *Service) ConstructWordModifier() {
	// * process word modifier
	for key := range enum.WordModifier {
		r.ProcessWord(string(key), true)
		enum.WordModifier[key] = r.No
	}

	// * process word section
	for key := range enum.WordSection {
		r.ProcessWord(string(key), true)
		enum.WordSection[key] = r.No
	}

	// * process word suffix
	for key := range enum.WordSuffix {
		r.ProcessWord(string(key), true)
		enum.WordSuffix[key].TokenNo = r.No
	}
}
