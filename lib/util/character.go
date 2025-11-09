package util

func CharacterVowel(r rune) bool {
	vowels := "aeiou"
	for _, v := range vowels {
		if r == v {
			return true
		}
	}
	return false
}
