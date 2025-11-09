package util

func BytesToUint64(b []byte) uint64 {
	return uint64(b[0])<<56 |
		uint64(b[1])<<48 |
		uint64(b[2])<<40 |
		uint64(b[3])<<32 |
		uint64(b[4])<<24 |
		uint64(b[5])<<16 |
		uint64(b[6])<<8 |
		uint64(b[7])
}

func Uint64ToBytes(value uint64) []byte {
	return []byte{
		byte(value >> 56),
		byte(value >> 48),
		byte(value >> 40),
		byte(value >> 32),
		byte(value >> 24),
		byte(value >> 16),
		byte(value >> 8),
		byte(value),
	}
}

func MapperPayloadExtract(value []byte) (bool, uint64, uint64) {
	flag := value[0:8]
	tokenNo := BytesToUint64(value[8:16])
	count := BytesToUint64(value[16:24])

	special := flag[0]&0b00000001 == 1

	return special, tokenNo, count
}

func MapperPayloadBuild(special bool, tokenNo uint64, count uint64) []byte {
	payload := make([]byte, 24)
	if special {
		copy(payload[0:8], []byte{0b00000001, 0, 0, 0, 0, 0, 0, 0})
	}
	copy(payload[8:16], Uint64ToBytes(tokenNo))
	copy(payload[16:24], Uint64ToBytes(count))

	return payload
}
