package pogreb

import (
	"os"

	"github.com/akrylysov/pogreb/fs"
)

type FileSystem struct {
	OSMMap fs.FileSystem
	Mem    fs.FileSystem
}

func (r *FileSystem) OpenFile(name string, flag int, perm os.FileMode) (fs.File, error) {
	return r.OSMMap.OpenFile(name, flag, perm)
}

func (r *FileSystem) Stat(name string) (os.FileInfo, error) {
	return r.OSMMap.Stat(name)
}

func (r *FileSystem) Remove(name string) error {
	return r.Mem.Remove(name)
}

func (r *FileSystem) Rename(oldpath string, newpath string) error {
	return r.Mem.Rename(oldpath, newpath)
}

func (r *FileSystem) ReadDir(name string) ([]os.FileInfo, error) {
	return r.OSMMap.ReadDir(name)
}

func (r *FileSystem) CreateLockFile(name string, perm os.FileMode) (fs.LockFile, bool, error) {
	return r.Mem.CreateLockFile(name, perm)
}
