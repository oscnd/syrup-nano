package pogreb

import (
	"os"

	"github.com/akrylysov/pogreb/fs"
)

type File struct {
	fs.File
	Mem fs.File
}

type FileSystem struct {
	OSMMap fs.FileSystem
	Mem    fs.FileSystem
}

func (r *File) Write(p []byte) (n int, err error) {
	return r.Mem.Write(p)
}

func (r *File) WriteAt(p []byte, off int64) (n int, err error) {
	return r.Mem.WriteAt(p, off)
}

func (r *File) Sync() error {
	return r.Mem.Sync()
}

func (r *File) Truncate(size int64) error {
	return r.Mem.Truncate(size)
}

func (r *FileSystem) OpenFile(name string, flag int, perm os.FileMode) (fs.File, error) {
	osmMapFile, err := r.OSMMap.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}

	memFile, err := r.Mem.OpenFile(name, flag, perm)
	if err != nil {
		return nil, err
	}

	return &File{
		File: osmMapFile,
		Mem:  memFile,
	}, nil
}

func (r *FileSystem) Stat(name string) (os.FileInfo, error) {
	return r.Mem.Stat(name)
}

func (r *FileSystem) Remove(name string) error {
	return r.Mem.Remove(name)
}

func (r *FileSystem) Rename(oldpath string, newpath string) error {
	return r.Mem.Rename(oldpath, newpath)
}

func (r *FileSystem) ReadDir(name string) ([]os.FileInfo, error) {
	return r.Mem.ReadDir(name)
}

func (r *FileSystem) CreateLockFile(name string, perm os.FileMode) (fs.LockFile, bool, error) {
	return r.Mem.CreateLockFile(name, perm)
}
