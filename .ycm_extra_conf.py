import os, sys
import subprocess

#change gcc version with command
#%s/5.3.0/4.8.5/g
#gcc -xc++ -E -v -

default_flags = [
'-Wall',
'-Wextra',
'-Werror',
'-Wc++98-compat',
'-Wno-long-long',
'-Wno-variadic-macros',
'-fexceptions',
'-DNDEBUG',
'-std=c++11',
'-x',
'c++',
  #project define here
  #cuda
'-isystem',
'/usr/local/cuda/include',
'-isystem',
'/usr/include/c++/4.2.1',
'-isystem',
'./release/third_party/include',
'-isystem',
"/Library/Developer/CommandLineTools/usr/include/c++/v1",

'-DHAS_CUDA',
'-I',
'/Users/dzh/github/Paddle/paddle/parameter',
'-I',
'/Users/dzh/github/Paddle/paddle/pserver',
'-I',
'/Users/dzh/github/Paddle/paddle/trainer',
'-I',
'/Users/dzh/github/Paddle/paddle/gserver',
'-I',
'/Users/dzh/github/Paddle/paddle/math',
'-I',
'/Users/dzh/github/Paddle/paddle/api',
'-I',
'/Users/dzh/github/Paddle/paddle/utils',
'-I',
'/Users/dzh/github/Paddle/paddle/cuda',
'-I',
'/Users/dzh/github/Paddle',

# This path will only work on OS X, but extra paths that don't exist are not harmful
'-isystem'
'/System/Library/Frameworks/Python.framework/Headers',
'-isystem',
'/usr/include',
'-isystem',
'/usr/local/include',
'-isystem',
'/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1',
'-isystem',
'/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include',
  
#define with g++ complier include files
'-isystem',
'/usr/local/Cellar/gcc48/4.8.5/lib/gcc/4.8/gcc/x86_64-apple-darwin16.0.0/4.8.5/../../../../../../include/c++/4.8.5',
'-isystem',
'/usr/local/Cellar/gcc48/4.8.5/lib/gcc/4.8/gcc/x86_64-apple-darwin16.0.0/4.8.5/../../../../../../include/c++/4.8.5/x86_64-apple-darwin16.0.0',
'-isystem',
'/usr/local/Cellar/gcc48/4.8.5/lib/gcc/4.8/gcc/x86_64-apple-darwin16.0.0/4.8.5/../../../../../../include/c++/4.8.5/backward',
'-isystem',
'/usr/local/Cellar/gcc48/4.8.5/lib/gcc/4.8/gcc/x86_64-apple-darwin16.0.0/4.8.5/include',
'-isystem',
'/usr/local/Cellar/gcc48/4.8.5/include',
'-isystem',
'/usr/local/Cellar/gcc48/4.8.5/lib/gcc/4.8/gcc/x86_64-apple-darwin16.0.0/4.8.5/include-fixed',
]

def DirectoryOfThisScript():
  return os.path.dirname( os.path.abspath( __file__ ) )


def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag

    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )

    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break

      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break

    if new_flag:
      new_flags.append( new_flag )
  return new_flags

# Thanks to https://github.com/decrispell/vim-config for this code
def FlagsForFile( filename, **kwargs ):
    """ given the source filename, return the compiler flags """
    opt_basename = '.clang_complete'
    curr_dir = os.path.dirname(filename)
    opt_fname = os.path.join(curr_dir, opt_basename)
    # keep traversing up the tree until we find the file, or hit the root
    while not os.path.exists(opt_fname):
        new_dir = os.path.dirname(curr_dir)
        if new_dir == curr_dir:
          # we've reached the root of the tree
          break
        curr_dir = new_dir
        opt_fname = os.path.join(curr_dir, opt_basename)
    try:
      fd = open(opt_fname, 'r')
    except IOError:
        return {'flags': default_flags, 'do_cache': True}
    flags = [line.strip() for line in fd.readlines()]
    relative_to = os.path.dirname(opt_fname)
    flags = MakeRelativePathsInFlagsAbsolute(flags, relative_to)
    return {
      'flags': flags, 'do_cache': True
    }
