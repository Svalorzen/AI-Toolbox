def Settings( **kwargs ):
  return {
    'flags': [
        '-x', 'c++',
        '-std=c++2a',
        '-Wall', '-Wextra', '-Werror',
        '-DAI_LOGGING_ENABLED',
        '-Iinclude/',
        '-isystem/usr/include/eigen3/'
    ],
  }
