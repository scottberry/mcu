from distutils.core import setup

setup(name='mcu',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/scottberry/mcu',
      author='Scott Berry',
      author_email='scottdberry@gmail.com',
      license='GNU GPLv3',
      packages=['mcu'],
      install_requires=[
          'numpy',
          'mahotas',
      ],
      zip_safe=False)
