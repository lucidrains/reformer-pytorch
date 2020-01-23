from setuptools import setup, find_packages

setup(
  name = 'reformer_pytorch',
  packages = find_packages(exclude=['example']),
  version = '0.8',
  license='GPLv3+',
  description = 'Reformer, the Efficient Transformer, Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/reformer-pytorch',
  download_url = 'https://github.com/lucidrains/reformer-pytorch/archive/v_08.tar.gz',
  keywords = ['transformers', 'attention', 'artificial intelligence'],
  install_requires=[
      'revtorch',
      'torch',
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)