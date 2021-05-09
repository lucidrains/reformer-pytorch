from setuptools import setup, find_packages

setup(
  name = 'reformer_pytorch',
  packages = find_packages(exclude=['examples', 'pretraining']),
  version = '1.4.2',
  license='MIT',
  description = 'Reformer, the Efficient Transformer, Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/reformer-pytorch',
  keywords = ['transformers', 'attention', 'artificial intelligence'],
  install_requires=[
    'axial-positional-embedding>=0.1.0',
    'einops',
    'local-attention',
    'product-key-memory',
    'torch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
