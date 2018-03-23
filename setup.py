try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    'description': 'sportvu',
    'author': 'Jackson Wang',
    'author_email': 'kcjacksonwang@gmail.com',
    'version': '0.0.1',
    'packages': find_packages(),
    'name': 'sportvu'
}

setup(**config)
