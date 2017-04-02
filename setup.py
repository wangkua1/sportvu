try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'sportvu',
    'author': 'Jackson Wang',
    'author_email': 'kcjacksonwang@gmail.com',
    'version': '0.0.1',
    'packages': ['sportvu'],
    'name': 'sportvu'
}

setup(**config)
