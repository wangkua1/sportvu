"""test_docopt.py

Usage:
	test_docopt.py <dir-name>
	test_docopt.py --list <dir-names>
	test_docopt.py <dir-name> <dummy>

"""
from docopt import docopt
arguments = docopt(__doc__, version='something 1.1.1')
print(arguments)
