import json


import maelstrom


class Logger:
    """Class for adding and writing logging information"""

    def __init__(self, filename):
        self.filename = filename
        self.results = dict()

    def add(self, *args):
        """Adds a section to the logger

        If only one argument is provided, then a section is initialized to an empty dictionary. This
        is useful if the order of the keys should be specified, without the values being available.

        If two arguments are provided, and the first is None, then the root is replaced with the
        second argument.

        Args:
            arg1: Section name
            arg2-N: Subsection name
            value: Value to add
        """
        if len(args) == 1:
            self.results[args[0]] = dict()
        elif len(args) == 2 and args[0] is None:
            self.results = args[1]
            return
        value = args[-1]
        args = args[:-1]
        curr = self.results
        for i, k in enumerate(args):
            if k not in curr:
                curr[k] = dict()
            if i == len(args) - 1:
                curr[k] = value
            else:
                curr = curr[k]

    def __str__(self):
        output = json.dumps(self.results, indent=4)
        return output

    def write(self):
        maelstrom.util.create_directory(self.filename)
        with open(self.filename, "w") as file:
            output = self.__str__()
            file.write(output)
