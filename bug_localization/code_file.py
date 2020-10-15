import re

from bug_localization.utils import get_matching_bracket

CLASS_NAME_PATTERN = r"(?<=\b\sclass\s)\w+|(?<=\b\sinterface\s)\w+"


def replace_by_pattern(obj: str, pattern: str):
    """
    Can be used for detection and localization of expressions like -> {...}, ->{...}, = {...}, ={...}, static {...}, static{...}
    for not mixing them up with method declarations
    :param obj: object (class or interface) represented as string without line terminators
    :param pattern: pattern to clean up
    :return: obj: object as string with curly brackets replaced by [],
    open_idx = pattern_start_idx + dist_to_open_idx: local index of opening bracket,
    clos_idx: local index of closing bracket
    """
    obj_list = list(obj)
    dist_to_open_idx = len(pattern) - 1
    pattern_start_idx = obj.find(pattern)
    clos_idx = -1
    if pattern_start_idx == -1:
        return obj, pattern_start_idx, clos_idx
    else:
        while pattern_start_idx > -1:
            clos_idx = get_matching_bracket(obj, pattern_start_idx + dist_to_open_idx)
            obj_list[pattern_start_idx + dist_to_open_idx] = "["
            obj_list[clos_idx] = "]"
            obj = "".join(obj_list)
            pattern_start_idx = obj.find(pattern)
    return obj, pattern_start_idx + dist_to_open_idx, clos_idx


def process_object(obj: str):
    """
    Fetches method names from the given code object (class or interface) except methods implemented ones
    in anonymous classes. Handles lambda expressions, static initializations, constructors, inner classes and
    common methods. Returns local borders of the methods/blocks

    :param obj: string representing the object without line terminators
    :return: methods: dict{key: name of the method, value: [opening bracket index, closing bracket index]
    """
    methods = {}
    # lambda expressions
    obj, _, _ = replace_by_pattern(obj, "-> {")
    obj, _, _ = replace_by_pattern(obj, "->{")
    # initialization of Collections, where possible errors should be
    # caught on the build stage
    obj, _, _ = replace_by_pattern(obj, "= {")
    obj, _, _ = replace_by_pattern(obj, "={")
    # static initialization blocks:
    obj, open_idx, clos_idx = replace_by_pattern(obj, "static {")
    if open_idx != -1 and clos_idx != -1:
        methods['<cinit>'] = [open_idx, clos_idx]
    obj, open_idx, clos_idx = replace_by_pattern(obj, "static{")
    if open_idx != -1 and clos_idx != -1:
        methods['<cinit>'] = [open_idx, clos_idx]
    return obj, methods


class CodeFile:
    """
    Represents file with code for further manipulations with its objects and their methods
    """

    def __init__(self, file, path, language='java'):
        self.file = file
        self.path = path
        self.language = language
        self.multiobject = False
        self.line_to_code = file.split("\n")
        self.flat_file = "".join(self.line_to_code)
        self.line_to_count = []
        self.symbol_to_line = {}
        num_of_lines = 0
        num_of_symbols = 0
        for s in range(len(file)):
            if file[s] == '\n':
                num_of_lines += 1
            else:
                self.symbol_to_line[num_of_symbols] = num_of_lines
                num_of_symbols += 1

    def get_objects_borders(self):
        """
        Fetches index of borders (line terminators are not taken into account) of all of the classes/interfaces
        existing in the code file, including inner ones and excluding anonymous
        :return: dict{key: name of the object, value: [opening bracket index, closing bracket index]}
        """
        object_borders = {}
        for i, line in enumerate(self.line_to_code):
            self.line_to_count.append(len(line))
            object_open_line = line.find(" class ") if line.find(" class ") > -1 else line.find(" interface ")
            if object_open_line == -1:
                continue
            object_open_idx = line.find("{")
            if object_open_idx == -1:
                while object_open_idx == -1:
                    i += 1
                    line = self.line_to_code[i]
                    object_open_idx = line.find("{")
            object_open_idx += sum(self.line_to_count[0:i])
            object_clos_idx = get_matching_bracket(self.flat_file, object_open_idx)
            object_name = re.findall(CLASS_NAME_PATTERN, line)[0]
            object_borders[object_name] = [object_open_idx, object_clos_idx]
        return object_borders

    def get_methods_borders(self):
        object_borders = self.get_objects_borders()
        if len(object_borders) > 1:
            self.multiobject = True
