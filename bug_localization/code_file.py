import logging
import re

from bug_localization.utils import get_matching_bracket

logging.basicConfig(filename='get_changed_methods.log', filemode='w', format='%(levelname)s: %(message)s',
                    level=logging.INFO)

OBJECT_NAME_PATTERN = r"(?<=\bclass\s)\w+|(?<=\binterface\s)\w+|(?<=companion object\s)\w+|(?<=companion object\s)"
METHOD_NAME_PATTERN = r".*\s(.*?)\("
COMMENT_PATTERNS = [r"/\*\*(.?)+", r"/\*(.?)+", r"//(.?)+", r"\*(.?)+"]
INLINE_COMMENT_PATTERN = r"\/\*(.?)+\*\/"
OBJECT_KEY_WORDS_IND = [" class ", " interface ", " companion object "]
OBJECT_KEY_WORDS = ["class ", "interface "]


def repl(m):
    return '#' * len(m.group())


def replace_by_pattern(obj: str, obj_name: str, pattern: str):
    """
    Can be used for detection and localization of expressions like -> {...}, ->{...}, = {...}, ={...}, {{...}},
    static {...}, static{...}
    for not mixing them up with method declarations

    :param obj: object (class or interface) represented as string without line terminators
    :param obj_name: name of the object being processed (mostly for logging purposes)
    :param pattern: pattern to clean up

    :return: obj: object as string with curly brackets replaced by []
    Since there could be several pattern matches, collections of opening/closing symbols position are returned
    open_idxs = pattern_start_idx + dist_to_open_idx: local indexes of opening brackets,
    clos_idxs: local indexes of closing brackets
    """
    obj_list = list(obj)
    dist_to_open_idx = len(pattern) - 1
    pattern_start_idx = obj.find(pattern)
    clos_idxs = [-1]
    open_idxs = [-1]
    if pattern_start_idx == -1:
        return obj, open_idxs, clos_idxs
    else:
        while pattern_start_idx > -1:
            clos_idxs.append(get_matching_bracket(obj, pattern_start_idx + dist_to_open_idx))
            obj_list[pattern_start_idx + dist_to_open_idx] = "["
            obj_list[clos_idxs[-1]] = "]"
            obj = "".join(obj_list)
            pattern_start_idx = obj.find(pattern)
            open_idxs.append(pattern_start_idx + dist_to_open_idx)
    return obj, open_idxs, clos_idxs


def process_object(obj: str, obj_name: str, language: str):
    """
    Fetches method names from the given code object (class or interface) except methods implemented ones
    in anonymous classes. Handles lambda expressions, static initializations, constructors, inner classes and
    common methods. Returns local borders of the methods/blocks

    :param obj: string representing the object without line terminators
    :param obj_name: name of the object
    :param language: language of the object being processed

    :return: methods: dict{key: name of the method, value: [opening bracket index, closing bracket index]
    """
    methods = {}
    # lambda expressions
    obj, _, _ = replace_by_pattern(obj, obj_name, "-> {")
    obj, _, _ = replace_by_pattern(obj, obj_name, "->{")
    # initialization of Collections, where possible errors should be
    # caught on the build stage
    obj, _, _ = replace_by_pattern(obj, obj_name, "= {")
    obj, _, _ = replace_by_pattern(obj, obj_name, "={")
    # every time someone uses double brace initialization, a kitten gets killed,
    # possible error also are to be caught while building
    obj, _, _ = replace_by_pattern(obj, obj_name, "{{")
    obj, _, _ = replace_by_pattern(obj, obj_name, "{ {")
    # static initialization blocks
    if language == "java":
        obj, static_open_idxs, static_clos_idxs = replace_by_pattern(obj, obj_name, "static {")
        for static_open_idx, static_clos_idx in zip(static_open_idxs, static_clos_idxs):
            if static_open_idx == -1 and static_clos_idx == -1:
                continue
            methods[obj_name + ".<cinit>"] = [static_open_idx, static_clos_idx]
        obj, static_open_idxs, static_clos_idxs = replace_by_pattern(obj, obj_name, "static{")
        for static_open_idx, static_clos_idx in zip(static_open_idxs, static_clos_idxs):
            if static_open_idx == -1 and static_clos_idx == -1:
                continue
            methods[obj_name + ".<cinit>"] = [static_open_idx, static_clos_idx]
    if language == "kt":
        obj, static_open_idxs, static_clos_idxs = replace_by_pattern(obj, obj_name, "init {")
        for static_open_idx, static_clos_idx in zip(static_open_idxs, static_clos_idxs):
            if static_open_idx == -1 and static_clos_idx == -1:
                continue
            methods[obj_name + ".<init>"] = [static_open_idx, static_clos_idx]
        obj, static_open_idxs, static_clos_idxs = replace_by_pattern(obj, obj_name, "init{")
        for static_open_idx, static_clos_idx in zip(static_open_idxs, static_clos_idxs):
            if static_open_idx == -1 and static_clos_idx == -1:
                continue
            methods[obj_name + ".<init>"] = [static_open_idx, static_clos_idx]
    # common methods handling
    method_open_idx = obj.find("{")
    curr_pos = 0
    while method_open_idx > -1:
        method_clos_idx = get_matching_bracket(obj, method_open_idx)
        curr_line = obj[0:method_open_idx]
        semicol_idx = curr_line.rfind(";")
        if semicol_idx > -1:
            curr_line = curr_line[semicol_idx:]
        obj = obj[method_clos_idx + 1:]
        if curr_line.find(" enum ") > -1:
            curr_pos += method_clos_idx + 1
            method_open_idx = obj.find("{")
            continue
        method_name = re.findall(METHOD_NAME_PATTERN, curr_line)
        if method_name:
            curr_pos += method_clos_idx + 1
            method_open_idx = obj.find("{")
            continue
        method_name = method_name[0]
        methods[method_name] = [curr_pos + method_open_idx, curr_pos + method_clos_idx]
        curr_pos += method_clos_idx + 1
        method_open_idx = obj.find("{")
    return obj, methods


class CodeFile:
    """
    Represents file with code for further manipulations with its objects and their methods
    """

    def __init__(self, file, file_name, language='java'):
        self.file_name = file_name
        self.file = file
        self.file = re.sub(INLINE_COMMENT_PATTERN, "", self.file)
        for pattern in COMMENT_PATTERNS:
            self.file = re.sub(pattern, repl, self.file)
        self.language = language
        self.multiobject = False
        self.line_to_code = self.file.split("\n")
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
        i = 0
        while i < len(self.line_to_code):
            line = self.line_to_code[i]
            self.line_to_count.append(len(line))
            object_open_line = -1
            for key_word in OBJECT_KEY_WORDS:
                if line.startswith(key_word):
                    object_open_line = 0
                    break
            for key_word in OBJECT_KEY_WORDS_IND:
                if object_open_line > -1:
                    break
                object_open_line = line.find(key_word)
                if object_open_line > -1:
                    break
            if object_open_line == -1:
                i += 1
                continue
            obj_name = re.findall(OBJECT_NAME_PATTERN, line)
            if obj_name:
                logging.debug("Object border detection confusion: " + line)
                i += 1
                continue
            obj_name = obj_name[0]
            if not obj_name:
                obj_name = "$Companion"
            object_open_idx = line.find("{")
            if self.flat_file[sum(self.line_to_count):].find("{") == -1:
                break
            while object_open_idx == -1:
                i += 1
                line = self.line_to_code[i]
                self.line_to_count.append(len(line))
                object_open_idx = line.find("{")
            object_open_idx += sum(self.line_to_count[0:i])
            object_clos_idx = get_matching_bracket(self.flat_file, object_open_idx)
            logging.info("get_objects_borders() found object = {} on lines {} - {}".
                         format(obj_name,
                                self.symbol_to_line[object_open_idx],
                                self.symbol_to_line[object_clos_idx]))
            object_borders[obj_name] = [object_open_idx, object_clos_idx]
            i += 1
        return object_borders

    def get_methods_borders(self):
        """
        Fetches methods from classes which are located in the file in reverse-order,
        each processed class is substituted with meaningless sting
        Reverse-order guarantees that methods of inner classes won't be messed up with methods of outer ones

        :return: dict{key: method name, value: [start line, end line]
        """
        result = {}
        object_borders = self.get_objects_borders()
        if len(object_borders) > 1:
            self.multiobject = True
        hierarchy = object_borders.copy()
        for obj_name in reversed(object_borders):
            borders = object_borders[obj_name]
            del hierarchy[obj_name]
            for parent in reversed(hierarchy):
                if hierarchy[parent][0] < borders[0] and hierarchy[parent][1] > borders[1]:
                    obj_name = parent + "." + obj_name
            obj, object_methods = process_object(self.flat_file[borders[0] + 1:borders[1]],
                                                 obj_name,
                                                 self.language)
            logging.info("get_methods_borders() processed object = {} on lines {} - {}".
                         format(obj_name,
                                self.symbol_to_line[borders[0]],
                                self.symbol_to_line[borders[1]]))
            ff_list = list(self.flat_file)
            ff_list[borders[0]:borders[1]] = "x" * (borders[1] - borders[0])
            self.flat_file = "".join(ff_list)
            for method in object_methods:
                method_borders = [x + borders[0] for x in object_methods[method]]
                if obj_name.endswith(method):
                    method = "<init>"
                result[obj_name + "." + method] = [self.symbol_to_line[x] + 1 for x in method_borders]
        if self.language == "kt":
            curr_pos = 0
            curr_file_part = self.flat_file[curr_pos:]
            local_function_idx = curr_file_part.find("fun ")
            while local_function_idx > -1:
                curr_line = self.line_to_code[self.symbol_to_line[curr_pos + local_function_idx]]
                local_method_name = re.findall(METHOD_NAME_PATTERN, curr_line)[0]
                local_method_open_idx = curr_pos + local_function_idx + curr_file_part[local_function_idx:].find("{")
                local_method_clos_idx = get_matching_bracket(self.flat_file, local_method_open_idx)
                if local_method_clos_idx == -1:
                    curr_num_line = self.symbol_to_line[curr_pos + local_function_idx]
                    while self.line_to_code[curr_num_line] != "":
                        curr_num_line += 1
                    result[self.file_name + "." + local_method_name] = \
                        [self.symbol_to_line[curr_pos + local_function_idx] + 1, curr_num_line]
                else:
                    result[self.file_name + "." + local_method_name] = [self.symbol_to_line[local_method_open_idx] + 1,
                                                                        self.symbol_to_line[local_method_clos_idx] + 1]
                curr_file_part = self.flat_file[local_method_clos_idx:]
                curr_pos = local_method_clos_idx
                local_function_idx = curr_file_part.find("fun")
        return result
