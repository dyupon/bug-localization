class Frame:
    def __init__(self, report_id: str, frame_position: int, frame: dict, path=""):
        self.report_id = report_id
        self.frame_position = frame_position
        self.frame = frame["method_name"]
        self.file_name = frame["file_name"]
        self.line_number = frame["line_number"]
        self.path = path
        self.change_authors = set()

    def fill_path(self):
        pass

    def get_frame(self):
        return self.frame

    def get_file_name(self):
        return self.file_name

    def get_line_number(self):
        return self.line_number

    def get_path(self):
        return self.path

    def get_position(self):
        return self.frame_position

    def get_language(self):
        if self.frame.startswith("java.") or self.frame.startswith("sun."):
            return 0
        if self.file_name is None:
            return None
        if self.file_name.endswith(".java"):
            return 0
        elif self.file_name.endswith(".kt"):
            return 1
        elif self.file_name == "<generated>":
            return -1

    def get_file_source(self):
        return 0 if self.frame.find("com.") else 1

    def get_frame_length(self):
        return len(self.frame)

    def get_report_id(self):
        return self.report_id

    def days_since_file_changed(self, commits_hexsha: list):
        pass

    def get_num_people_changed(self):
        pass

    def get_method_length(self, commits_hexsha: list):
        pass
