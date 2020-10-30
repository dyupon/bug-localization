class Frame:
    def __init__(self, report_id: str, frame_id: int, frame_position: int, frame: dict, path=""):
        self.report_id = report_id
        self.frame_id = frame_id
        self.frame_position = frame_position
        self.method_name = frame["method_name"]
        self.file_name = frame["file_name"]
        self.line_number = frame["line_number"]
        self.path = path
        self.repo_path = "../../master"
        self.change_authors = set()

    def set_path(self, path: str):
        self.path = path

    def fill_path(self):
        pass

    def get_frame(self):
        return self.method_name

    def get_file_name(self):
        return self.file_name

    def get_line_number(self):
        return self.line_number

    def get_path(self):
        return self.path

    def days_since_file_changed(self, commits_hexsha: list):
        pass

    def get_num_people_changed(self):
        pass
