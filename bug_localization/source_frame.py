from bug_localization.frame import Frame


class SourceFrame(Frame):
    def __init__(
            self, report_id: str, frame_position: int, frame: dict, repo_path=""
    ):
        super().__init__(report_id, frame_position, frame, repo_path)

    def get_num_days_since_file_changed(self, commits_hexsha: list):
        return -100500

    def get_num_people_changed(self):
        return 0

    def get_file_source(self):
        return 0

    def get_file_length(self):
        return 0

    def get_method_length(self):
        return 0

    def get_num_of_args(self):
        return 0

    def get_num_file_lines(self):
        return 0
