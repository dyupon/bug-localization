from bug_localization.frame import Frame


class SourceFrame(Frame):
    def __init__(
            self, report_id: str, frame_position: int, frame: dict, path=""
    ):
        super().__init__(report_id, frame_position, frame, path)

    def days_since_file_changed(self, commits_hexsha: list):
        return -100500

    def get_num_people_changed(self):
        return 0
