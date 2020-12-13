import logging
import re

import git
import datetime
import dateutil.tz
import pickle

from bug_localization.frame import Frame
from bug_localization.utils import get_matching_bracket

logging.basicConfig(
    filename="gather_data.log",
    filemode="w",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
REPO_PATH = "../../master"


class ProjectFrame(Frame):
    """
    Represents single frame from the error report
    """

    with open("files.pickle", "rb") as f:
        file_system = pickle.load(f)

    with open("blame_cache.pickle", "rb") as f:
        BLAME_CACHE = pickle.load(f)

    def __init__(self, report_id: str, frame_position: int, frame: dict, repo_path=""):
        super().__init__(report_id, frame_position, frame, repo_path)
        if self.file_name == "<generated>":
            self.file_name = self.frame.split("$")[0].split(".")[-1]
            if self.file_name + ".java" in self.file_system:
                self.file_name += ".java"
            elif self.file_name + ".kt" in self.file_system:
                self.file_name += ".kt"
            elif self.file_name + ".scala" in self.file_system:
                self.file_name += ".scala"
        self.file = None
        self.method_open_idx = None
        self.method_clos_idx = None

    def format_fs(self):
        for file in self.file_system:
            self.file_system[file] = self.file_system[file].replace("\\", "/")

    def get_num_days_since_file_changed(self, commits_hexsha: list):
        repo = git.Repo(REPO_PATH, odbt=git.db.GitDB)
        min_datetime = datetime.datetime(
            3018, 10, 30, tzinfo=dateutil.tz.tzoffset("UTC", +10800)
        )
        fix = None
        for commit_hexsha in commits_hexsha:
            commit = repo.commit(commit_hexsha)
            if commit.authored_datetime < min_datetime:
                min_datetime = commit.authored_datetime
                fix = repo.commit(commit_hexsha)
        date_diff = -1
        if self.frame in self.BLAME_CACHE:
            affecting_commits = self.BLAME_CACHE[self.frame]
        else:
            return date_diff
        for i, commit in enumerate(affecting_commits):
            if fix.authored_datetime < commit.authored_datetime:
                date_diff = (fix.authored_datetime - affecting_commits[i - 1].authored_datetime).days
                break
            else:
                email = commit.author_email.lower()
                self.change_authors.add(email.split("@")[0])
        return date_diff

    def get_num_people_changed(self):
        return len(self.change_authors)

    def get_file_length(self):
        if self.file_name not in self.file_system:
            return 0
        path = "../" + self.file_system[self.file_name] + "/" + self.file_name
        with open(path, "r") as f:
            try:
                file = f.read()
                self.file = file
            except UnicodeDecodeError:
                return None
            return len(file)

    def get_method_length(self):
        if not self.file:
            return None
        clean_frame = re.sub("\d", "", self.frame)
        clean_frame = clean_frame.replace("$", ".")
        splitted_frame = clean_frame.split(".")
        if splitted_frame[-1] == "<init>":
            method_name = splitted_frame[-2]
        else:
            splitted_frame = [x for x in splitted_frame if x != ""]
            method_name = splitted_frame[-1]
        if not self.line_number:
            method_pos = self.file.find(method_name)
            if method_pos == -1:
                return None
            method_open_idx = self.file[method_pos:].find("{")
            method_open_idx += method_pos
            self.method_open_idx = method_open_idx
            method_clos_idx = get_matching_bracket(self.file, method_open_idx)
            self.method_clos_idx = method_clos_idx
            return method_clos_idx - method_open_idx
        else:
            line_to_code = self.file.split("\n")
            for i, line in reversed(list(enumerate(line_to_code[:self.line_number]))):
                method_pos = line.find(method_name)
                if method_pos == -1:
                    continue
                if (method_pos > 0 and line[method_pos - 1] != " ") or line_to_code[i-1].find(";") > -1:
                    continue
                method_open_idx = "\n".join(line_to_code[i:self.line_number]).find("{") + 1
                method_open_idx += len("\n".join(line_to_code[:i]))
                self.method_open_idx = method_open_idx
                method_clos_idx = get_matching_bracket(self.file, method_open_idx)
                self.method_clos_idx = method_clos_idx
                if method_clos_idx - method_open_idx < 0:
                    return None
                return method_clos_idx - method_open_idx
        return 0

    def get_file_source(self):
        return 1

    def get_num_of_args(self):
        if not self.method_open_idx or not self.method_clos_idx or not self.file:
            return 0
        reverse_part = self.file[:self.method_open_idx][::-1]
        args_open_bracket_idx = reverse_part.find("(")
        args_clos_bracket_idx = reverse_part.find(")")
        return len(reverse_part[args_clos_bracket_idx:args_open_bracket_idx].split(","))

    def get_num_file_lines(self):
        if not self.file:
            return 0
        return len(self.file.split("\n"))
