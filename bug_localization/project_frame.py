import logging
import git
import datetime
import dateutil.tz
import pickle

from bug_localization.frame import Frame

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
            return -1
        path = "../" + self.file_system[self.file_name] + "/" + self.file_name
        with open(path, "r") as file:
            try:
                data = file.read()
            except UnicodeDecodeError:
                return -1
            return len(data)

    def get_method_length(self):
        pass

    def get_file_source(self):
        return 1

