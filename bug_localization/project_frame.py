import logging
import git
import os
import datetime
import dateutil.tz
import pickle

from bug_localization.commit import Commit
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

    if os.path.exists('blame_cache.pickle'):
        with open("blame_cache.pickle", "rb") as f:
            BLAME_CACHE = pickle.load(f)
    else:
        BLAME_CACHE = {}

    def __init__(self, report_id: str, frame_position: int, frame: dict, path=""):
        super().__init__(report_id, frame_position, frame, path)
        if self.file_name == "<generated>":
            self.file_name = self.frame.split("$")[0].split(".")[-1]
            if self.file_name + ".java" in self.file_system:
                self.file_name += ".java"
            elif self.file_name + ".kt" in self.file_system:
                self.file_name += ".kt"

    def fill_path(self):
        result_list = []
        if self.file_name in self.file_system:
            result_list.append(
                os.path.join(self.file_system[self.file_name], self.file_name)
            )
        intersect_min = float('inf')
        result = ""
        if len(result_list) > 1:
            for res in result_list:
                intersect = set(res.split("//")).intersection(set(self.frame.split(".")))
                if len(intersect) < intersect_min:
                    intersect_min = len(intersect)
                    result = res
            self.path = result[result.find("\\") + 1:].replace("\\", "/")
        elif not result_list:
            print(self.file_name)
            print(self.frame)
            print(self.report_id)
            self.path = ""
        else:
            self.path = result_list[0][result_list[0].find("\\") + 1:].replace("\\", "/")

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
        affecting_commits = set()
        if self.frame in self.BLAME_CACHE:
            affecting_commits = self.BLAME_CACHE[self.frame]
        else:
            try:
                for blame_entry in repo.blame("HEAD", self.path, "-w -M -C"):
                    affecting_commit = Commit(repo.commit(blame_entry[0]))
                    affecting_commits.add(affecting_commit)
                affecting_commits = list(affecting_commits)
                affecting_commits.sort(key=lambda x: x.authored_datetime, reverse=False)
                self.BLAME_CACHE[self.frame] = affecting_commits
            except git.exc.GitCommandError:
                return date_diff
            finally:
                with open("blame_cache.pickle", "wb") as bc:
                    proxy = dict(self.BLAME_CACHE)
                    pickle.dump(proxy, bc)
        for commit in affecting_commits:
            if fix.authored_datetime < commit.authored_datetime:
                date_diff = (fix.authored_datetime - commit.authored_datetime).days
                break
            else:
                email = commit.author_email.lower()
                self.change_authors.add(email.split("@")[0])
        return date_diff

    def get_num_people_changed(self):
        return len(self.change_authors)
