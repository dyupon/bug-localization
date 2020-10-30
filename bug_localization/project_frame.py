import logging
import git
import os
import datetime
import dateutil.tz

from bug_localization.frame import Frame

logging.basicConfig(filename='gather_data.log', filemode='w', format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)


class ProjectFrame(Frame):
    """
    Represents single frame from the error report
    """

    def __init__(self, report_id: str, frame_id: int, frame_position: int, frame: dict, path=""):
        super().__init__(report_id, frame_id, frame_position, frame, path)

    def fill_path(self):
        result = []
        for root, dir, files in os.walk(self.repo_path):
            if self.file_name in files:
                result.append(os.path.join(root, self.file_name))
        if len(result) > 1:
            raise Exception("We have two different files with the same name and somehow need to handle it: "
                            "report_id = {}, file_name = {}".format(self.report_id, self.file_name))
        if self.method_name.startswith("java.") and len(result) == 0:
            pass
        self.path = result[0][result[0].find("\\") + 1:].replace("\\", "/")

    def days_since_file_changed(self, commits_hexsha: list):
        repo = git.Repo(self.repo_path, odbt=git.db.GitDB)
        min_datetime = datetime.datetime(3018, 10, 30, tzinfo=dateutil.tz.tzoffset('UTC', +10800))
        fix = None
        for commit_hexsha in commits_hexsha:
            commit = repo.commit(commit_hexsha)
            if commit.authored_datetime < min_datetime:
                min_datetime = commit.authored_datetime
                fix = repo.commit(commit_hexsha)
        path = self.path
        affecting_commits = repo.iter_commits(rev="master", paths=path)
        date_diff = -1
        for commit in affecting_commits:
            if fix.authored_datetime > commit.authored_datetime:
                date_diff = (fix.authored_datetime - commit.authored_datetime).days
                break
            else:
                self.change_authors.add(commit.author)
        return date_diff

    def get_num_people_changed(self):
        if len(self.change_authors) == 0:
            raise Exception("days_since_file_changed() should be called first")
        return len(self.change_authors)


