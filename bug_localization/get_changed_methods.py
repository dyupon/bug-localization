import logging
import re

import git
from bug_localization.code_file import CodeFile

logging.basicConfig(filename='get_changed_methods.log', filemode='w', format='%(levelname)s: %(message)s', level=logging.INFO)

DIFF_LINES_PATTERN = "\\n@@\s-\d+(?:,\d+)?\s\+(\d+)(?:,)?((?:\d+)?)"


def get_changed_lines(repo: git.repo.base.Repo, commit: git.objects.commit.Commit):
    diff_lines = {}
    for parent_commit in commit.parents:
        diff = repo.git.diff(parent_commit, commit, "--unified=0")
        diff_list = diff.split("diff --git")
        for changed_file, curr_diff in zip(commit.stats.files, diff_list[1:]):
            if changed_file.endswith(".java") or changed_file.endswith(".kt"):
                file_name = changed_file.split("/")[-1]
                curr_diff_pattern = file_name + DIFF_LINES_PATTERN
                match = re.search(curr_diff_pattern, curr_diff)
                while match is not None:
                    changes = re.findall(DIFF_LINES_PATTERN, curr_diff)
                    curr_diff = re.sub(DIFF_LINES_PATTERN, "", curr_diff)
                    for curr_change in changes:
                        start_line = int(curr_change[0])
                        num_lines_changed = int(curr_change[1] or 0)
                        if changed_file in diff_lines:
                            existing_changes = diff_lines[changed_file]
                            diff_lines[changed_file] = [existing_changes, [start_line, start_line + num_lines_changed]]
                        else:
                            diff_lines[changed_file] = [start_line, start_line + num_lines_changed]
                    match = re.search(curr_diff_pattern, curr_diff)
            else:
                continue
        logging.info(
            "get_changed_lines(repo = {}, commit = {}): ".format(repo.git_dir, commit.hexsha) + str(diff_lines))
        return diff_lines


def get_methods(repo: git.repo.base.Repo, commit: git.objects.commit.Commit):
    result = {}
    for changed_file in commit.stats.files:
        file = repo.git.show('{}:{}'.format(commit.hexsha, changed_file))
        if changed_file.endswith(".kt"):
            code_file = CodeFile(file, language="kotlin")
        else:
            code_file = CodeFile(file)
        file_methods = code_file.get_methods_borders()
        for method in file_methods:
            result[method] = file_methods[method]
    return result


def get_changed_methods(repo_path="../../intellij-community",
                        commit_sha="a47e39be87fae50b80c2113858272af3ea9d62a5"):
    repo = git.Repo(repo_path, odbt=git.db.GitDB)
    commit = repo.commit(commit_sha)
    diff_lines = get_changed_lines(repo, commit)
    methods_lines = get_methods(repo, commit)
    return methods_lines


if __name__ == '__main__':
    print(get_changed_methods())
