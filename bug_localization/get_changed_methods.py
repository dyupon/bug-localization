import json
import logging
import pickle
import re
import csv

import git
from bug_localization.code_file import CodeFile

logging.basicConfig(filename='get_changed_methods.log', filemode='w', format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)

DIFF_LINES_PATTERN = "\\n@@\s-\d+(?:,\d+)?\s\+(\d+)(?:,)?((?:\d+)?)"
ISSUE_NUMBER_PATTERN = "(?<=(?:[^\w])EA-)[\d]+|(?<=^EA-)[\d]+"
CORRUPTED_ISSUES = set()


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
                            existing_changes.append([start_line, start_line + num_lines_changed])
                            diff_lines[changed_file] = existing_changes
                        else:
                            diff_lines[changed_file] = [[start_line, start_line + num_lines_changed]]
                    match = re.search(curr_diff_pattern, curr_diff)
            else:
                continue
        logging.info(
            "get_changed_lines(repo = {}, commit = {}): ".format(repo.git_dir, commit.hexsha) + str(diff_lines))
        return diff_lines


def get_methods(repo: git.repo.base.Repo, commit: git.objects.commit.Commit, issue: str):
    result = {}
    for changed_file in commit.stats.files:
        file = repo.git.show('{}:{}'.format(commit.hexsha, changed_file))
        file_name = changed_file.split("/")[-1]
        if changed_file.endswith(".kt"):
            code_file = CodeFile(file, file_name[:-3], language="kt")
        elif changed_file.endswith(".java"):
            code_file = CodeFile(file, file_name[:-5], file_name)
        else:
            continue
        try:
            file_methods = code_file.get_methods_borders()
            result[changed_file] = file_methods
        except Exception as e:
            CORRUPTED_ISSUES.add(issue)
            logging.error("Failed to parse object " + changed_file + " for commit " + commit.hexsha + ": " + str(e))
    return result


def get_changed_methods(repo: git.repo.base.Repo, commit_sha: str, issue: str):
    commit = repo.commit(commit_sha)
    diff_lines = get_changed_lines(repo, commit)
    methods_lines = get_methods(repo, commit, issue)
    result = []
    for changed_file in diff_lines:
        if changed_file in methods_lines:
            methods = methods_lines[changed_file]
            changed = diff_lines[changed_file]
            for change in changed:
                for method in methods:
                    borders = methods[method]
                    if max([change[0], borders[0]]) <= min([change[1], borders[1]]):
                        result.append(".".join(changed_file.split("/")[:-1]) + "." + method)
    return result


if __name__ == '__main__':
    repo_path = "../../master"
    repo = git.Repo(repo_path, odbt=git.db.GitDB)
    # print(get_changed_methods(repo))
    commits = list(repo.iter_commits("master"))
    issue_to_changed = {}
    issues = set()
    with open('../issue_report_ids.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            issues.add(row[0])
    commits_cnt = 0
    for commit in commits:
        commits_cnt += 1
        commit_msg = commit.message
        issue_match = re.findall(ISSUE_NUMBER_PATTERN, commit_msg)
        if issue_match is not None:
            for issue in issue_match:
                if issue in issues:
                    try:
                        print(commit.hexsha)
                        if issue in issue_to_changed:
                            issue_to_changed[issue] = issue_to_changed[issue] + get_changed_methods(repo,
                                                                                                    commit.hexsha,
                                                                                                    issue)
                        else:
                            issue_to_changed[issue] = get_changed_methods(repo, commit.hexsha, issue)
                        logging.info("Issue number: " + issue + ", commit: " + commit.hexsha)
                    except git.exc.GitCommandError as ex:
                        logging.error("Failed to get changed methods: " + str(ex))
        if commits_cnt % 1000 == 0:
            print("Amount of commits elapsed: " + str(commits_cnt))
    with open("issue_to_changed.json", "w") as mapping_file:
        json.dump(issue_to_changed, mapping_file)
    with open('corrupted_issues.pickle', 'wb') as f:
        pickle.dump(CORRUPTED_ISSUES, f)
