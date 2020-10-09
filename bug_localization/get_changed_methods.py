import logging
import re
from collections import deque

import git

logging.basicConfig(filename='app.log', filemode='w', format='%(levelname)s: %(message)s', level=logging.INFO)

DIFF_LINES_PATTERN = "\\n@@\s-\d+(?:,\d+)?\s\+(\d+)(?:,)?((?:\d+)?)"


def get_matching_bracket(s, i):
    if s[i] != '{':
        return -1
    d = deque()
    for k in range(i, len(s)):
        if s[k] == '}':
            d.popleft()
        elif s[k] == '{':
            d.append(s[i])
        if not d:
            return k
    return -1


def get_changed_lines(repo: git.repo.base.Repo, commit: git.objects.commit.Commit):
    diff_lines = {}
    for parent_commit in commit.parents:
        diff = repo.git.diff(parent_commit, commit, "--unified=0")
        diff_list = diff.split("diff --git")
        for changed_file, curr_diff in zip(commit.stats.files, diff_list[1:]):
            if changed_file.endswith(".java") or changed_file.endswith(".kt"):
                classname = changed_file.split("/")[-1]
                curr_diff_pattern = classname + DIFF_LINES_PATTERN
                match = re.search(curr_diff_pattern, curr_diff)
                while match is not None:
                    changes = re.findall(DIFF_LINES_PATTERN, curr_diff)
                    curr_diff = re.sub(DIFF_LINES_PATTERN, "", curr_diff)
                    for curr_change in changes:
                        start_line = int(curr_change[0])
                        num_lines_changed = int(curr_change[1] or 0)
                        if classname in diff_lines:
                            existing_changes = diff_lines[classname]
                            diff_lines[classname] = [existing_changes, [start_line, start_line + num_lines_changed]]
                        else:
                            diff_lines[classname] = [start_line, start_line + num_lines_changed]
                    match = re.search(curr_diff_pattern, curr_diff)
            else:
                continue
        logging.info(
            "get_changed_lines(repo = {}, commit = {}): ".format(repo.git_dir, commit.hexsha) + str(diff_lines))
        return diff_lines


def get_methods_borders(repo: git.repo.base.Repo, commit: git.objects.commit.Commit):
    result = {}
    for changed_file in commit.stats.files:
        if not changed_file.endswith(".java") or changed_file.endswith(".kt"):
            continue
        file = repo.git.show('{}:{}'.format(commit.hexsha, changed_file))
        s_to_linenumber = {}
        num_of_lines = 0
        num_of_symbols = 0
        for s in range(len(file)):
            if file[s] == '\n':
                num_of_lines += 1
            else:
                s_to_linenumber[num_of_symbols] = num_of_lines
                num_of_symbols += 1
        if changed_file.endswith(".java"):
            file = file.replace("\n", "")
            main_obj_o = file.find("{")
            main_obj_e = get_matching_bracket(file, main_obj_o)
            inner_obj = file[main_obj_o + 1:main_obj_e]
            inner_obj_list = list(inner_obj)
            lambda_idx = inner_obj.find("-> {")
            while lambda_idx > -1:
                lambda_clos_idx = get_matching_bracket(inner_obj, lambda_idx + 3)
                inner_obj_list[lambda_idx + 3] = "["
                inner_obj_list[lambda_clos_idx] = "]"
                inner_obj = "".join(inner_obj_list)
                lambda_idx = inner_obj.find("-> {")
            method_idx = inner_obj.find("{")
            while method_idx > -1:
                method_clos_idx = get_matching_bracket(inner_obj, method_idx + 3)
                inner_obj_list[method_idx + 3] = "["
                inner_obj_list[method_clos_idx] = "]"
                start_line = s_to_linenumber.get(method_idx + main_obj_o)
                # TODO: fetch method name from the start_line, append to the dict(result)
                method_idx = inner_obj.find("{")
        return result


def get_changed_methods(repo_path="../../intellij-community",
                        commit_sha="a47e39be87fae50b80c2113858272af3ea9d62a5"):
    repo = git.Repo(repo_path, odbt=git.db.GitDB)
    commit = repo.commit(commit_sha)
    diff_lines = get_changed_lines(repo, commit)
    # method_lines = get_methods_borders(repo, commit)
    return diff_lines


if __name__ == '__main__':
    print(get_changed_methods())
