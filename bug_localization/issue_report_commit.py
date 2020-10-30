import git
import re
import pandas as pd
import numpy as np

ISSUE_NUMBER_PATTERN = "(?<=(?:[^\w])EA-)[\d]+|(?<=^EA-)[\d]+"

if __name__ == '__main__':
    repo = git.Repo("../../master", odbt=git.db.GitDB)
    commits = list(repo.iter_commits("master"))
    mapping = pd.read_csv("../issue_report_ids.csv", dtype={"issue_id": str, "report_id": str})
    mapping["commit_hexsha"] = ""
    issues = set(pd.read_csv("../issue_report_ids.csv", dtype={"issue_id": str, "report_id": str})['issue_id'])
    commits_cnt = 0
    for commit in commits:
        commits_cnt += 1
        commit_msg = commit.message
        issue_match = re.findall(ISSUE_NUMBER_PATTERN, commit_msg)
        if issue_match is not None:
            for issue in issue_match:
                if issue in issues:
                    idx = np.where(mapping["issue_id"] == issue)[0][0]
                    curr_hexsha = mapping.at[idx, "commit_hexsha"]
                    if curr_hexsha != "":
                        curr_hexsha.append(commit.hexsha)
                        mapping.at[idx, "commit_hexsha"] = curr_hexsha
                        print(commit.hexsha)
                    else:
                        mapping.at[idx, "commit_hexsha"] = [commit.hexsha]
                        print(commit.hexsha)
        if commits_cnt % 1000 == 0:
            print("Amount of commits elapsed: " + str(commits_cnt))
    mapping.to_csv("../issue_report_commit_ids.csv", index=False)
