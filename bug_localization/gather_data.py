import json
import pandas as pd
import logging
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm

from bug_localization.project_frame import ProjectFrame
from bug_localization.source_frame import SourceFrame

logging.basicConfig(
    filename="gather_data.log",
    filemode="w",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

REPO_PATH = "../../master"


def get_report_exception_type(report):
    if not report["class"] or report["class"] is None:
        return None
    return 0 if report["class"][0].startswith("java.") else 1


def get_report_num_frames(report):
    return len(report["frames"])


def get_root_cause_check(changed_methods, frame):
    for changed_method in changed_methods:
        if changed_method.find(frame) > -1:
            return 1
    return 0


def load_file_tree():
    with open("files.pickle", "rb") as fs:
        file_system = pickle.load(fs)
    for key in file_system:
        file_system[key] = file_system[key].replace("\\", ".")
    return file_system


def get_frame(method_name, file_name, ft, report, report_id, frame_position):
    sub_directory = ".".join(method_name.split(".")[:5])
    frame = None
    for key in ft:
        if ft[key].find(sub_directory) > -1 and (ft.get(file_name, False) or file_name == "<generated>"):
            frame = ProjectFrame(
                report_id, frame_position, report["frames"][frame_position]
            )
            break
    if not frame:
        frame = SourceFrame(
            report_id, frame_position, report["frames"][frame_position]
        )
    return frame


def process_report(report_id):
    issue_id = issue_to_report.loc[issue_to_report["report_id"] == report_id, "issue_id"].values[0]
    if str(issue_id) in corrupted_issues or str(issue_id) not in issue_to_changed.keys():
        return None
    with open("../reports/" + str(report_id) + ".json", "r") as f:
        try:
            report = json.load(f)
        except UnicodeDecodeError as ex:
            logging.error("Bad source report {} encoding: ".format(report_id))
            return None
        commits_hexsha = issue_to_report.loc[
            issue_to_report["report_id"] == report["id"], "commit_hexsha"
        ]
        commits_hexsha = commits_hexsha.values[0].replace("'", "")[1:-1].split(", ")
        df_upd = []
        for frame_position in range(0, len(report["frames"])):
            method_name = report["frames"][frame_position]["method_name"]
            frame = get_frame(method_name, report["frames"][frame_position]["file_name"], ft,
                              report, report_id, frame_position)
            frame.format_fs()
            df_upd.append(
                [
                    report["timestamp"],  # timestamp
                    frame.get_report_id(),  # report_id
                    issue_id,  # issue_id
                    frame.get_file_name(),  # file_name
                    frame.get_frame(),  # frame
                    frame.get_line_number(),  # line_number
                    frame.get_position(),  # distance_to_top
                    frame.get_language(),  # language
                    frame.get_file_source(),  # source
                    frame.get_frame_length(),  # frame_length
                    frame.get_num_days_since_file_changed(commits_hexsha),  # days_since_file_changed
                    frame.get_num_people_changed(),  # num_people_changed
                    get_report_exception_type(report),  # exception_type
                    frame.get_file_length(),  # file_length
                    get_root_cause_check(issue_to_changed[str(issue_id)], frame.get_frame())  # is_rootcause
                ]
            )
        return df_upd


if __name__ == "__main__":
    issue_to_report = pd.read_csv("issue_report_commit_ids.csv")
    issue_to_report.dropna(subset=["commit_hexsha"], inplace=True)
    issue_to_report.reset_index(drop=True, inplace=True)

    with open("issue_to_changed_formatted.json", "r") as f:
        issue_to_changed = json.load(f)

    with open("corrupted_issues.pickle", "rb") as f:
        corrupted_issues = pickle.load(f)

    df = pd.DataFrame(
        columns=[
            "timestamp",
            "report_id",
            "issue_id",
            "file_name",
            "frame",
            "line_number",
            "distance_to_top",
            "language",
            "source",
            "frame_length",
            "days_since_file_changed",
            "num_people_changed",
            "exception_type",
            "file_length",
            "is_rootcause"
        ]
    )
    ft = load_file_tree()
    res = Parallel(n_jobs=8)(
        delayed(process_report)(r)
        for r in tqdm(issue_to_report["report_id"], desc="Progress")
    )
    with open("backup.pickle", "wb") as b:
        pickle.dump(res, b)

    for d in res:
        if d:
            df = df.append(pd.DataFrame(d, columns=df.columns))

    df.to_csv("data.csv", index=False)
