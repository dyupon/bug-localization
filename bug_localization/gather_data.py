import json
import pandas as pd
import logging
import pickle
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


def get_frame(path, ft, report, report_id, frame_position):
    sub_directory = ".".join(path.split(".")[:3])
    frame = None
    for key in ft:
        if ft[key].find(sub_directory):
            frame = ProjectFrame(
                report_id, frame_position, report["frames"][frame_position]
            )
            break
    if not frame:
        frame = SourceFrame(
            report_id, frame_position, report["frames"][frame_position]
        )
    return frame


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
            "exception_type",
            "is_rootcause"
        ]
    )
    cnt = 0
    for report_id in issue_to_report["report_id"]:
        issue_id = issue_to_report.loc[issue_to_report["report_id"] == report_id, "issue_id"].values[0]
        if str(issue_id) in corrupted_issues or str(issue_id) not in issue_to_changed.keys():
            continue
        cnt += 1
        with open("../reports/" + str(report_id) + ".json", "r") as f:
            try:
                report = json.load(f)
            except UnicodeDecodeError as ex:
                logging.error("Bad source report {} encoding: ".format(report["id"]))
                continue
            # remove non unique lines from frame
            report["frames"] = [dict(t) for t in {tuple(d.items()) for d in report["frames"]}]
            commits_hexsha = issue_to_report.loc[
                issue_to_report["report_id"] == report["id"], "commit_hexsha"
            ]
            commits_hexsha = commits_hexsha.values[0].replace("'", "")[1:-1].split(", ")
            if len(report["class"]) > 1:
                logging.error(
                    "Several exceptions for the error report, report_id = {}".format(
                        report_id
                    )
                )
            df_upd = []
            ft = load_file_tree()
            for frame_position in range(0, len(report["frames"])):
                method_name = report["frames"][frame_position]["method_name"]
                frame = get_frame(method_name, ft, report, report_id, frame_position)
                file_name = frame.get_file_name()
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
                        get_report_exception_type(report),  # exception_type
                        get_root_cause_check(issue_to_changed[str(issue_id)], frame.get_frame())  # is_rootcause
                    ]
                )
            if df_upd:
                df = df.append(pd.DataFrame(df_upd, columns=df.columns))
        logging.info("Report {} processed".format(report_id))
        if cnt % 1000 == 0:
            print("Amount of elapsed reports: {}".format(cnt))
    df.to_csv("data.csv", index=False)
