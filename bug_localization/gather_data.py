import json
import pandas as pd
import logging
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


if __name__ == "__main__":
    issue_to_report = pd.read_csv("../issue_report_commit_ids.csv")
    issue_to_report.dropna(subset=["commit_hexsha"], inplace=True)
    issue_to_report.reset_index(drop=True, inplace=True)
    df = pd.DataFrame(
        columns=[
            "report_id",
            "file_name",
            "frame",
            "line_number",
            "distance_to_top",
            "language",
            "source",
            "frame_length",
            "exception_type",
        ]
    )
    cnt = 0
    for report_id in issue_to_report["report_id"]:
        cnt += 1
        with open("../reports/" + str(report_id) + ".json", "r") as f:
            try:
                report = json.load(f)
            except UnicodeDecodeError as ex:
                logging.error("Bad source report {} encoding: ".format(report["id"]))
                continue
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
            for frame_position in range(0, len(report["frames"])):
                method_name = report["frames"][frame_position]["method_name"]
                if method_name.startswith("com."):
                    frame = ProjectFrame(
                        report_id, frame_position, report["frames"][frame_position]
                    )
                elif method_name.startswith("java."):
                    frame = SourceFrame(
                        report_id, frame_position, report["frames"][frame_position]
                    )
                file_name = frame.get_file_name()
                df_upd.append(
                    [
                        frame.get_report_id(),  # report_id
                        frame.get_file_name(),  # file_name
                        frame.get_frame(),  # frame
                        frame.get_line_number(),  # line_number
                        frame.get_position(),  # distance_to_top
                        frame.get_language(),  # language
                        frame.get_file_source(),  # source
                        frame.get_frame_length(),  # frame_length
                        get_report_exception_type(report)  # exception_type
                    ]
                )
            df.append(pd.DataFrame(df_upd, columns=df.columns))
        logging.info("Report {} processed".format(report_id))
        if cnt % 1000 == 0:
            print("Amount of elapsed reports: {}".format(cnt))
    df.to_csv("data.csv", index=False)
