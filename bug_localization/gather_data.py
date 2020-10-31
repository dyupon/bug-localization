import json
import pandas as pd
import logging
import pickle
from bug_localization.project_frame import ProjectFrame
from bug_localization.source_frame import SourceFrame

logging.basicConfig(filename='gather_data.log', filemode='w', format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO)

REPO_PATH = "../../master"

if __name__ == '__main__':
    issue_to_report = pd.read_csv("../issue_report_commit_ids.csv")
    issue_to_report.dropna(subset=["commit_hexsha"], inplace=True)
    issue_to_report.reset_index(drop=True, inplace=True)
    with open('files.pickle', 'rb') as f:
        file_system = pickle.load(f)
    meta_info = {}  # frame_id: "report_id", "exception", "commit_hash"
    df = pd.DataFrame(columns=["frame_id", "file_name", "frame", "line_number",
                               "distance_to_top", "language", "source", "days_since_file_changed",
                               "num_people_changed", "frame_length", "exception_type"],
                      index=["frame_id"])
    frame_id_cnt = 0
    for report_id in issue_to_report["report_id"]:
        with open("../reports/" + str(report_id) + ".json", "r") as f:
            report = json.load(f)
            commits_hexsha = issue_to_report.loc[issue_to_report["report_id"] == report["id"], "commit_hexsha"]
            commits_hexsha = commits_hexsha[0].replace("\'", "")[1:-1].split(", ")
            meta_upd = {i: [report["id"],  # report_id
                            *report["class"],  # exception
                            commits_hexsha  # commit_hash
                            ] for i in
                        range(frame_id_cnt, frame_id_cnt + len(report["frames"]))}
            meta_info.update(meta_upd)
            print(report_id)
            logging.info("report_id: {}".format(report_id))
            if len(report["class"]) > 1:
                logging.error("Several exceptions for the error report, report_id = {}".format(report_id))
            df_upd = []
            for frame_id, frame_position in zip(range(frame_id_cnt, frame_id_cnt + len(report["frames"])),
                                                range(0, len(report["frames"]))):
                method_name = report["frames"][frame_position]["method_name"]
                if method_name.startswith("com."):
                    frame = ProjectFrame(report_id, frame_id, frame_position, report["frames"][frame_position])
                elif method_name.startswith("java."):
                    frame = SourceFrame(report_id, frame_id, frame_position, report["frames"][frame_position])
                file_name = frame.get_file_name()
                if not (file_name.endswith(".java") or file_name.endswith(".kt")):
                    continue
                frame.fill_path()
                print(frame.get_path())
                df_upd.append([
                    frame_id,  # frame_id
                    file_name,  # file_name
                    frame.get_frame(),  # frame
                    frame.get_line_number(),  # line_number
                    frame_position,  # distance_to_top
                    0 if file_name.endswith(".java") else 1,  # language
                    0 if frame.get_frame().find("com.") else 1,  # source
                    frame.days_since_file_changed(commits_hexsha),  # days_since_file_changed
                    frame.get_num_people_changed(),  # num_people_changed
                    len(frame.get_frame()),  # frame_length
                    0 if report["class"][0].startswith("java.") else 1  # exception_type
                ])
            df.append(pd.DataFrame(df_upd, columns=df.columns))
            frame_id_cnt = len(report["frames"])
    df.to_csv("data.csv", index=False)
