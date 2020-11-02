from collections import deque


def get_matching_bracket(string, open_idx):
    if string[open_idx] != "{":
        return -1
    d = deque()
    for i in range(open_idx, len(string)):
        if string[i] == "}":
            d.popleft()
        elif string[i] == "{":
            d.append(string[open_idx])
        if not d:
            return i
    return -1
