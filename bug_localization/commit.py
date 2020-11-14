import git


class Commit:
    """
    Simple wrapper for git.objects.commit.Commit which can be pickled
    """

    def __init__(self, commit: git.objects.commit.Commit):
        self.authored_datetime = commit.authored_datetime
        self.hexsha = commit.hexsha
        self.author_email = commit.author.email
        self.author_name = commit.author.name
