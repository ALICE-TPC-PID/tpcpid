import subprocess
from . import logger

LOG = logger.logger(min_severity="DEBUG", task_name="git")

def git(*args, quiet=False):
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.check_output(
        ["git"] + list(args),
        encoding="utf-8",
        stderr=stderr
    ).strip()

def normalize_remote(url: str):
    # convert git@github.com:user/repo.git → https://github.com/user/repo
    if url.startswith("git@") and ":" in url:
        host = url.split("@", 1)[1].split(":", 1)[0]
        path = url.split(":", 1)[1]
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://{host}/{path}"
    return url

def safe_git_tag():
    tag = git("tag", "--points-at", "HEAD")
    if tag:
        return tag

    try:
        return git("describe", "--tags", "--abbrev=0", quiet=True)
    except subprocess.CalledProcessError:
        return None

def full_git_config(save_to_file=None, verbose=True):
    remote = normalize_remote(git("config", "--get", "remote.origin.url"))
    tag = safe_git_tag() or None
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")

    if verbose:
        LOG.info(f"Remote: {remote}")
        LOG.info(f"Branch: {branch}")
        LOG.info(f"Commit: {commit}")
        LOG.info(f"Tag: {tag if tag else 'N/A'}")

    if save_to_file:
        with open(save_to_file, "w") as f:
            f.write(f"Remote: {remote}\n")
            f.write(f"Branch: {branch}\n")
            f.write(f"Commit: {commit}\n")
            f.write(f"Tag: {tag if tag else 'N/A'}\n")
            if tag:
                f.write("Reproduce:\n")
                f.write(f"git clone {remote}\n")
                f.write(f"git fetch --tags\n")
                f.write(f"git checkout {tag}\n")
            else:
                f.write("Reproduce:\n")
                f.write(f"git clone {remote}\n")
                f.write(f"git fetch origin {branch}\n")
                f.write(f"git checkout {commit}\n")
