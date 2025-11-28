import subprocess
from . import logger

LOG = logger.logger("Git")

def git(*args):
    return subprocess.check_output(["git"] + list(args), encoding="utf-8").strip()

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
    # try tag exactly on HEAD first (cheap and no error)
    tag = git("tag", "--points-at", "HEAD")
    if tag:
        return tag

    # try nearest tag, but catch the case where no tags exist
    try:
        return git("describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        return None

def full_git_config(save_to_file : str, verbose=True):
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")
    remote = normalize_remote(git("config", "--get", "remote.origin.url"))
    tag = safe_git_tag() or "N/A"

    if verbose:
        LOG.info(f"Branch: {branch}")
        LOG.info(f"Commit: {commit}")
        LOG.info(f"Remote: {remote}")
        LOG.info(f"Tag: {tag}")
    if save_to_file:
        with open(save_to_file, "w") as f:
            f.write(f"Branch: {branch}\n")
            f.write(f"Commit: {commit}\n")
            f.write(f"Remote: {remote}\n")
            f.write(f"Tag: {tag}\n")