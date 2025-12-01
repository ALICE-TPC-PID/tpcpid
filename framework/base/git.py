import os, subprocess
from . import logger

LOG = logger.logger(min_severity="DEBUG", task_name="git")

def git(*args, quiet=False, path=None):
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.check_output(
        ["git"] + list(args),
        encoding="utf-8",
        stderr=stderr,
        cwd=path  # <-- run git in the specified folder
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

def safe_git_tag(path=None):
    tag = git("tag", "--points-at", "HEAD", path=path)
    if tag:
        return tag

    try:
        return git("describe", "--tags", "--abbrev=0", quiet=True, path=path)
    except subprocess.CalledProcessError:
        return None

def full_git_config(save_to_file=None, verbose=True, path=None):
    remote = normalize_remote(git("config", "--get", "remote.origin.url", path=path))
    tag = safe_git_tag(path=path) or None
    branch = git("rev-parse", "--abbrev-ref", "HEAD", path=path)
    commit = git("rev-parse", "HEAD", path=path)

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
            f.write("Reproduce:\n")
            if tag:
                f.write(f"  git clone {remote}\n")
                f.write(f"  git fetch --tags\n")
                f.write(f"  git checkout {tag}\n")
            else:
                f.write(f"  git clone {remote}\n")
                f.write(f"  git fetch origin {branch}\n")
                f.write(f"  git checkout {commit}\n")
                
def checkout_from_config(git_config: dict, path: str = None):
    """
    Given a git config dictionary:
        {
            "remote": "...",
            "branch": "...",
            "commit": "latest" or SHA,
            "tag": "N/A" or tagname
        }
    clone + checkout the correct state.
    Returns git_config with commit replaced by the resolved hash.
    """

    remote = git_config.get("remote", "https://github.com/ALICE-TPC-PID/tpcpid.git")
    branch = git_config.get("branch", "main")
    commit = git_config.get("commit", "latest")
    tag = git_config.get("tag", "N/A")

    # If no path specified: create temporary directory
    created_temp = False
    if path is None:
        path = tempfile.mkdtemp(prefix="gitrepo_")
        created_temp = True

    LOG.info(f"Using repo path: {path}")

    # Clone repo if empty folder
    if not os.listdir(path):
        LOG.info(f"Cloning {remote} ...")
        subprocess.check_call(["git", "clone", remote, path])

    # Always ensure repo is up-to-date
    subprocess.check_call(["git", "fetch", "--all", "--tags"], cwd=path)

    if tag != "N/A":
        LOG.info(f"Checking out tag {tag}")
        subprocess.check_call(["git", "checkout", tag], cwd=path)

    else:
        if commit == "latest":
            LOG.info(f"Checking out latest commit on {branch}")
            subprocess.check_call(["git", "checkout", branch], cwd=path)
            subprocess.check_call(["git", "pull", "origin", branch], cwd=path)

        else:
            LOG.info(f"Checking out commit {commit}")
            subprocess.check_call(["git", "checkout", commit], cwd=path)

    # Read actual commit hash
    resolved_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        encoding="utf-8"
    ).strip()

    LOG.info(f"Resolved commit: {resolved_hash}")

    # Return updated config
    git_config["commit"] = resolved_hash
    return git_config
