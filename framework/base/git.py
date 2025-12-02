import os, subprocess
from . import logger

LOG = logger.logger(min_severity="DEBUG", task_name="git")

def fetch_upstream(remote: str, path: str):
    try:
        subprocess.check_call(
            ["git", "fetch", remote],
            cwd=path
        )
    except subprocess.CalledProcessError as e:
        LOG.error(f"Failed to fetch from remote '{remote}' in path '{path}': {e}. Continuing for now. PLEASE CHECK YOUR GIT STATUS!")

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
    else:
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
            f.write("\nReproduce:\n")
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

def diff_to_latest_upstream_tag(path=None, diff_file=None, info_file=None):
    """
    Create a diff between the current working tree and the latest upstream tag.
    Optionally writes diff and the tag used into separate files.

    Returns a tuple: (diff_str, tag_str)
    """

    if path is None:
        path = os.getcwd()

    LOG.info(f"Generating diff in repo: {path}")

    # Ensure repo exists
    if not os.path.isdir(os.path.join(path, ".git")):
        raise RuntimeError(f"Not a git repository: {path}")

    repo_url = normalize_remote(git("config", "--get", "remote.origin.url", path=path))
    LOG.info(f"Repository URL: {repo_url}")

    # Fetch all tags from upstream
    LOG.info("Fetching tags from upstream ...")
    subprocess.check_call(["git", "fetch", "--tags"], cwd=path)

    # Resolve latest tag
    try:
        latest_tag = git("tag", "--sort=-creatordate", path=path).splitlines()[0]
    except subprocess.CalledProcessError:
        raise RuntimeError("Repository has no tags available.")

    LOG.info(f"Latest upstream tag: {latest_tag}")

    # Create diff (uses .gitignore automatically)
    diff = git("diff", latest_tag, path=path)

    # Write diff file if requested
    if diff_file:
        with open(diff_file, "w") as f:
            f.write(diff)
        LOG.info(f"Diff written to: {diff_file}")

    # Write tag file if requested
    if info_file:
        with open(info_file, "r") as f:
            git_info = f.readlines()
        with open(info_file, "w") as f:
            f.writelines(git_info)
            f.write(f"\n===============================================================\n")
            f.write(f"\nGit diff mode was enabled. Printing information to which the diff was made:\n")
            f.write(f"Remote URL: {repo_url}\n")
            f.write(f"Remote tag: {latest_tag}\n")
            f.write(f"Diff file: {diff_file}\n")
            f.write("\n------\n")
            f.write("\nTo reproduce the state with this diff:\n")
            f.write(f"  git clone {repo_url}\n")
            f.write(f"  git fetch --tags\n")
            f.write(f"  git checkout {latest_tag}\n")
            f.write(f"  git apply {diff_file}\n")
        LOG.info(f"Diff information written to: {info_file}")

    return diff, repo_url, latest_tag
