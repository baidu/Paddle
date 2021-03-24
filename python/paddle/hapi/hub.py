# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import shutil
import zipfile
from paddle.utils.download import get_path_from_url

MASTER_BRANCH = 'main'
DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
HUB_DIR = os.path.expanduser(os.path.join('~', '.cache', 'paddle', 'hub'))


def import_module(name, path):
    import importlib.util
    from importlib.abc import Loader
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    return module


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner,
                                                            repo_name, branch)


def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch


def _get_cache_or_reload(github, force_reload, verbose=True):
    # Setup hub_dir to save downloaded files
    hub_dir = HUB_DIR
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = branch.replace('/', '_')
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(hub_dir,
                            '_'.join([repo_owner, repo_name, normalized_br]))

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbose:
            sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, normalized_br + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)

        get_path_from_url(url, hub_dir, decompress=False)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        # rename the repo
        shutil.move(extracted_repo, repo_dir)

    return repo_dir


def _load_entry_from_hubconf(m, name):
    '''load entry from hubconf
    '''
    if not isinstance(name, str):
        raise ValueError(
            'Invalid input: model should be a str of function name')

    func = getattr(m, name, None)

    if func is None or not callable(func):
        raise RuntimeError('Canot find callable {} in hubconf'.format(name))

    return func


def _check_module_exists(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _check_dependencies(m):
    dependencies = getattr(m, VAR_DEPENDENCY, None)

    if dependencies is not None:
        missing_deps = [
            pkg for pkg in dependencies if not _check_module_exists(pkg)
        ]
        if len(missing_deps):
            raise RuntimeError('Missing dependencies: {}'.format(', '.join(
                missing_deps)))


def list(repo_dir, source='github', force_reload=False):
    r"""
    List all entrypoints available in `github` hubconf.

    Args:
        repo_dir(str): github or local path
            github (str): a str with format "repo_owner/repo_name[:tag_name]" with an optional
                tag/branch. The default branch is `master` if not specified.
            local (str): local repo path
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        ```python
        import paddle

        paddle.hub.help('lyuwenyu/PaddleClas:hub_L', source='github', force_reload=True)

        ```
    """
    if source not in ('github', 'local'):
        raise ValueError(
            'Unknown source: "{}". Allowed values: "github" | "local".'.format(
                source))

    if source == 'github':
        repo_dir = _get_cache_or_reload(repo_dir, force_reload, True)

    sys.path.insert(0, repo_dir)
    hub_module = import_module(MODULE_HUBCONF, repo_dir + '/' + MODULE_HUBCONF)
    sys.path.remove(repo_dir)

    entrypoints = [
        f for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith('_')
    ]

    return entrypoints


def help(repo_dir, model, source='github', force_reload=False):
    """
    Show help information of model

    Args:
        repo_dir(str): github or local path
            github (str): a str with format "repo_owner/repo_name[:tag_name]" with an optional
                tag/branch. The default branch is `master` if not specified.
            local (str): local repo path
        model (str): model name
        source (str): source of repo_dir, (github, local), 
            Default is 'github'
        force_reload (bool, optional):
            Default is `False`
    Return:
        docs

    Example:
        ```python
        import paddle

        paddle.hub.help('lyuwenyu/PaddleClas:hub_L', model='ResNet18Test', source='github')
        ```
    """
    if source not in ('github', 'local'):
        raise ValueError(
            'Unknown source: "{}". Allowed values: "github" | "local".'.format(
                source))

    if source == 'github':
        repo_dir = _get_cache_or_reload(repo_dir, force_reload, True)

    sys.path.insert(0, repo_dir)
    hub_module = import_module(MODULE_HUBCONF, repo_dir + '/' + MODULE_HUBCONF)
    sys.path.remove(repo_dir)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


def load(repo_dir, model, *args, source='github', force_reload=False, **kwargs):
    """
    Load model

    Args:
        repo_dir(str): github or local path
            github (str): a str with format "repo_owner/repo_name[:tag_name]" with an optional
                tag/branch. The default branch is `master` if not specified.
            local (str): local repo path
        mdoel (str): model name
        source (str): github | local
            Default is 'github'
        *args, **kwargs: model parameters
    Return:
        paddle model
    Example:
        ```python
        import paddle
        paddle.hub.load('lyuwenyu/PaddleClas:hub_L', model='ResNet18Test', source='github')
        ```
    """

    if source not in ('github', 'local'):
        raise ValueError(
            'Unknown source: "{}". Allowed values: "github" | "local".'.format(
                source))

    if source == 'github':
        repo_dir = _get_cache_or_reload(repo_dir, force_reload, True)

    sys.path.insert(0, repo_dir)
    hub_module = import_module(MODULE_HUBCONF, repo_dir + '/' + MODULE_HUBCONF)
    sys.path.remove(repo_dir)

    _check_dependencies(hub_module)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry(*args, **kwargs)
