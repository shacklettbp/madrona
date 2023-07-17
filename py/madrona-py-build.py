import sys
import zipfile
import hashlib
import csv
import time
import io
import stat
import base64
import copy
import importlib.machinery
from pathlib import Path
from email.message import Message as EmailMessage
from email.policy import EmailPolicy

def _read_toml():
    import tomli

    with open("pyproject.toml", "rb") as f:
        toml = tomli.load(f)

    name = toml['project']['name']

    try:
        version = toml['project']['version']
    except KeyError:
        raise RuntimeError("Need to specify package version in pyproject.toml")

    if '-' in name or '.' in name or '/' in name or '\\' in name:
        raise RuntimeError("Invalid project name")

    return name, version, toml

def _encode_wheel_metadata():
    policy = EmailPolicy(max_line_length=0, mangle_from_=False, utf8=True)
    msg = EmailMessage(policy=policy)

    msg["Wheel-Version"] = '1.0'
    msg["Generator"] = 'madrona-py-build'
    msg["Root-Is-Purelib"] = 'false'

    return msg.as_bytes()

def _encode_pkg_metadata(toml):
    import pyproject_metadata
    parsed = pyproject_metadata.StandardMetadata.from_pyproject(toml)
    return bytes(parsed.as_rfc822())

def _get_wheel_name(versioned_name):
    import packaging.tags

    # sys_tags has the most interpreter specific tag first. For an editable
    # install that's all that is needed
    tag = str(next(packaging.tags.sys_tags()))
    return f"{versioned_name}-{tag}.whl"

def _add_wheel_member(wheel_file, name, data, timestamp):
    member_info = zipfile.ZipInfo(name, date_time=timestamp)
    member_info.compress_type = zipfile.ZIP_DEFLATED
    member_info.external_attr = (0o664 | stat.S_IFREG) << 16
    wheel_file.writestr(member_info, data)

def _write_wheel(wheel_path, record_member, contents):
    timestamp = time.gmtime(time.time())[0:6]

    wheel_file = zipfile.ZipFile(wheel_path, "w",
            compression=zipfile.ZIP_DEFLATED)

    record_data = io.StringIO()
    record_writer = csv.writer(record_data, delimiter=',',
                               quotechar='"', lineterminator='\n')

    for k, v in contents.items():
        _add_wheel_member(wheel_file, k, v, timestamp)
        member_hash = hashlib.sha256(v).digest()
        hash_encoded = base64.urlsafe_b64encode(member_hash).rstrip(b'=').decode('ascii')
        record_writer.writerow((k, f"sha256={hash_encoded}", len(v)))

    record_writer.writerow((record_member, "", ""))
    _add_wheel_member(wheel_file, record_member, record_data.getvalue().encode("utf-8"), timestamp)

    wheel_file.close()

def _build_redir_pth(name):
    return f"import _{name}_redir".encode('utf-8')

def _merge_toml_and_cfg_settings(toml, config_settings):
    try:
        cfg = copy.deepcopy(toml['tool']['madrona'])
    except KeyError:
        cfg = {}

    if config_settings == None:
        config_settings = {}

    for k, v in config_settings.items():
        parts = k.split('.')

        sub_cfg = cfg
        for p in parts[:-1]:
            sub_cfg = sub_cfg[p]

        sub_cfg[parts[-1]] = v

    return cfg

def _build_redir_py(toml, config_settings):
    cfg = _merge_toml_and_cfg_settings(toml, config_settings)

    try:
        pkgs = cfg['packages']

        if len(pkgs) == 0:
            raise KeyError()
    except KeyError:
        raise RuntimeError("pyproject.toml doesn't specify any packages in [tool.madrona.packages]")

    redir = (Path(__file__).parent / "_redir_template.py").read_text()

    redir += "setup_redirection({\n"

    # FIXME, should be a bit more general here than just assuming it is
    # the interpreter specific module
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    for pkg, pkg_cfg in pkgs.items():
        ext_only = pkg_cfg.get('ext-only', False)

        try:
            exts = pkg_cfg['extensions']
        except KeyError:
            exts = []

        try:
            ext_out_dir = pkg_cfg['ext-out-dir']
            ext_out_dir = Path(ext_out_dir).resolve()

        except KeyError:
            if len(exts) > 0 or ext_only:
                raise RuntimeError(f"Package '{pkg}' has extensions but [tool.madrona.packages.{pkg}.ext-out-dir] is not defined for editable install")

        if ext_only:
            ext_path = ext_out_dir / f"{pkg}{ext_suffix}"
            redir += f"    '{pkg}': r'{ext_path}',\n"
        else:
            pkg_init_path = Path(pkg_cfg['path']).resolve() / '__init__.py'
            redir += f"    '{pkg}': r'{pkg_init_path}',\n"

            for ext in exts:
                ext_path = ext_out_dir / f"{ext}{ext_suffix}"
                redir += f"    '{pkg}.{ext}': r'{ext_path}',\n"
    
    redir += "})\n"

    return redir.encode('utf-8')

def prepare_metadata_for_build_editable(metadata_directory, config_settings):
    name, version, toml = _read_toml()

    dist_info_dir = Path(metadata_directory) / f"{name}-{version}.dist-info"
    dist_info_dir.mkdir(parents=True)

    encoded_metadata = _encode_pkg_metadata(toml)

    (dist_info_dir / "METADATA").write_bytes(encoded_metadata)
    (dist_info_dir / "WHEEL").write_bytes(_encode_wheel_metadata())

    return str(dist_info_dir)

def build_editable(wheel_directory, config_settings=None,
                   metadata_directory=None):
    name, version, toml = _read_toml()

    versioned_name = f"{name}-{version}"
    wheel_name = _get_wheel_name(versioned_name)
    wheel_path = Path(wheel_directory) / wheel_name

    dist_info_name = f"{versioned_name}.dist-info"

    record_name = f"{dist_info_name}/RECORD"

    if not wheel_path.parent.exists():
        self.wheelpath.parent.mkdir(parents=True)

    _write_wheel(wheel_path, record_name, {
        f"_{name}_redir.py" : _build_redir_py(toml, config_settings),
        f"_{name}.pth" : _build_redir_pth(name),
        f"{dist_info_name}/METADATA": _encode_pkg_metadata(toml),
        f"{dist_info_name}/WHEEL": _encode_wheel_metadata(),
    })

    return wheel_name

def get_requires_for_build_editable(config_settings = None):
    return ["scikit-build-core>=0.4.7", "tomli>=2.0.1", "pyproject_metadata>=0.7.1", "packaging>=23.1"]
