import os
import os.path as osp
import shutil
import time
import datetime

import torch

from ..util.slconfig import SLConfig

class Error(OSError):
    pass

def slcopytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copyfile,
             ignore_dangling_symlinks=False):
    """
    modified from shutil.copytree without copystat.
    
    Recursively copy a directory tree.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    errors = []
    if os.path.isdir(src):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        os.makedirs(dst)
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            try:
                if os.path.islink(srcname):
                    linkto = os.readlink(srcname)
                    if symlinks:
                        # We can't just leave it to `copy_function` because legacy
                        # code with a custom `copy_function` may rely on copytree
                        # doing the right thing.
                        os.symlink(linkto, dstname)
                    else:
                        # ignore dangling symlink if the flag is on
                        if not os.path.exists(linkto) and ignore_dangling_symlinks:
                            continue
                        # otherwise let the copy occurs. copy2 will raise an error
                        if os.path.isdir(srcname):
                            slcopytree(srcname, dstname, symlinks, ignore,
                                    copy_function)
                        else:
                            copy_function(srcname, dstname)
                elif os.path.isdir(srcname):
                    slcopytree(srcname, dstname, symlinks, ignore, copy_function)
                else:
                    # Will raise a SpecialFileError for unsupported file types
                    copy_function(srcname, dstname)
            # catch the Error from the recursive copytree so that we can
            # continue with other files
            except Error as err:
                errors.extend(err.args[0])
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
    else:
        copy_function(src, dst)

    if errors:
        raise Error(errors)
    return dst

def check_and_copy(src_path, tgt_path):
    if os.path.exists(tgt_path):
        return None

    return slcopytree(src_path, tgt_path)


def remove(srcpath):
    if os.path.isdir(srcpath):
        return shutil.rmtree(srcpath)
    else:
        return os.remove(srcpath)  


def preparing_dataset(pathdict, image_set, args):
    start_time = time.time()
    dataset_file = args.dataset_file
    data_static_info = SLConfig.fromfile('util/static_data_path.py')
    static_dict = data_static_info[dataset_file][image_set]

    copyfilelist = []
    for k,tgt_v in pathdict.items():
        if os.path.exists(tgt_v):
            if args.local_rank == 0:
                print("path <{}> exist. remove it!".format(tgt_v))
                remove(tgt_v)
            # continue
        
        if args.local_rank == 0:
            src_v = static_dict[k]
            assert isinstance(src_v, str)
            if src_v.endswith('.zip'):
                # copy
                cp_tgt_dir = os.path.dirname(tgt_v)
                filename = os.path.basename(src_v)
                cp_tgt_path = os.path.join(cp_tgt_dir, filename)
                print('Copy from <{}> to <{}>.'.format(src_v, cp_tgt_path))
                os.makedirs(cp_tgt_dir, exist_ok=True)
                check_and_copy(src_v, cp_tgt_path)          

                # unzip
                import zipfile
                print("Starting unzip <{}>".format(cp_tgt_path))
                with zipfile.ZipFile(cp_tgt_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(cp_tgt_path))      

                copyfilelist.append(cp_tgt_path)
                copyfilelist.append(tgt_v)
            else:
                print('Copy from <{}> to <{}>.'.format(src_v, tgt_v))
                os.makedirs(os.path.dirname(tgt_v), exist_ok=True)
                check_and_copy(src_v, tgt_v)
                copyfilelist.append(tgt_v)
    
    if len(copyfilelist) == 0:
        copyfilelist = None
    args.copyfilelist = copyfilelist
        
    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    if copyfilelist:
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Data copy time {}'.format(total_time_str))
    return copyfilelist


    