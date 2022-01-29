# -*- coding=utf-8 -*-
import os
import yaml
import argparse
import socket
from io import BytesIO
import multiprocessing
from functools import partial
from easydict import EasyDict
from glob import glob

import numpy as np
from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn

from env_config import LMDB_ROOTS


Image.MAX_IMAGE_PIXELS = np.inf

def resize_and_convert(img, size, resample, quality=100, return_img=False):
    # # Already done in downloading script
    img = trans_fn.center_crop(img, size)
    img = trans_fn.resize(img, size, resample)
    if return_img:
        return img
    else:
        buffer = BytesIO()
        #img.save(buffer, format='jpeg', quality=quality)
        img.save(buffer, format='png')
        val = buffer.getvalue()
        #img.seek(0)
        #val = img.read()
        return val


def resize_multiple(img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


global_semaphore = None
def resize_worker(img_file, size, resample, return_img=False):
    i, file = img_file
    global_semaphore.acquire()
    try:
        img = Image.open(file)
        img = img.convert('RGB')
        #out = resize_multiple(img, sizes=sizes, resample=resample)
        out = resize_and_convert(img, size, resample, quality=100, return_img=return_img)
    except Exception as e:
        print(" Exception while processing {}; Message: {}".format(file, e))
        raise e

    return i, out


def worker_init(semaphore):
    global global_semaphore
    global_semaphore = semaphore


def prepare_img(output_dir, img_paths, n_worker, size=128, resample=Image.LANCZOS):
    semaphore = multiprocessing.Semaphore(128)
    resize_fn = partial(resize_worker, size=size, resample=resample, return_img=True)
    sample_ids = np.arange(len(img_paths))
    total = 0
    # Start processing
    files = []
    for sid,path in zip(sample_ids[total:], img_paths[total:]):
        key = f'{size}-{str(sid).zfill(8)}'
        output_path = os.path.join(output_dir, key+".png")
        if not os.path.exists(output_path):
            files.append((sid,path))
    with multiprocessing.Pool(n_worker, initializer=worker_init, initargs=(semaphore,)) as pool:
        for i, img in tqdm(pool.imap_unordered(resize_fn, files), initial=total, total=len(img_paths)):
            key = f'{size}-{str(i).zfill(8)}'
            output_path = os.path.join(output_dir, key+".png")
            try:
                img.save(output_path)
            except BaseException as e:
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise e
            semaphore.release()

    n_imgs = len(glob(os.path.join(output_dir, "*")))
    print(" [*] Total {} image saved to {}".format(n_imgs, output_dir))
    assert n_imgs == len(img_paths)


def prepare_lmdb(env_func, img_paths, n_worker, size=128, resample=Image.LANCZOS, specific_indices=None, scan=False, n_steps=None):
    semaphore = multiprocessing.Semaphore(128)
    resize_fn = partial(resize_worker, size=size, resample=resample)
    sample_ids = np.arange(len(img_paths))

    total = 0

    with env_func() as env:
        with env.begin(write=True) as txn:

            if txn.id() != 0: # DB already has entries!
                """
                print(" [*] Filtering already processed entries...")
                for cursor,(sid,path) in enumerate(tqdm(zip(sample_ids,img_paths), total=len(img_paths))):
                    key = f'{size}-{str(sid).zfill(8)}'.encode('utf-8')
                    if txn.get(key) is None:
                        print(" [*] Ignores {} existing entries, {} remaining!".format(total, len(img_paths)-total))
                        break # end searching on the first failure, i.e., discard results afterward
                    else:
                        total += 1
                """
                length_record = txn.get("length".encode("utf-8"))
                if specific_indices is not None:
                    total = 0
                    print(" [*] Dataset fixing mode for specific indices = {}".format(specific_indices))
                elif length_record is None: # somehow corrupted
                    total = 0
                    print(" [!] Length record is corrupted, reset to 0...")
                else:
                    total = int(length_record.decode("utf-8"))
                if total >= len(img_paths):
                    print(" [*] Found existing complete db, skip!")
                    if scan:
                        pass # Keep going for scanning
                    else:
                        return
                else:
                    total = max(total-2*n_worker, 0)
                    print(" [*] Start from previous end point at {}!".format(total))

    if n_steps is not None:
        sample_ids = sample_ids[:total+n_steps]
        img_paths = img_paths[:total+n_steps]

    if scan and total > 0:
        print(" [*] Start scanning")
        with env_func() as env:
            with env.begin(write=False) as txn:
                files = []
                for i in tqdm(range(total)):
                    key = f'{size}-{str(i).zfill(8)}'.encode('utf-8')
                    if txn.get(key) is None:
                        files.append(sample_ids[i], img_paths[i])
                        print(" [!] Found corrupted key at {}".format(i))
        print(" [*] Found {} corrupted records, fixing...".format(len(files)))
    elif specific_indices is None:
        files = [(sid,path) for sid,path in zip(sample_ids[total:], img_paths[total:])]
    else:
        files = [(sample_ids[idx],img_paths[idx]) for idx in specific_indices]
    # Start processing
    with multiprocessing.Pool(n_worker, initializer=worker_init, initargs=(semaphore,)) as pool:
        env = env_func()
        try:
            for i, img in tqdm(pool.imap(resize_fn, files), initial=total, total=len(img_paths)):
                key = f'{size}-{str(i).zfill(8)}'.encode('utf-8')
                with env.begin(write=True) as txn:
                    txn.put(key, img)
                    if specific_indices is None:
                        total += 1
                        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))
                del img
                semaphore.release()
                if i%100000:
                    import gc; gc.collect()
        finally:
            env.close()

def parse_list(s):
    return [int(v) for v in s.split(",")]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument("--specific_indices", type=parse_list, default=None)
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--n_steps", type=int, default=None)
    args = parser.parse_args()

    hostname = socket.gethostname()
    cur_lmdb_root = None
    if hostname in LMDB_ROOTS:
        cur_lmdb_root = LMDB_ROOTS[hostname]
    else:
        for entry in LMDB_ROOTS["unspecified"]:
            if os.path.exists(entry):
                cur_lmdb_root = entry
    if cur_lmdb_root is None:
        cur_lmdb_root = config.data_params.lmdb_root
    if not os.path.exists(cur_lmdb_root):
        os.makedirs(cur_lmdb_root)
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)
    print(" [*] Config {} loaded!".format(args.config))

    if not hasattr(config.data_params, "num_valid"):
        config.data_params.num_valid = 0

    img_paths = sorted(glob(os.path.join(config.data_params.raw_data_root, "*")))
    print(" [*] Make dataset `{}` at `{}`, resolution {}, num samples {} (will use {} train, {} valid)".format(
        config.data_params.dataset,
        os.path.join(config.data_params.raw_data_root, "*"),
        config.train_params.data_size,
        len(img_paths),
        config.data_params.num_train,
        config.data_params.num_valid))

    if not args.scan:
        assert len(img_paths) != 0, "Found no samples at {}".format(os.path.join(config.data_params.raw_data_root, "*"))
        assert len(img_paths) >= config.data_params.num_train + config.data_params.num_valid, \
            "{} train and {} valid, sum up {} samples is more than dataset with {} samples".format(config.data_params.num_train, config.data_params.num_valid, config.data_params.num_train + config.data_params.num_valid, len(img_paths))

    train_paths = img_paths[:config.data_params.num_train]
    valid_paths = img_paths[config.data_params.num_train:config.data_params.num_train+config.data_params.num_valid]

    if args.img:
        train_img_dir = os.path.join(cur_lmdb_root, config.data_params.dataset, "train-img")
        valid_img_dir = os.path.join(cur_lmdb_root, config.data_params.dataset, "valid-img")
        
        if not os.path.exists(train_img_dir): os.makedirs(train_img_dir)
        if not args.train_only:
            if not os.path.exists(valid_img_dir): os.makedirs(valid_img_dir)

        print(" [*] Processing training set...")
        prepare_img(
            train_img_dir, train_paths, args.n_worker, 
            size=config.train_params.data_size, 
            resample=Image.LANCZOS) 

        if not args.train_only:
            print(" [*] Processing validation set...")
            prepare_img(
                valid_img_dir, valid_paths, args.n_worker,
                size=config.train_params.data_size,
                resample=Image.LANCZOS)
    else:
        train_lmdb_dir = os.path.join(cur_lmdb_root, config.data_params.dataset, "train")
        valid_lmdb_dir = os.path.join(cur_lmdb_root, config.data_params.dataset, "valid")

        if not os.path.exists(train_lmdb_dir): os.makedirs(train_lmdb_dir)
        if not args.train_only:
            if not os.path.exists(valid_lmdb_dir): os.makedirs(valid_lmdb_dir)

        print(" [*] Processing training set...")
        env_func = lambda: lmdb.open(train_lmdb_dir, map_size=1024 ** 4, readahead=False)
        prepare_lmdb(
            env_func, train_paths, args.n_worker, 
            size=config.train_params.data_size, 
            resample=Image.LANCZOS, 
            specific_indices=args.specific_indices,
            scan=args.scan,
            n_steps=args.n_steps)

        if not args.train_only:
            print(" [*] Processing validation set...")
            env_func = lambda: lmdb.open(valid_lmdb_dir, map_size=1024 ** 4, readahead=False)
            prepare_lmdb(
                env_func, valid_paths, args.n_worker,
                size=config.train_params.data_size,
                resample=Image.LANCZOS, 
                specific_indices=args.specific_indices,
                scan=args.scan,
                n_steps=args.n_steps)

