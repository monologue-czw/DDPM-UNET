import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(64, 512), resample=Image.BICUBIC, lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img = imgs
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            hr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                lr_img_bytes = image_convert_bytes(lr_img)
                hr_img_bytes = image_convert_bytes(hr_img)
                sr_img_bytes = image_convert_bytes(sr_img)
                #  以上三段为gpt加入的
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img_bytes)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img_bytes)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img_bytes)



        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))


'''                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
'''

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(64, 512), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
    else:
        print("Opening LMDB environment at:", out_path)
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img = imgs
            if not lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
                sr_img.save(
                    '{}/sr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], i.zfill(5)))
            else:
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
            total += 1
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='F:\\ffhq'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='F:\\ffhq1')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    #  resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    # args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])

    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)

'''该脚本出现问题，该脚本的目的是对512*512的数据集进行操作，变成输出路径下三个子文件夹，包括16*16，512*512，以及16*16插值变成512的图片，我查询一些代码修改方法，但是没有成功，报错位置始终一致（77行）'''
'''方便复制到chatgpt，我把报错代码复制到下面
12170/52001 images processed Process Process-1:
Process Process-2:
Process Process-3:
Traceback (most recent call last):
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 536, in _save
    fh = fp.fileno()
         ^^^^^^^^^
AttributeError: '_idat' object has no attribute 'fileno'
Traceback (most recent call last):
Traceback (most recent call last):

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 536, in _save
    fh = fp.fileno()
         ^^^^^^^^^
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 536, in _save
    fh = fp.fileno()
         ^^^^^^^^^
AttributeError: '_idat' object has no attribute 'fileno'
AttributeError: '_idat' object has no attribute 'fileno'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\conda\envs\pytorch\Lib\multiprocessing\process.py", line 314, in _bootstrap
    self.run()
  File "D:\conda\envs\pytorch\Lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "D:\pycharm\Image-Super-Resolution-via-Iterative-Refinement-master\Image-Super-Resolution-via-Iterative-Refinement-master\data\prepare_data.py", line 77, in prepare_process_worker
    hr_img.save(
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\Image.py", line 2439, in save
    save_handler(self, fp, filename)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1402, in _save
    ImageFile._save(im, _idat(fp, chunk), [("zip", (0, 0) + im.size, 0, rawmode)])
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 540, in _save
    _encode_tile(im, fp, tile, bufsize, None, exc)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 560, in _encode_tile
    fp.write(data)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1077, in write
    self.chunk(self.fp, b"IDAT", data)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1064, in putchunk
    fp.write(data)
OSError: [Errno 22] Invalid argument
  File "D:\conda\envs\pytorch\Lib\multiprocessing\process.py", line 314, in _bootstrap
    self.run()
  File "D:\conda\envs\pytorch\Lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "D:\pycharm\Image-Super-Resolution-via-Iterative-Refinement-master\Image-Super-Resolution-via-Iterative-Refinement-master\data\prepare_data.py", line 77, in prepare_process_worker
    hr_img.save(
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\Image.py", line 2439, in save
    save_handler(self, fp, filename)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1402, in _save
    ImageFile._save(im, _idat(fp, chunk), [("zip", (0, 0) + im.size, 0, rawmode)])
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 540, in _save
    _encode_tile(im, fp, tile, bufsize, None, exc)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 560, in _encode_tile
    fp.write(data)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1077, in write
    self.chunk(self.fp, b"IDAT", data)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1064, in putchunk
    fp.write(data)
OSError: [Errno 22] Invalid argument
  File "D:\conda\envs\pytorch\Lib\multiprocessing\process.py", line 314, in _bootstrap
    self.run()
  File "D:\conda\envs\pytorch\Lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "D:\pycharm\Image-Super-Resolution-via-Iterative-Refinement-master\Image-Super-Resolution-via-Iterative-Refinement-master\data\prepare_data.py", line 77, in prepare_process_worker
    hr_img.save(
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\Image.py", line 2439, in save
    save_handler(self, fp, filename)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1402, in _save
    ImageFile._save(im, _idat(fp, chunk), [("zip", (0, 0) + im.size, 0, rawmode)])
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 540, in _save
    _encode_tile(im, fp, tile, bufsize, None, exc)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\ImageFile.py", line 560, in _encode_tile
    fp.write(data)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1077, in write
    self.chunk(self.fp, b"IDAT", data)
  File "D:\conda\envs\pytorch\Lib\site-packages\PIL\PngImagePlugin.py", line 1064, in putchunk
    fp.write(data)
OSError: [Errno 22] Invalid argument
12170/52001 images processed 
进程已结束，退出代码为 0
'''