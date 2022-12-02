import zipfile
import os
def zip_ya(startdir, file_news):
    z = zipfile.ZipFile(file_news, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(startdir):
        fpath = dirpath.replace(startdir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
            print('success')
    z.close()

root = '/home/cx/cx1/TransT_experiments/TransT_mt_iou_seg/pysot_toolkit/results/GOT-10k'
names = os.listdir(root)
for name in names:
    path = os.path.join(root,name)
    startdir = path
    file_news = startdir+'.zip'
    zip_ya(startdir, file_news)