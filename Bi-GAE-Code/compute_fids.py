import os
import argparse
import numpy as np
import json
def save_config(config,filename):
    config_content = {}
    for key,value in config.items():
        config_content[key] = value
    fw = open(filename,'w',encoding='utf-8')
    dic_json = json.dumps(config_content,ensure_ascii=False,indent=4)
    fw.write(dic_json)
    fw.close()

def load_config(config_file):
    f = open(config_file,encoding='utf-8')
    res = f.read()
    config_content = json.loads(res)
    return  config_content

os.environ['CUDA_VISIBLE_DEVICES']="0,1"
from subprocess import Popen, PIPE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, default=".", help='workdir')
parser.add_argument('--image_size', type=int, default=512, help='size of images')
parser.add_argument("--num_exp",type=int,default=21,help="id of experiments")
parser.add_argument("--nlat",type=int,default=512,help="id of experiments")
opt = parser.parse_args()
print(opt)
files = os.listdir(opt.basedir)
results = {}
for f in files:
    if os.path.isdir(os.path.join(opt.basedir,f)):
        results[f] = {}
        # "recon":[],"gene":[]
        aim_files = os.listdir(os.path.join(opt.basedir,f))
        aim_iters = []
        path0 = os.path.join(opt.basedir,f)
        path2 = os.path.join(path0,"raw")
        if "raw" in aim_files:
            for t in aim_files:
                try:
                    aim_iters.append(int(t))
                except Exception as eee:
                    print(eee)
                    continue

            print(aim_iters)

            for it in aim_iters:
                results[f][it] = {}
                # "biganqp/9000/recon"
                path1 = os.path.join(path0,"%d" % it)

                for k in ["recon","gener"]:
                    if 'wgan' in f and "gener" in k:
                        continue
                    results[f][it][k] = []
                    path1_com = os.path.join(path1,k)
                    for i in ["2048"]:
                        p = Popen(["python", "-m", "pytorch_fid", path1_com, path2, "--dims", i],
                              stdout=PIPE)
                        stdout, stderror = p.communicate()
                        output = stdout.decode('UTF-8').strip()
                        lines = output.split(os.linesep)
                        print(lines)
                        for line in lines:
                            if "FID" in line:
                                msg = line.split()
                                results[f][it][k].append(float(msg[-1].strip()))

                    tmp_mean = np.mean(results[f][it][k])
                    results[f][it][k].append(tmp_mean)
        else:
            continue
    else:
        continue

save_config(results,"fids_exp%d_size%d-3.json" % (opt.num_exp,opt.image_size))
# lines = output.split(os.linesep)