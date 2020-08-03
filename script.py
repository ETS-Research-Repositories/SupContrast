from random import choice

from cchelper import JobSubmiter

epoch = 1000
test_epoch=100
batch_size = 750
account = ["def-mpederso", "rrg-mpederso"]
# jobs = ["python main_supcon.py --batch_size 768   --learning_rate 0.5   --temp 0.1   --cosine ",
#         "python main_ce.py --batch_size 768 --learning_rate 0.8 --cosine ",
#         "python main_supcon.py --batch_size 768 --learning_rate 0.5 --temp 0.5 --cosine --method SimCLR"
#         ]

baseline = [
    # supcontrast
    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5   --temp 0.1  --epochs={epoch} --cosine --save_dir=save/supcontrast/baseline && "
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 5  --epochs={test_epoch} --ckpt save/supcontrast/baseline/last.pth  > save/supcontrast/baseline/result.txt",

    # simclr
    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine --method SimCLR --save_dir=save/simclr/baseline &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/simclr/baseline/last.pth > save/simclr/baseline/result.txt",

    # sup_ce
    f"python main_ce.py --batch_size {batch_size}  --learning_rate 0.8  --epochs={epoch} --cosine --save_dir=save/sup_ce > save/sup_ce.txt"
]

proposed1 = [
    # simclr
    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine --method SimCLR --cluster_regweigt=0.0001 --train_cluster --save_dir=save/simclr/cluster_0.0001 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/simclr/cluster_0.0001/last.pth > save/simclr/cluster_0.0001/result.txt",

    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine --method SimCLR --cluster_regweigt=0.001 --train_cluster --save_dir=save/simclr/cluster_0.001 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/simclr/cluster_0.001/last.pth > save/simclr/cluster_0.001/result.txt",

    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine --method SimCLR --cluster_regweigt=0.01 --train_cluster --save_dir=save/simclr/cluster_0.01 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/simclr/cluster_0.01/last.pth > save/simclr/cluster_0.01/result.txt",

    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine --method SimCLR --cluster_regweigt=0.1 --train_cluster --save_dir=save/simclr/cluster_0.1 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/simclr/cluster_0.1/last.pth > save/simclr/cluster_0.1/result.txt",
]
proposed2 = [
    # supcontrast
    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine  --cluster_regweigt=0.0001 --train_cluster --save_dir=save/supcontrast/cluster_0.0001 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/supcontrast/cluster_0.0001/last.pth > save/supcontrast/cluster_0.0001/result.txt",

    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine  --cluster_regweigt=0.001 --train_cluster --save_dir=save/supcontrast/cluster_0.001 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/supcontrast/cluster_0.001/last.pth > save/supcontrast/cluster_0.001/result.txt",

    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine  --cluster_regweigt=0.01 --train_cluster --save_dir=save/supcontrast/cluster_0.01 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/supcontrast/cluster_0.01/last.pth > save/supcontrast/cluster_0.01/result.txt",

    f"python main_supcon.py --batch_size {batch_size}   --learning_rate 0.5  --temp 0.5  --epochs={epoch} --cosine  --cluster_regweigt=0.1 --train_cluster --save_dir=save/supcontrast/cluster_0.1 &&"
    f"python main_linear.py --batch_size {batch_size}   --learning_rate 1  --epochs={test_epoch}  --ckpt save/supcontrast/cluster_0.1/last.pth > save/supcontrast/cluster_0.1/result.txt",
    
]

submitter = JobSubmiter(project_path="./", time=20, mem=32, gres="gpu:4", on_local=False,cpus_per_task=24)
submitter.prepare_env(
    ["source ./venv/bin/activate", "export OMP_NUM_THREADS=1 ", ]
)
for cmd in [*baseline, *proposed1]:
    submitter.account = choice(account)
    submitter.run(cmd)
