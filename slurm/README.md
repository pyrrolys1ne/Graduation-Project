# Slurm 服务器运行手册

本目录用于原生 Linux + NVIDIA GPU 的 Slurm 集群。登录节点只做代码同步、配置和作业提交；
模型加载、GPU 预检、服务启动和实验必须在 Slurm 分配的计算节点内运行。

## 1. 登录与同步

账号密码和二次验证码由 SSH 交互完成。脚本不会保存密码、验证码或绕过二次验证。

从本机首次同步项目：

```bash
slurm/sync_project.sh <user@login-host> '<remote-project-path>'
```

该命令不会上传 `.venv*`、`results/`、本地 `slurm/cluster.env` 或 Git 元数据。若中心禁用
`rsync`，改用 `sftp` 上传仓库；不要上传 WSL 虚拟环境。

## 2. 查询集群参数

登录后先查询实际分区、QOS 和 module 名称，不要直接照抄示例值：

```bash
sinfo
scontrol show partition
sacctmgr list qos
module avail python
module avail cuda
```

复制配置并填写查询结果：

```bash
cd <remote-project-path>
cp slurm/cluster.env.example slurm/cluster.env
vi slurm/cluster.env
```

至少核对：

- `SLURM_PARTITION`、`SLURM_QOS`、`SLURM_ACCOUNT`
- `SLURM_GRES`，通常是 `gpu:1`，也可能要求具体型号
- `SLURM_MODULES`，例如集群实际提供的 Python/CUDA module
- `PROJECT_ROOT` 与持久化 `STORAGE_ROOT`
- `TORCH_INDEX_URL`，应按 PyTorch 官方支持矩阵选择，不按驱动显示的 CUDA 版本机械命名

脚本会在每个计算节点作业开始时执行 `. /etc/profile`，然后 `module purge` 和
`module load ${SLURM_MODULES}`。无需把项目 module 永久写入 `~/.bashrc`，这样更容易复现实验环境。

## 3. 存储接入

普通 Slurm 用户通常无权执行 `mount`。home/project 文件系统由集群管理员挂载；本项目只把
模型缓存、数据集和结果目录接到持久化共享存储：

```bash
slurm/prepare_storage.sh
```

脚本设置并创建 `HF_HOME`、`TORCH_HOME`，并尝试建立：

```text
data/datasets -> $DATASET_ROOT
results       -> $RESULTS_ROOT
```

已有非空目录不会被移动或覆盖。请先人工核对、迁移后再重跑脚本。作业临时文件优先写
`$SLURM_TMPDIR/encoder-sched`，但正式结果始终写共享存储。

## 4. 初始化环境

安装环境需要 GPU 作业，因为 bootstrap 最后会执行严格 CUDA 预检：

```bash
slurm/submit.sh bootstrap
squeue -u "$USER"
```

日志位于 `slurm/logs/encoder-bootstrap-<jobid>.out` 和 `.err`。bootstrap 会创建全新的 `.venv`、
安装项目依赖、执行运行时预检并运行测试。若计算节点不能访问公网，应先按中心提供的方法准备
wheel 或内部 PyPI/Hugging Face 镜像，再调整安装源。

若共享 Hugging Face 缓存尚无模型权重，并且计算节点允许访问 Hugging Face：

```bash
slurm/submit.sh cache-model
```

计算节点不能联网时，应在可联网机器执行同样的 Hugging Face 下载，并将完整缓存同步到
`$HF_HOME`。当前配置启用 `local_files_only: true`，缺少权重时正式服务会直接失败，不会临时联网。

环境建立后运行 GPU 预检和两个尺寸的真实 CLIP smoke test：

```bash
slurm/submit.sh check
```

## 5. 交互调试

中心禁止 `salloc`，因此交互入口只使用 `srun`：

```bash
slurm/interactive.sh
nvidia-smi
encoder-sched-check --config "$ENCODER_CONFIG"
exit
```

可以在命令末尾附加 Slurm 参数覆盖本地配置，例如：

```bash
slurm/interactive.sh --time=00:30:00 --cpus-per-task=4
```

## 6. 提交实验

单次服务与客户端同节点实验：

```bash
slurm/submit.sh experiment
```

脚本在同一个 GPU allocation 中启动服务、等待 `/health` 就绪、回放负载并清理后台服务。
结果写入 `$RESULTS_ROOT/slurm/<jobid>/`。

提交多臂、多轮重复实验：

```bash
slurm/submit.sh repeated
```

`REPEATED_WORKLOAD` 留空时，Python 实验入口会按实验组选择经过验证的默认负载。不要把欠饱和
负载强行用于吞吐对照。重复实验默认使用独立的 `REPEATED_CONCURRENCY=0`（开环）和
`REPEATED_SHARDS=4`，不会继承单次实验的连接上限。可以临时覆盖资源或导出的实验变量：

```bash
REPEATED_GROUP=graph REPEATS=5 slurm/submit.sh repeated --time=08:00:00
```

注意：命令行前缀变量是否被 `sbatch --export=ALL` 继承取决于 shell 导出状态。需要稳定复现时，
应把实验参数写入 `slurm/cluster.env`，或先执行 `export REPEATED_GROUP=graph`。

## 7. 启动常驻服务

需要人工访问 API 时可提交服务作业：

```bash
slurm/submit.sh service --time=02:00:00
```

服务只在 allocation 生命周期内存在，不使用 systemd。作业日志会打印计算节点和端口，并在
`$RESULTS_ROOT/slurm-services/<jobid>.endpoint` 保存端点。计算节点通常不能从公网直连；是否允许
经登录节点做 SSH 端口转发必须遵守中心规定。停止服务：

```bash
scancel <jobid>
```

## 8. 查看状态与日志

```bash
squeue -u "$USER"
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode,AllocTRES
tail -f slurm/logs/<job-name>-<jobid>.out
tail -f slurm/logs/<job-name>-<jobid>.err
```

若中心提供 `speek`、`jload` 或 `sload`，可用于查看实时输出和节点负载，但原始实验结果仍以
`$RESULTS_ROOT` 中落盘文件为准。

## 9. CUDA MPS 说明

当前仓库尚未实现 TTS runtime、replica router 或 MPS 生命周期管理，因此这些 Slurm 模板只运行
现有 CLIP/VLM 系统。不能仅通过在作业脚本中启动 `nvidia-cuda-mps-control` 就声称支持 TTS。
后续接入 TTS 时，应在单节点独占 GPU allocation 内为每个作业创建独立的 MPS pipe/log 目录，
校验 replica 确实连接到 MPS server，并由作业 trap 关闭 MPS；不得复用其他作业的全局 MPS 状态。
