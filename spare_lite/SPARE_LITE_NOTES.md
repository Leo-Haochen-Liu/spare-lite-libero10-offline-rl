# SpaRe-lite Notes

Last updated: 2026-04-25

| 中文 | English |
| --- | --- |
| 这份文档记录了当前 `spare_lite` 中 `candidate` proxy experiment 的关键点，方便之后快速恢复上下文。 | This document records the key points of the current `candidate` proxy experiment in `spare_lite`, so the logic can be recovered quickly later. |
| 当前 phase-A validation 是一个轻量级的 proxy experiment，而不是完整的 online RL environment。 | The current phase-A validation is a lightweight proxy experiment rather than a full online RL environment. |
| 当前问题被压缩成一个单步的 candidate selection problem。 | The current problem is reduced to a single-step candidate selection problem. |
| 一个 `candidate` 是一个候选 latent 向量，而不是一条 trajectory，也不是一整个 rollout。 | A `candidate` is a candidate latent vector rather than a trajectory or a full rollout. |
| 一个 `candidate` 更接近于在固定输入 `x` 下的一个候选输出表征。 | A `candidate` is better understood as one candidate output representation under a fixed input `x`. |
| 当前代码里会先构造一个 `candidate tensor`，用来同时表示一个 batch 里所有样本的所有 candidate embeddings。 | The current code first constructs a `candidate tensor`, which is used to represent all candidate embeddings for all samples in one batch. |
| 这个 `candidate tensor` 的形状写成 `[batch_size, num_candidates, hidden_dim]`。 | The shape of this `candidate tensor` is written as `[batch_size, num_candidates, hidden_dim]`. |
| `batch_size` 表示一次一起处理多少个输入样本。 | `batch_size` indicates how many input samples are processed together. |
| `num_candidates` 表示每个输入样本会构造多少个 candidate embeddings。 | `num_candidates` indicates how many candidate embeddings are constructed for each input sample. |
| `hidden_dim` 表示每个 candidate embedding 向量的维度。 | `hidden_dim` indicates the dimension of each candidate embedding vector. |
| 如果 `batch_size = 2`、`num_candidates = 4`、`hidden_dim = 3`，那么这个 candidate tensor 的形状就是 `[2, 4, 3]`。 | If `batch_size = 2`, `num_candidates = 4`, and `hidden_dim = 3`, then the shape of this candidate tensor is `[2, 4, 3]`. |
| 这个 `[2, 4, 3]` 的意思是：有 2 个样本，每个样本有 4 个 candidates，每个 candidate 用一个 3 维向量表示。 | This `[2, 4, 3]` means there are 2 samples, each sample has 4 candidates, and each candidate is represented by a 3-dimensional vector. |
| 当前 candidate 的构造逻辑写在 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_spare_latent_smoke.py`。 | The current candidate construction logic is implemented in `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/rl_spare_latent_smoke.py`. |
| 关键函数是 `make_candidates(...)`。 | The key function is `make_candidates(...)`. |
| 当前 phase-A proxy experiment 没有单独的 candidate 文件，candidate 是运行时在线生成的。 | The current phase-A proxy experiment does not use a separate candidate file, and candidates are generated online at runtime. |
| 当前本地最直接执行 phase-A 验证的 bash 脚本是 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh`。 | The most direct local bash script for executing the current phase-A validation is `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh`. |
| 当前本地执行多 seed latent sweep 的 bash 脚本是 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh`。 | The local bash script for running the current multi-seed latent sweep is `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh`. |
| 当前本地执行单次 latent smoke 的 bash 脚本是 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent.sh`。 | The local bash script for running the current single latent smoke is `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent.sh`. |
| 当前本地同步代码和数据到远端的 bash 脚本是 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/transfer_remote_assets.sh`。 | The local bash script for syncing code and data to the remote machine is `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/transfer_remote_assets.sh`. |
| 当前远端实际执行 phase-A bundle 的路径是 `/root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh`。 | The current remote path that actually executes the phase-A bundle is `/root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_phase_a_bundle.sh`. |
| 当前远端实际执行 multi-seed latent sweep 的路径是 `/root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh`。 | The current remote path that actually executes the multi-seed latent sweep is `/root/autodl-tmp/SimpleVLA-RL/spare_lite/run_remote_rl_spare_latent_sweep.sh`. |
| 对于每个输入样本，第一个 candidate 被定义为 `expert candidate`。 | For each input sample, the first candidate is defined as the `expert candidate`. |
| 对于每个输入样本，其余 candidates 被定义为 `negative candidates`。 | For each input sample, the remaining candidates are defined as `negative candidates`. |
| 当前默认设置下，每个输入样本有 1 个 expert candidate 和 3 个 negative candidates，总共 4 个 candidates。 | In the current default setup, each input sample has 1 expert candidate and 3 negative candidates, for a total of 4 candidates. |
| 当前没有外部标注文件来区分 expert 和 negative，这完全是代码层面的约定。 | There is currently no external annotation file that marks expert versus negative, and this is entirely a code-level convention. |
| 当前代码约定 candidate index `0` 是 expert candidate，candidate indices `1...K-1` 是 negative candidates。 | The current code convention treats candidate index `0` as the expert candidate and candidate indices `1...K-1` as negative candidates. |
| 当前 negative candidates 的生成方式是：从 reference latent 出发，加上随机扰动来构造负样本。 | The current negative candidates are generated by starting from the reference latent and adding random perturbations to construct negative samples. |
| 在 `orthogonal_noise` 模式下，噪声在 reference latent 方向上的分量会被去掉，只保留正交方向的扰动。 | In `orthogonal_noise` mode, the component of the noise along the reference latent direction is removed so that only orthogonal perturbations remain. |
| 当前 negative candidates 是合成出来的扰动样本，而不是直接从数据集里读取的动作。 | The current negative candidates are synthetic perturbation samples rather than actions read directly from the dataset. |
| 当前已经开始补真实 negative candidate 的路径，`/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/export_hdf5_to_rl_jsonl.py` 现在支持从 LIBERO demo 动作池里抽取 dataset-derived negatives。 | The path toward real negative candidates has now started: `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/export_hdf5_to_rl_jsonl.py` now supports sampling dataset-derived negatives from the LIBERO demo action pool. |
| 当前固定的小型本地 LIBERO subset 路径是 `/Users/haochenliu/Downloads/libero_demo_cache/small_train.jsonl`，包含 16 条样本。 | The current fixed small local LIBERO subset path is `/Users/haochenliu/Downloads/libero_demo_cache/small_train.jsonl`, and it contains 16 samples. |
| 当前固定的中型本地 LIBERO subset 路径是 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/medium_train.jsonl`，包含 80 条样本。 | The current fixed medium local LIBERO subset path is `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/local_data/medium_train.jsonl`, and it contains 80 samples. |
| 当前固定的远端 small subset 路径是 `/root/autodl-tmp/data/libero_small/small_train.jsonl`。 | The current fixed remote small subset path is `/root/autodl-tmp/data/libero_small/small_train.jsonl`. |
| 当前固定的远端 medium subset 路径是 `/root/autodl-tmp/data/libero_medium/medium_train.jsonl`。 | The current fixed remote medium subset path is `/root/autodl-tmp/data/libero_medium/medium_train.jsonl`. |
| 当前 JSONL 数据里包含的字段主要有 `image_path`、`instruction`、`action`，以及一些变体里的 `action_tokens`。 | The current JSONL data mainly contains fields such as `image_path`, `instruction`, `action`, and `action_tokens` in some variants. |
| 当前这个 LIBERO subset 本身不包含用于 proxy validation 的 reward 字段。 | The current LIBERO subset itself does not contain a reward field for the proxy validation. |
| 当前 phase-A proxy reward 不是从 LIBERO subset 里读出来的，而是在运行时人工构造的。 | The current phase-A proxy reward is not read from the LIBERO subset and is instead constructed manually at runtime. |
| 当前 `R1` 的定义是：expert candidate 的 `R1 = 1`，negative candidates 的 `R1 = 0`。 | The current `R1` is defined so that the expert candidate gets `R1 = 1` and the negative candidates get `R1 = 0`. |
| 当前原始 reward 本质上是一个 proxy reward，也就是选中 expert candidate 得 1，选中 negative candidate 得 0。 | The current original reward is essentially a proxy reward: selecting the expert candidate gives 1, and selecting a negative candidate gives 0. |
| `expert_hit` 表示当前被选中的 candidate 是否正好是 expert candidate。 | `expert_hit` indicates whether the currently selected candidate is exactly the expert candidate. |
| 因为当前 expert candidate 被定义为 index `0`，所以当 chosen index = 0 时，`expert_hit = 1`；否则 `expert_hit = 0`。 | Because the current expert candidate is defined as index `0`, when the chosen index = 0, `expert_hit = 1`; otherwise `expert_hit = 0`. |
| 平均 `expert_hit` 表示 policy 选中 expert candidate 的比例。 | Average `expert_hit` indicates the proportion of times the policy selects the expert candidate. |
| `expert_hit` 不是最终任务的 success rate，它只是当前 lightweight validation 里的一个 proxy metric。 | `expert_hit` is not the final task success rate; it is only a proxy metric in the current lightweight validation. |
| 当前这套实现还不是海报里描述的 fully wired final implementation。 | The current setup is still not the fully wired final implementation described in the poster. |
| 当前 `z_ref` 还没有接到最终的 SpatialVLA ego3D encoder。 | The current `z_ref` is not yet connected to the final SpatialVLA ego3D encoder. |
| 当前远端 validation 路径使用的是 `/root/autodl-tmp/checkpoints/facebook-dinov2-base`，也就是 DINOv2。 | The current remote validation path uses `/root/autodl-tmp/checkpoints/facebook-dinov2-base`, which is DINOv2. |
| 所以当前 reference branch 是基于 DINOv2 的，而不是真正的 SpatialVLA ego3D reference branch。 | So the current reference branch is DINOv2-based rather than the true SpatialVLA ego3D reference branch. |
| 在 `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/modeling.py` 里，结构上确实存在一个可学习的 policy-side latent head。 | In `/Users/haochenliu/Documents/research/SimpleVLA-RL/spare_lite/modeling.py`, there is structurally a learnable policy-side latent head. |
| 但是在当前 lightweight bandit validation 里，reward 侧实际使用的仍然是构造出来的 candidate embeddings。 | However, in the current lightweight bandit validation, the reward side still uses the constructed candidate embeddings in practice. |
| 所以当前的 `R2` 仍然属于 proxy validation setup，而不是最终 fully wired 的 `policy-side spatial latent vs SpatialVLA ego3D latent` 实现。 | So the current `R2` still belongs to a proxy validation setup rather than the final fully wired `policy-side spatial latent vs SpatialVLA ego3D latent` implementation. |
| 当前 reward 结构写成 `R = R1 + lambda * R2`。 | The current reward structure is written as `R = R1 + lambda * R2`. |
| 但是当前 `R2` 不是直接使用 raw cosine。 | However, the current `R2` does not directly use the raw cosine. |
| 当前设计里，`R2 = ReLU(cosine(z_pol, z_ref) - 0.72)`。 | In the current design, `R2 = ReLU(cosine(z_pol, z_ref) - 0.72)`. |
| 更准确地说，当前 reward 写成 `R = R1 + lambda * max(cosine(z_pol, z_ref) - 0.72, 0)`。 | More precisely, the current reward is `R = R1 + lambda * max(cosine(z_pol, z_ref) - 0.72, 0)`. |
| 这就是当前的 `centered_relu` reward design。 | This is the current `centered_relu` reward design. |
| 当前 phase-A setup 作为 lightweight proxy experiment 是有用的，但它还不是最终完整的 SpaRe implementation。 | The current phase-A setup is useful as a lightweight proxy experiment, but it is still not the final full SpaRe implementation. |
| 当前最重要的简化包括：expert 和 negative 是代码约定而不是外部标注，`R1` 是人为构造的 proxy reward，`z_ref` 目前基于 DINOv2，以及当前 `R2` 路径仍然是 proxy validation 路径。 | The most important current simplifications are that expert and negative are defined by code convention rather than external annotation, `R1` is a constructed proxy reward, `z_ref` is currently DINOv2-based, and the current `R2` path is still a proxy validation path. |
