# Study IV portable evidence provenance

Authoritative remote root:

`/home/localadmin/experiment_logs/minigrid/twin_q_qualification_400k_v2`

The offline qualification completed six runs (two labelling arms x three Q
initialization seeds), each at 10,000 updates. The immutable latent banks
contain 12,498 states / 12,242 transitions from 256 training episodes and
6,225 states / 6,097 transitions from 128 held-out episodes. The final clean
run was created at 2026-09-01 22:41:01 BST and completed at 22:44:16 BST.

The compact files in this directory were copied directly from the remote root
on 2 September 2026. Local and remote SHA-256 hashes matched:

- `aggregate.csv`: `1d654fa485517f83a70b49ee49127ecc6067af388b4e4dcd948c2a9e1fc9c8e7`
- `decision.json`: `3267a07beb452d7f4b5710f6ee94e503ed5c74ac4f3946678580313d45754abe`
- `encoder_integrity.json`: `40cf5630d35dded70335cab9f52450d808239205e4c0438a7e6e506db958d7c6`
- `experiment_complete.json`: `da2aaea547a12395c03da07b6eccfae4ca49e6ac70f20c21c188be88e11d5214`
- `experiment_spec.json`: `a9f9cbf60a1e2ce52d97898fdb191363a5e75a8176e8bc6c18cc481ec9176044`
- `final_runs.csv`: `7a5f57f933fddc0e1be1fb9c15867a1e9e27a3108208aaccad49eed281b9cae5`
- `supervisor_status.json`: `436ea783c6cc4e5fbb1fdda430b70d2173f220b7db637b11fc506c0351d041a1`

Raw latent caches and 23 MB resumable Q checkpoints remain on office-4090.
