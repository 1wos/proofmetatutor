# Agent Skills

Agent Skills authored for this project, in the [google/skills](https://github.com/google/skills)
format (frontmatter `name` + `description`, then a markdown playbook).

## `cloud-tpu-training`
A reusable skill that teaches an agent to provision Cloud TPU VMs and run
JAX/Flax or Keras 3 training jobs (including a Gemma LoRA fine-tune), read/write
artifacts through the `/gcs` fuse mount, and tear resources down for cost
control. It fills a gap in `google/skills/skills/cloud/` (which has Cloud Run,
GKE, BigQuery, AlloyDB… but no TPU skill).

### Contributing it upstream
```bash
git clone https://github.com/google/skills.git
cp -r skills/cloud-tpu-training skills/skills/cloud/cloud-tpu-training
# follow CONTRIBUTING.md, then open a PR adding the cloud-tpu-training skill
```
This is distilled from `docs/cloud_tpu_runbook.md` — the same commands used to
train ProofMetaTutor's verifier on a TPU v6e.
