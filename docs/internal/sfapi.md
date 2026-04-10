---
search: false
---

# SFAPI Runner

The backend dispatches simulation jobs to **NERSC Perlmutter** via the
[Superfacility API](https://docs.nersc.gov/services/sfapi/) (SFAPI).
All SFAPI logic lives in `backend/app/sfapi_runner.py`.

## Submission flow

1. **Estimate resources** from the request parameters:
   - `target_per_task`: 50 events for ttbar, 100 for other channels
   - `n_tasks = ceil(events / target_per_task)`, capped at 256
   - `tasks_per_node = 32` (Perlmutter CPU: 128 cores, DDSim is memory-bound)
   - `n_nodes = ceil(n_tasks / 32)`, capped at 8

2. **Select queue** based on estimated wall time:

   | Condition | QOS | Wall time |
   | :-- | :-- | :-- |
   | est < 25 min AND 1 node | `debug` | 30 min |
   | est < 2 h | `regular` | 2 h |
   | otherwise | `regular` | 4 h |

3. **Render sbatch script** from `backend/app/sbatch_template.sh.j2` (Jinja2).

4. **Upload script** to Perlmutter scratch via SFAPI, then `submit_job()`.

5. **Record** `nersc_jobid` and set state to `submitted`.

## Sbatch template

Template variables: `request_id`, `channel`, `events`, `pileup`, `seed`,
`project`, `qos`, `time_limit`, `n_nodes`, `tasks_per_node`,
`events_per_task`, `work_dir`, `image`, `repo_branch`, `user_email`,
`upload_to_hf`, `output_hf_repo`, `hf_token`.

### Single-task path

```bash
shifter bash -c "
  source /workspace/scripts/cli/setup_container_env.sh &&
  bash /workspace/scripts/cli/run_pipeline_docker.sh --channel {channel}
"
```

### Multi-node / multi-task path

```bash
srun --exact shifter bash -c '
  source /workspace/scripts/cli/setup_container_env.sh
  TASK_ID=${SLURM_PROCID:-0}
  export RUN_ID=$TASK_ID
  export EVENTS={events_per_task}
  export SEED=$(({seed} + TASK_ID))
  bash /workspace/scripts/cli/run_pipeline_docker.sh --channel {channel}
'
```

Each SLURM task gets a unique `RUN_ID` and `SEED`, producing independent
event batches that are merged after all tasks complete.

### Work directory layout

```
/pscratch/sd/{u}/{nersc_user}/colliderml/{request_id}/
├── repo/              # shallow clone of colliderml-production@{branch}
├── runs/
│   ├── 0/             # task 0 output
│   ├── 1/             # task 1 output
│   └── …
└── merged/            # (if multi-task) combined output
```

### Output upload

If `upload_to_hf=true` and `HF_TOKEN` is set, the sbatch epilogue
creates a HuggingFace dataset repo `{HF_DATASET_ORG}/ColliderML-Service-{request_id}`
and uploads all Parquet files from `runs/`.

## Polling

A background `asyncio` task polls every `POLL_INTERVAL_SECONDS` (default 60):

1. Call `job.update()` via SFAPI (run in a thread — the SFAPI client is synchronous)
2. Track state transitions
3. On terminal state → call `_finalise(request_id, user, slurm_state)`

Terminal SLURM states: `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`, `NODE_FAIL`.

### Finalisation

- **COMPLETED:** mark state `completed`, set `output_hf_repo`, reconcile credits
  (refund overage if actual < estimated)
- **Other terminal state:** mark state `failed`, record `error_message`, refund full charge

## Mock mode

When `SFAPI_CLIENT_ID` and `SFAPI_CLIENT_SECRET` are unset, the runner
operates in **mock mode**:

- Logs `"Mock submission for request {id}"`
- Records `nersc_jobid = "mock-0"`
- Schedules a background task that sleeps 2 s, then calls `_finalise` with `COMPLETED`

This is the default for local development and staging deploys without
NERSC credentials.

## SFAPI credentials

| Env var | Purpose |
| :-- | :-- |
| `SFAPI_CLIENT_ID` | NERSC IRIS OAuth2 client ID |
| `SFAPI_CLIENT_SECRET` | PEM-encoded private key for the IRIS client |
| `NERSC_PROJECT` | Allocation account (e.g., `m4958`) |
| `NERSC_USER` | Service account username on Perlmutter |

Obtain credentials from [iris.nersc.gov](https://iris.nersc.gov/) under
the project's API client management page.
