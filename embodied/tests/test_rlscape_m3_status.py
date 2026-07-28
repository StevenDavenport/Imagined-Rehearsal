import json

from scripts import rlscape_m3_status as status


def test_status_collects_training_milestones_and_eval_pairs(tmp_path):
  training = tmp_path / 'training'
  checkpoint = training / 'ckpt' / 'one'
  checkpoint.mkdir(parents=True)
  (checkpoint / 'done').touch()
  import pickle
  (checkpoint / 'step.pkl').write_bytes(pickle.dumps(250_000))
  (checkpoint.parent / 'latest').write_text('one')
  (training / 'episodes.jsonl').write_text(json.dumps({
      'episode/score': 1,
      'episode/log/task_success/max': 1,
  }) + '\n')
  milestone = tmp_path / 'milestones' / 'step_000000250000'
  milestone.mkdir(parents=True)
  (milestone / 'archive_complete').touch()
  evaluation = tmp_path / 'evaluations'
  evaluation.mkdir()
  (evaluation / 'queue.json').write_text(json.dumps({
      'status': 'running',
      'states': {
          'a': {'status': 'complete'},
          'b': {'status': 'failed'},
      },
  }))
  (tmp_path / 'supervisor_status.json').write_text(json.dumps({
      'status': 'training',
      'total_restarts': 2,
  }))

  result = status.collect(tmp_path)

  assert result['training_step'] == 250_000
  assert result['milestones'] == 1
  assert result['recent_success_rate'] == 1.0
  assert result['evaluation_pairs_complete'] == 1
  assert result['evaluation_pairs_failed'] == 1
  assert result['total_restarts'] == 2
