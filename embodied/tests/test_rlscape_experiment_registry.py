import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'experiments' / 'rlscape' / 'registry.json'


def test_rlscape_experiment_registry_is_unique_and_well_formed():
  payload = json.loads(REGISTRY.read_text())
  experiments = payload['experiments']
  ids = [item['id'] for item in experiments]
  slugs = [item['slug'] for item in experiments]
  assert len(ids) == len(set(ids))
  assert len(slugs) == len(set(slugs))
  assert ids == [
      'stage0', 'stage1a', 'stage1b', 'stage2', 'stage3a', 'stage3b',
      'stage4a', 'stage4b', 'stage5a', 'stage5b', 'stage5c', 'stage6']
  assert all(re.fullmatch(r'stage\d+[a-z]?', item) for item in ids)
  assert all(re.fullmatch(r'[a-z0-9_]+', item) for item in slugs)


def test_complete_stages_point_to_maintained_repository_artifacts():
  experiments = json.loads(REGISTRY.read_text())['experiments']
  complete = [item for item in experiments if item['status'] == 'complete']
  assert {item['id'] for item in complete} == {
      'stage0', 'stage1a', 'stage1b', 'stage2', 'stage3a', 'stage3b',
      'stage4a', 'stage4b'}
  for item in complete:
    assert (ROOT / item['runner']).is_file()
    if protocol := item.get('protocol'):
      assert (ROOT / protocol).is_file()
  assert (ROOT / 'reports/rlscape_ir_research_diary/report.tex').is_file()


def test_ready_stages_have_runners_protocols_but_no_completion_claims():
  experiments = json.loads(REGISTRY.read_text())['experiments']
  ready = [item for item in experiments if item['status'] == 'ready_to_run']
  assert {item['id'] for item in ready} == {'stage5a'}
  for item in ready:
    assert (ROOT / item['protocol']).is_file()
    assert (ROOT / item['runner']).is_file()
    assert item['hypothesis_status'] == 'unresolved'
    assert not (ROOT / f'{item["id"]}_complete').exists()
