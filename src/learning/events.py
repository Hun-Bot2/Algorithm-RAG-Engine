"""Validated v1 evidence envelopes. No inferred time or mastery."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re


KINDS = {'attempt', 'timer_started', 'timer_stopped', 'timer_confirmed'}


def identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError('invalid identifier')
    return value


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('timestamp must be text')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('invalid timestamp') from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('UTC timestamp required')
    return parsed.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def problem_key(value):
    if not isinstance(value, str) or not re.fullmatch(
        r'(leetcode|atcoder|codeforces|baekjoon):[A-Za-z0-9][A-Za-z0-9_.-]*', value
    ):
        raise ValueError('platform-qualified problem key required')


@dataclass(frozen=True)
class Event:
    event_id: str
    learner_id: str
    occurred_at: str
    recorded_at: str
    kind: str
    source: str
    payload: dict
    supersedes: str | None = None
    schema_version: int = 1

    def normalized(self):
        identifier(self.event_id)
        identifier(self.learner_id)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported event version')
        if (not isinstance(self.kind, str) or self.kind not in KINDS
                or not isinstance(self.source, str) or self.source not in {'manual', 'platform_import'}):
            raise ValueError('unsupported kind/source')
        occurred, recorded = timestamp(self.occurred_at), timestamp(self.recorded_at)
        if occurred > recorded:
            raise ValueError('event occurs after recording')
        if self.supersedes is not None:
            identifier(self.supersedes)
            if self.supersedes == self.event_id:
                raise ValueError('event cannot supersede itself')
        if not isinstance(self.payload, dict):
            raise ValueError('payload must be an object')
        p = dict(self.payload)
        if self.kind == 'attempt':
            expected = {'problem_key', 'outcome', 'hints'}
            if 'activity_type' in p:
                if self.source != 'manual':
                    raise ValueError('activity context requires manual evidence')
                if not isinstance(p['activity_type'], str) or p['activity_type'] not in {'practice', 'review', 'transfer'}:
                    raise ValueError('invalid activity type')
                expected.add('activity_type')
            if p.get('activity_type') == 'transfer':
                problem_key(p.get('transfer_from'))
                if p['transfer_from'] == p.get('problem_key'):
                    raise ValueError('transfer must use a different source problem')
                expected.add('transfer_from')
            if self.source == 'platform_import':
                expected |= {'platform', 'external_id'}
                if p.get('outcome') != 'reported_completion':
                    raise ValueError('imports are reported completion only')
                identifier(p.get('external_id'))
                if not isinstance(p.get('problem_key'), str) or p.get('platform') != p['problem_key'].split(':')[0]:
                    raise ValueError('import platform mismatch')
            if not isinstance(p.get('outcome'), str) or p['outcome'] not in {'independent', 'assisted', 'failed', 'reported_completion'}:
                raise ValueError('invalid outcome')
            if p.get('hints') is not None and type(p['hints']) is not bool:
                raise ValueError('hints must be boolean or null')
            if p.get('outcome') == 'independent' and p.get('hints') is not False:
                raise ValueError('independent outcome requires explicit no hints')
            problem_key(p.get('problem_key'))
        else:
            if self.source != 'manual':
                raise ValueError('timer actions require manual evidence')
            identifier(p.get('interval_id'))
            expected = {'interval_id'}
            if self.kind == 'timer_started':
                expected.add('problem_key')
                problem_key(p.get('problem_key'))
            elif self.kind == 'timer_confirmed':
                expected |= {'started_at', 'ended_at'}
                p['started_at'] = timestamp(p.get('started_at'))
                p['ended_at'] = timestamp(p.get('ended_at'))
                if not p['started_at'] <= p['ended_at'] <= occurred:
                    raise ValueError('invalid confirmed interval bounds')
        if set(p) != expected:
            raise ValueError('unexpected or missing payload fields')
        # Reject non-JSON values and detach caller-owned dictionaries.
        p = json.loads(json.dumps(p, allow_nan=False))
        return Event(self.event_id, self.learner_id, occurred, recorded,
                     self.kind, self.source, p, self.supersedes, self.schema_version)

    def document(self):
        return asdict(self.normalized())


def active_events(events):
    superseded = {e.supersedes for e in events if e.supersedes is not None}
    return sorted((e for e in events if e.event_id not in superseded),
                  key=lambda e: (e.occurred_at, e.event_id))


def timer_state(events):
    """Replay explicit intervals; unconfirmed elapsed time remains unavailable."""
    intervals = {}
    # Starts/stops at equal timestamps are ordered by action, then stable ID.
    order = {'timer_started': 0, 'timer_stopped': 1, 'timer_confirmed': 2}
    for e in sorted(active_events(events), key=lambda e: (
            e.occurred_at, order.get(e.kind, 3), e.event_id)):
        if e.kind == 'attempt':
            continue
        key = e.payload['interval_id']
        if e.kind == 'timer_started':
            if key in intervals:
                raise ValueError('duplicate timer start')
            intervals[key] = {'interval_id': key, 'problem_key': e.payload['problem_key'],
                              'started_at': e.occurred_at, 'status': 'open',
                              'confirmed_seconds': None}
            continue
        state = intervals.get(key)
        if state is None:
            raise ValueError('timer action without start')
        if e.kind == 'timer_stopped':
            if state['status'] != 'open':
                raise ValueError('duplicate or misplaced timer stop')
            state.update(status='unconfirmed', stopped_at=e.occurred_at)
        else:
            if state['status'] != 'unconfirmed':
                raise ValueError('confirmation requires one stopped interval')
            start, end = e.payload['started_at'], e.payload['ended_at']
            if not state['started_at'] <= start <= end <= state['stopped_at']:
                raise ValueError('confirmation outside recorded actions')
            seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
            state.update(status='confirmed', confirmed_start=start, confirmed_end=end,
                         confirmed_seconds=seconds)
    return [intervals[k] for k in sorted(intervals)]


def confirmed_seconds(events):
    """Union elapsed intervals to avoid double-counting simultaneous timers."""
    spans = sorted((s['confirmed_start'], s['confirmed_end']) for s in timer_state(events)
                   if s['status'] == 'confirmed')
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
               for start, end in merged)
