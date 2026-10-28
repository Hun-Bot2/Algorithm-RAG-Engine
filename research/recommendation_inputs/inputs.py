"""Source-preserving input views; no algorithm generation or missing-field inference."""
import re

HEAD = re.compile(r'^\s*(?:#{1,6}\s*)?(Problem Statement|Input(?: Format)?|Output(?: Format)?|Constraints|Examples?|Notes?|Sample Input(?:\s+\d+)?|Sample Output(?:\s+\d+)?)\s*:?\s*$', re.I)


def section_inputs(text):
    sections = {key: [] for key in ('description', 'input', 'output', 'constraints', 'examples', 'notes')}
    counts = {key: 0 for key in sections}
    section = 'description'
    for line in text.splitlines():
        match = HEAD.fullmatch(line)
        if match:
            label = match[1].lower()
            if label.startswith(('example', 'sample')):
                section = 'examples'
            elif label.startswith('note'):
                section = 'notes'
            elif section not in {'examples', 'notes'}:
                section = 'description' if label == 'problem statement' else label.split()[0]
                counts[section] += 1
            continue
        sections[section].append(line)
    sections = {key: '\n'.join(value).strip() for key, value in sections.items()}
    issues = [f'duplicate_{key}_heading' for key in ('input', 'output', 'constraints') if counts[key] > 1]
    issues += [f'missing_{key}' for key in ('description', 'input', 'output') if not sections[key]]
    present = {key: bool(value) for key, value in sections.items()}
    task = '\n\n'.join(f'{key.upper()}\n{sections[key]}' for key in ('description', 'input', 'output', 'constraints') if sections[key])
    io = '\n\n'.join(f'{key.upper()}\n{sections[key]}' for key in ('input', 'output', 'constraints') if sections[key])
    return {'sections': sections, 'present': present, 'issues': issues, 'task_text': task, 'io_text': io}


def curriculum_inputs(problem, annotation):
    rationale = annotation.get('concept_rationale', '').strip()
    concepts = problem.get('concepts')
    if not rationale or not isinstance(concepts, list) or not concepts or not all(isinstance(c, str) and c for c in concepts):
        raise ValueError('source rationale and concepts required')
    return {'title': annotation['title'], 'idea_rationale': rationale,
            'idea_structured': f"CORE IDEA\n{rationale}\n\nCONCEPTS\n" + '\n'.join(concepts),
            'missing_strategy_fields': ['explicit state/invariant', 'correctness argument', 'complexity analysis'],
            'source_review': dict(problem.get('review') or {})}


def token_chunks(token_ids, window=254):
    if not 1 <= window <= 254 or any(not ids for ids in token_ids):
        raise ValueError('nonempty inputs and bounded original-token windows required')
    return [(i, ids[start:start+window]) for i, ids in enumerate(token_ids) for start in range(0, len(ids), window)]
