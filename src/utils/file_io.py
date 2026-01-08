"""
File I/O utilities
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read JSONL file
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue
    
    logger.info(f"Read {len(data)} records from {file_path}")
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """
    Write data to JSONL file
    
    Args:
        data: List of dictionaries
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Wrote {len(data)} records to {file_path}")


def read_json(file_path: Path) -> Dict[str, Any]:
    """
    Read JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Read JSON from {file_path}")
    return data


def write_json(data: Dict[str, Any], file_path: Path):
    """
    Write data to JSON file
    
    Args:
        data: Dictionary
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Wrote JSON to {file_path}")
