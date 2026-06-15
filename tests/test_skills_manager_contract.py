import os
import tempfile
from unittest import mock

import pytest

from services.memory.skills import SkillsManager


def test_get_relevant_skills_contract():
    """Test that SkillsManager.get_relevant_skills matches agent_loop.py expectations."""
    with tempfile.TemporaryDirectory() as td:
        sm = SkillsManager(td)
        
        # Add some test skills
        test_skills = [
            {
                "name": "email_helper",
                "description": "Send emails",
                "when_to_use": "When you need to send an email",
                "tags": ["email", "communication"],
                "confidence": 0.9,
                "status": "published",
                "source": "user"
            },
            {
                "name": "file_search",
                "description": "Search files",
                "when_to_use": "When you need to find files",
                "tags": ["files", "search"],
                "confidence": 0.8,
                "status": "published",
                "source": "user"
            },
            {
                "name": "code_analyzer",
                "description": "Analyze code",
                "when_to_use": "When you need to analyze code",
                "tags": ["code", "analysis"],
                "confidence": 0.7,
                "status": "draft",
                "source": "user"
            }
        ]
        
        # Create skill directories and files
        for skill in test_skills:
            skill_dir = os.path.join(td, "skills", "general", skill["name"])
            os.makedirs(skill_dir, exist_ok=True)
            skill_file = os.path.join(skill_dir, "SKILL.md")
            content = f"""---
name: {skill['name']}
description: {skill['description']}
confidence: {skill['confidence']}
status: {skill['status']}
source: {skill['source']}
tags: {', '.join(skill['tags'])}
when_to_use: {skill['when_to_use']}
---

# {skill['name']}

{skill['description']}

## When to Use

{skill['when_to_use']}

## Procedure

1. Analyze the problem
2. Apply solution
"""
            with open(skill_file, 'w') as f:
                f.write(content)
        
        # Test contract: query with threshold and max_items
        results = sm.get_relevant_skills(
            query="send email",
            skills=test_skills,
            threshold=0.3,
            max_items=5,
            min_confidence=0.0
        )
        
        # Verify return type and structure
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            assert "name" in result
            assert "description" in result
            assert "when_to_use" in result
            assert "confidence" in result
            assert "status" in result
            assert "tags" in result
            assert isinstance(result["confidence"], (int, float))
            assert isinstance(result["tags"], list)
        
        # Verify sorting and filtering
        assert len(results) <= 5  # max_items respected
        for result in results:
            assert result["confidence"] >= 0.3  # threshold respected
        
        # Verify semantic relevance (email_helper should rank high)
        email_found = any(r["name"] == "email_helper" for r in results)
        assert email_found, "Expected email_helper to be found for 'send email' query"
        
        # Verify all returned skills are either published or draft
        for result in results:
            assert result["status"] in ("published", "draft")


def test_get_relevant_skills_empty():
    """Test edge case: empty skills list or empty query."""
    with tempfile.TemporaryDirectory() as td:
        sm = SkillsManager(td)
        
        # Empty skills list
        results = sm.get_relevant_skills("test query", skills=[], threshold=0.3, max_items=5)
        assert results == []
        
        # Empty query
        results = sm.get_relevant_skills("", skills=[{"name": "test", "description": "test", "when_to_use": "test"}], threshold=0.3, max_items=5)
        assert results == []
        
        # No skills pass threshold
        results = sm.get_relevant_skills("test query", skills=[{"name": "test", "description": "test", "when_to_use": "test"}], threshold=0.9, max_items=5)
        assert results == []


def test_get_relevant_skills_confidence_filter():
    """Test confidence filtering logic."""
    with tempfile.TemporaryDirectory() as td:
        sm = SkillsManager(td)
        
        test_skills = [
            {
                "name": "high_confidence",
                "description": "High confidence skill",
                "when_to_use": "Use when needed",
                "confidence": 0.9,
                "status": "published",
                "source": "user"
            },
            {
                "name": "low_confidence",
                "description": "Low confidence skill",
                "when_to_use": "Use when needed",
                "confidence": 0.2,
                "status": "draft",
                "source": "user"
            },
            {
                "name": "no_confidence",
                "description": "No confidence skill",
                "when_to_use": "Use when needed",
                "confidence": None,
                "status": "draft",
                "source": "user"
            }
        ]
        
        # Create skill files
        for skill in test_skills:
            skill_dir = os.path.join(td, "skills", "general", skill["name"])
            os.makedirs(skill_dir, exist_ok=True)
            skill_file = os.path.join(skill_dir, "SKILL.md")
            content = f"""---
name: {skill['name']}
description: {skill['description']}
confidence: {skill['confidence']}
status: {skill['status']}
source: {skill['source']}
when_to_use: {skill['when_to_use']}
---

# {skill['name']}

{skill['description']}

## When to Use

{skill['when_to_use']}
"""
            with open(skill_file, 'w') as f:
                f.write(content)
        
        # Test with min_confidence=0.5
        # Should include high_confidence (published), exclude low_confidence (draft < 0.5)
        # and no_confidence (draft, no confidence value → legacy behavior: allow)
        results = sm.get_relevant_skills(
            query="high confidence",  # Use a query that will match
            skills=test_skills,
            threshold=0.1,
            max_items=5,
            min_confidence=0.5
        )
        
        # Both high_confidence and no_confidence should pass (no_confidence is legacy behavior)
        assert len(results) == 2
        result_names = [r["name"] for r in results]
        assert "high_confidence" in result_names
        assert "no_confidence" in result_names
        assert "low_confidence" not in result_names