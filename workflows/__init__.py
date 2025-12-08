"""Workflow orchestration using LangGraph."""

from .medical_qa_graph import MedicalQAWorkflow, create_workflow
from .image_qa_graph import ImageQAWorkflow, create_image_workflow, detect_input_type

__all__ = [
    'MedicalQAWorkflow', 
    'create_workflow',
    'ImageQAWorkflow',
    'create_image_workflow',
    'detect_input_type'
]

