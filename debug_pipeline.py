#!/usr/bin/env python3
"""
Script de debug para el pipeline LSA
"""

import sys
import os

print("=== DEBUG PIPELINE LSA ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

print("\n=== TESTING BASIC IMPORTS ===")
try:
    import cv2
    print("✓ cv2 imported")
except Exception as e:
    print(f"✗ cv2 error: {e}")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy error: {e}")

try:
    import mediapipe as mp
    print("✓ mediapipe imported")
except Exception as e:
    print(f"✗ mediapipe error: {e}")

try:
    from sklearn.cluster import KMeans
    print("✓ sklearn imported")
except Exception as e:
    print(f"✗ sklearn error: {e}")

print("\n=== TESTING PROJECT IMPORTS ===")
try:
    from frame_extractor import FrameExtractor
    print("✓ FrameExtractor imported")
except Exception as e:
    print(f"✗ FrameExtractor error: {e}")
    import traceback
    traceback.print_exc()

try:
    from landmark_extractor import LandmarkExtractor
    print("✓ LandmarkExtractor imported")
except Exception as e:
    print(f"✗ LandmarkExtractor error: {e}")
    import traceback
    traceback.print_exc()

try:
    from gloss_to_text import GlossToTextConverter
    print("✓ GlossToTextConverter imported")
except Exception as e:
    print(f"✗ GlossToTextConverter error: {e}")
    import traceback
    traceback.print_exc()

try:
    from model.sequence_model import LSASequenceModel
    print("✓ LSASequenceModel imported")
except Exception as e:
    print(f"✗ LSASequenceModel error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TESTING PIPELINE IMPORT ===")
try:
    from lsa_pipeline import LSAPipeline
    print("✓ LSAPipeline imported")
    
    # Try to create pipeline instance
    pipeline = LSAPipeline()
    print("✓ LSAPipeline instance created")
    
except Exception as e:
    print(f"✗ LSAPipeline error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG COMPLETED ===")
