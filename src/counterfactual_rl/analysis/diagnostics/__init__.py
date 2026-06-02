"""Offline CCE diagnostics — interrogate the consequence score on trained models.

Answers three questions without any new training:
  Q1  Is there exploitable structure?      (spread of CCE scores)
  Q2  Is CCE just TD error in disguise?     (rank correlation CCE vs |TD|)
  Q3  When they disagree, who is right?     (each vs MCTS-200 ground truth)
"""
