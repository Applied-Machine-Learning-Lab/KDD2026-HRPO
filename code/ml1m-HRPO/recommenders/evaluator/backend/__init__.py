try:
    from evaluator.backend.cpp.uni_evaluator import UniEvaluator
    print("Evaluate model with cpp")
except Exception:
    from evaluator.backend.python.uni_evaluator import UniEvaluator
    print("Evaluate model with python")
