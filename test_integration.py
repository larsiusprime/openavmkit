#!/usr/bin/env python
"""
Integration test for LayeredCompBaggingModel
"""

print("Testing LayeredCompBaggingModel integration...")

# Test 1: Import from utilities.modeling
from openavmkit.utilities.modeling import (
    GarbageModel, AverageModel, NaiveAreaModel, LocalAreaModel,
    PassThroughModel, GWRModel, MRAModel, XGBoostModel,
    LightGBMModel, CatBoostModel, LayeredCompBaggingModel,
    MultiMRAModel, GroundTruthModel, SpatialLagModel,
    LandSLICEModel, greedy_forward_loocv, TreeBasedCategoricalData
)
print("✓ All model classes imported successfully from utilities.modeling")

# Test 2: Import from main modeling module
from openavmkit.modeling import LayeredCompBaggingModel
print("✓ LayeredCompBaggingModel imported successfully from main modeling")

# Test 3: Verify layeredcompmodel package is available
import layeredcompmodel
print("✓ layeredcompmodel package available")

# Test 4: Quick instantiation test
from layeredcompmodel import LayeredCompBaggingModel as LCBM
lcb = LCBM(tree_count=5)
wrapped = LayeredCompBaggingModel(lcb)
print("✓ LayeredCompBaggingModel wrapper successfully instantiated")

# Test 5: Verify the wrapped model has the expected structure
assert hasattr(wrapped, 'model'), "Wrapper should have 'model' attribute"
assert isinstance(wrapped.model, LCBM), "Wrapped model should be a LayeredCompBaggingModel instance"
print("✓ Wrapper structure verified")

print("\n✅ Integration complete and verified!")
print("\nLayeredCompBaggingModel is now available in the openavmkit toolkit!")

