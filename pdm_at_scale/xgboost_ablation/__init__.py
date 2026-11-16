"""Compare the pipelines against XGBoost on BAAI embeddings (the ablation).

The stages run in this order, each as its own script:

1. ``build_dataset``: label the WAGO and SMC products.
2. ``embed``: embed the product texts with the BAAI model.
3. ``split``: split whole families into folds, or the products randomly.
4. ``train_xgboost``: train the binary classifier and the ordinal regressor.
5. ``evaluate``: score XGBoost and the pipelines against the labels.
"""
