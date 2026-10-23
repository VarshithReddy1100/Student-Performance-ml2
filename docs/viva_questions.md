# Viva Questions and Short Answers

1. **What are the two prediction tasks?** Regression predicts numeric G3; classification maps G3 to Low (0–9), Medium (10–14), or High (15–20).
2. **Why are G1 and G2 excluded?** They are prior grades strongly associated with G3. Excluding them prevents target leakage and tests prediction from non-grade information.
3. **What is target leakage?** Information unavailable at prediction time, or derived from the target, enters the inputs and produces unrealistically optimistic evaluation.
4. **What is a train/test split?** Training data estimates model parameters; held-out test data estimates generalization after choices are made.
5. **Why use K-fold cross-validation?** It repeats train/validation evaluation across folds, making model comparisons less dependent on one partition.
6. **What is overfitting?** The model fits training-specific noise; training performance is much better than validation performance.
7. **What is underfitting?** The model is too simple or constrained to capture patterns; both training and validation performance are weak.
8. **Explain bias-variance tradeoff.** High bias comes from restrictive assumptions; high variance comes from sensitivity to training samples. Generalization requires balancing them.
9. **What does a learning curve show?** Performance against training-set size. It helps diagnose whether more data may help and whether there is a train-validation gap.
10. **Why report MAE and RMSE?** Both are in grade units; RMSE penalizes large errors more than MAE.
11. **What does R² mean?** Fraction of target variance explained relative to a mean baseline; it can be negative on poor predictions.
12. **What is Ridge regularization?** L2 penalty on coefficient squares; it shrinks coefficients and helps control variance.
13. **What is Lasso regularization?** L1 penalty on absolute coefficient values; some coefficients can become exactly zero.
14. **Simple vs multiple linear regression?** Simple regression has one predictor; multiple regression has multiple predictors. This project's full-feature linear pipeline is multiple regression.
15. **How is polynomial regression still linear regression?** It is linear in learned coefficients, though the input features are expanded into powers/interactions.
16. **What is an SVM hyperplane?** A decision boundary separating classes in feature space.
17. **What are support vectors?** Training examples nearest the boundary that determine its position.
18. **What does C control in SVM?** The tradeoff between a wider margin and penalties for training violations; high C fits training examples more strictly.
19. **What does gamma control for an RBF SVM?** The influence radius of individual samples; high gamma makes a more local, potentially complex boundary.
20. **Why use kernels?** Kernels let SVMs represent nonlinear boundaries without explicitly computing a high-dimensional mapping.
21. **How does a decision tree split?** It selects a feature/threshold that improves a criterion, such as reducing Gini impurity or squared error.
22. **What is bootstrap sampling?** Sampling rows with replacement to train each forest tree.
23. **Why can a random forest generalize better than one tree?** Averaging diverse trees can reduce variance while retaining nonlinear relationships.
24. **What is a confusion matrix?** A table of actual versus predicted classes that shows which categories are confused.
25. **Why use macro-F1?** It computes F1 per class and averages equally, so a large class cannot hide poor minority-class performance.
26. **What are residuals?** Actual target minus prediction. Patterns in residual plots can indicate model misspecification or non-constant error variance.
27. **Why put preprocessing inside a pipeline?** Each CV training fold fits its own imputer, encoder and scaler, preventing preprocessing leakage.
28. **Does feature importance prove causation?** No. It describes model reliance, not causal effects.
29. **What are key dataset limitations?** Small, dated, observational, self-reported, and potentially nonrepresentative data.
30. **Should predictions be used to make decisions about real students?** No. This coursework tool is not a validated or fair high-impact assessment system.
