# XGBoost-VTE-Prediction
Predicted patients whether they have venous thrombosis embolism. Applying XGBoost

XGBoost stands for "eXtreme Gradient Boosting" and is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It is an open-source tool that has gained significant popularity in machine learning due to its performance and accuracy. The core XGBoost algorithm is based on the gradient boosting framework, which is a machine learning algorithm used for regression and classification problems.

XGBoost excels in performance, consistently achieving high accuracy in diverse machine learning tasks. It speeds up learning with parallel processing, making it ideal for large datasets. Distinct from conventional gradient boosting, XGBoost integrates L1 and L2 regularization to curb overfitting. It adeptly manages missing data, reducing preprocessing needs. With its unique "depth-first" tree approach and "max_depth" pruning, it crafts more efficient models than other gradient boosters.

In this project, I aimed to enhance XGBoost's performance and, as a result, employed PCA. Incorporating PCA streamlined the efficiency by condensing data dimensions, mitigating noise, and curbing overfitting. While it facilitated data visualization and the handling of multicollinearity, it risked making features less transparent. To ensure consistent model performance, I further implemented K-fold cross-validation.
