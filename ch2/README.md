Sample Output from the standalone py script:


(dl) ciaran@ciaran-XPS-13-9360:ch2$ python ./california_house_price_prediction.py 
Linear Regression RMSE:             69050
Decision Tree Regression RMSE:      0

Results for:  K-Fold Cross Validation with Decision Tree
  Scores		: [68434 66010 71781 69592 69127 76461 66551 69908 70605 69835]
  Mean			: 69830
  Standard Deviation	: 2761

Results for:  K-Fold Cross Validation with Linear Regression
  Scores		: [67450 67329 68361 74639 68314 71628 65361 68571 72476 68098]
  Mean			: 69223
  Standard Deviation	: 2657

Forest training RMSE	: 22155

Results for:  K-Fold Cross Validation with Random Forest
  Scores		: [50650 48686 50850 52929 51467 55693 51497 51992 53095 51734]
  Mean			: 51859
  Standard Deviation	: 1739

Winning model was: RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)


Starting grid search for best hyperparameters...
Best params found:  {'max_features': 8, 'n_estimators': 70}

Final score:             46856

(dl) ciaran@ciaran-XPS-13-9360:ch2$ 

