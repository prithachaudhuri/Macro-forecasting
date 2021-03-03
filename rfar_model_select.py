def rfar_model_select(indpro_insample, indpro_oos, maxlags, cv, ss, tscv):
    """
    Inputs:
    indpro_insample: insample data for estimation
    indpro_oos: out-of-sample data for final evaluation
    maxlags: lag length for AR models
    cv: type of cross validation selected
    ss: indices for K-fold CV
    tscv: indices for pseudo out-of-sample CV
    
    Output:
    rf_opt_pred: predictions for optimal Random Forest AR model
    """
    
    indpro_insample_df = pd.DataFrame(indpro_insample)
    indpro_oos_df = pd.DataFrame(indpro_oos)
    
    if cv == 'kfold':
        rf_rmse = np.zeros((len(maxlags), ss.n_splits))
        
        for i, lag in enumerate(maxlags):
            print('Lags: ', lag)

            # for each lag length, create lags of the variable. Split data into y and X
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Start cross-validation fold here 
            fold = 0

            for train_idx, test_idx in ss.split(indpro_insample):
                X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                y_train, y_test = y[train_idx], y[test_idx]

                # Use out-of-bag observations to get the optimal number of trees
                min_estimators = 15
                max_estimators = 175
                estimators = list(range(min_estimators, max_estimators+1))

                error_rate = np.zeros(((len(estimators)), 2))

                rf_model = RandomForestRegressor(criterion='mse', 
                                                  max_features=0.33, 
                                                  oob_score=True, 
                                                  random_state=123, 
                                                  n_jobs=15)

                for j,n in enumerate(estimators):
                    rf_model.set_params(n_estimators=n)
                    rf_model.fit(X_train, y_train)
                    error_rate[j, 0] = n
                    error_rate[j, 1] = rf_model.oob_score_

                min_oob_error_idx = error_rate[:,1].argmin()
                opt_n_trees = error_rate[min_oob_error_idx, 0]
                print('Number of trees for lag ', lag, 'and fold ', fold+1, ' is ', opt_n_trees)

                # Using optimal number of trees for the lag length, run RF model and calculate RMSE
                rf_model.set_params(n_estimators=int(opt_n_trees))
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)

                print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, rf_pred)))
                rf_rmse[i, fold] = sqrt(mean_squared_error(y_test, rf_pred))
                fold += 1
            
        rf_opt_lag_idx = np.mean(rf_rmse, axis=1).argmin()
        rf_opt_lag = maxlags[rf_opt_lag_idx]

        for l in range(rf_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:rf_opt_lag+1].fillna(0)
        y_oos = indpro_oos_df['INDPRO']
        X_oos = indpro_oos_df.iloc[:,1:rf_opt_lag+1].fillna(0)

        min_estimators = 15
        max_estimators = 175
        estimators = list(range(min_estimators, max_estimators+1))

        error_rate = np.zeros(((len(estimators)), 2))

        rf_model = RandomForestRegressor(criterion='mse', 
                                          max_features=0.33, 
                                          oob_score=True, 
                                          random_state=123, 
                                          n_jobs=15)

        for j,n in enumerate(estimators):
            rf_model.set_params(n_estimators=n)
            rf_model.fit(X_insample, y_insample)
            error_rate[j, 0] = n
            error_rate[j, 1] = rf_model.oob_score_

        min_oob_error_idx = error_rate[:,1].argmin()
        opt_n_trees = error_rate[min_oob_error_idx, 0]
        print('Optimal number of trees for RF K-fold is ', opt_n_trees)
        print('Optimal lag for RF K-fold is ', rf_opt_lag)

        # Using optimal number of trees for the optimal lag length, run RF model and calculate RMSE
        rf_model.set_params(n_estimators=int(opt_n_trees))
        rf_model.fit(X_insample, y_insample)
        rf_pred_oos = rf_model.predict(X_oos)
        print('Out-of-sample RMSE for best RF K-fold model ', sqrt(mean_squared_error(y_oos, rf_pred_oos)))
        
        return rf_pred_oos
    
    if cv == 'poos':
        rf_rmse = np.zeros((len(maxlags), tscv.n_splits))
        
        for i, lag in enumerate(maxlags):
            print('Lags: ', lag)

            # for each lag length, create lags of the variable. Split data into y and X
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Start cross-validation fold here 
            fold = 0

            for train_idx, test_idx in tscv.split(indpro_insample):
                X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                y_train, y_test = y[train_idx], y[test_idx]

                # Use out-of-bag observations to get the optimal number of trees
                min_estimators = 15
                max_estimators = 175
                estimators = list(range(min_estimators, max_estimators+1))

                error_rate = np.zeros(((len(estimators)), 2))

                rf_model = RandomForestRegressor(criterion='mse', 
                                                  max_features=0.33, 
                                                  oob_score=True, 
                                                  random_state=123, 
                                                  n_jobs=15)

                for j,n in enumerate(estimators):
                    rf_model.set_params(n_estimators=n)
                    rf_model.fit(X_train, y_train)
                    error_rate[j, 0] = n
                    error_rate[j, 1] = rf_model.oob_score_

                min_oob_error_idx = error_rate[:,1].argmin()
                opt_n_trees = error_rate[min_oob_error_idx, 0]
                print('Number of trees for lag ', lag, 'and fold ', fold+1, ' is ', opt_n_trees)

                # Using optimal number of trees for the lag length, run RF model and calculate RMSE
                rf_model.set_params(n_estimators=int(opt_n_trees))
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)

                print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, rf_pred)))
                rf_rmse[i, fold] = sqrt(mean_squared_error(y_test, rf_pred))
                fold += 1
            
        rf_opt_lag_idx = np.mean(rf_rmse, axis=1).argmin()
        rf_opt_lag = maxlags[rf_opt_lag_idx]

        for l in range(rf_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:rf_opt_lag+1].fillna(0)
        y_oos = indpro_oos_df['INDPRO']
        X_oos = indpro_oos_df.iloc[:,1:rf_opt_lag+1].fillna(0)

        min_estimators = 15
        max_estimators = 175
        estimators = list(range(min_estimators, max_estimators+1))

        error_rate = np.zeros(((len(estimators)), 2))

        rf_model = RandomForestRegressor(criterion='mse', 
                                          max_features=0.33, 
                                          oob_score=True, 
                                          random_state=123, 
                                          n_jobs=15)

        for j,n in enumerate(estimators):
            rf_model.set_params(n_estimators=n)
            rf_model.fit(X_insample, y_insample)
            error_rate[j, 0] = n
            error_rate[j, 1] = rf_model.oob_score_

        min_oob_error_idx = error_rate[:,1].argmin()
        opt_n_trees = error_rate[min_oob_error_idx, 0]
        print('Optimal number of trees for RF POOS is ', opt_n_trees)
        print('Optimal lag for RF POOS is ', rf_opt_lag)

        # Using optimal number of trees for the optimal lag length, run RF model and calculate RMSE
        rf_model.set_params(n_estimators=int(opt_n_trees))
        rf_model.fit(X_insample, y_insample)
        rf_pred_oos = rf_model.predict(X_oos)
        print('Out-of-sample RMSE for best RF POOS model ', sqrt(mean_squared_error(y_oos, rf_pred_oos)))
        
        return rf_pred_oos
        
    