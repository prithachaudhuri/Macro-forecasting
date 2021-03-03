def ridgear_model_select(indpro_insample, indpro_oos, maxlags, alphas, cv, ss, tscv):
    """
    Inputs:
    indpro_insample: insample data for estimation
    indpro_oos: out-of-sample data for final evaluation
    maxlags: lag length for AR models
    alphas: Ridge penalty parameters
    cv: type of cross validation selected
    ss: indices for K-fold CV
    tscv: indices for pseudo out-of-sample CV
    
    Output:
    ridge_opt_pred: predictions for optimal Ridge AR model
    """
    
    indpro_insample_df = pd.DataFrame(indpro_insample)
    
    if cv == 'kfold':
#         alphas = np.linspace(0,4,6)
        ridge_kfold_lags = np.zeros((len(maxlags), 3))

        for i,lag in enumerate(maxlags):
            print(' ')
            print('Lag: ', lag)

            ## For each lag length, create X and y datasets for ridge regression
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Choose hyperparameter: alpha for Ridge regression
            ridge_kfold = np.zeros((len(alphas), ss.n_splits))

            for j, a in enumerate(alphas):
                fold=0
                print(' ')
                print('Alpha: ', a)
                ridge_model = Ridge(a, normalize=True)
                ridge_model.set_params(alpha=a)

                for train_idx, test_idx in ss.split(indpro_insample):
                    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                    y_train, y_test = y[train_idx], y[test_idx]

                    ridge_model.fit(X_train, y_train) # fit ridge on training data
                    ridge_pred = ridge_model.predict(X_test) # predict using test data

                    print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, ridge_pred)))

                    ridge_kfold[j, fold] = sqrt(mean_squared_error(y_test, ridge_pred))
                    fold += 1

            kfold_alpha_min_idx = np.mean(ridge_kfold, axis=1).argmin()
            kfold_alpha_min = alphas[kfold_alpha_min_idx]
            print(' ')
            print('Optimal alpha for K-fold CV is: ', kfold_alpha_min)

            ridge_kfold_lags[i, 0] = lag
            ridge_kfold_lags[i, 1] = kfold_alpha_min
            ridge_kfold_lags[i, 2] = np.mean(ridge_kfold, axis=1)[kfold_alpha_min_idx]


        ## Optimal Ridge model
        kfold_opt_lag = int(ridge_kfold_lags[ridge_kfold_lags[:,2].argmin(),0])
        kfold_opt_alpha = ridge_kfold_lags[ridge_kfold_lags[:,2].argmin(),1]
        print('Optimal lag for K-fold CV is: ', kfold_opt_lag)

        indpro_insample_df = pd.DataFrame(indpro_insample)
        indpro_oos_df = pd.DataFrame(indpro_oos)
        for l in range(kfold_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        y_oos = indpro_oos_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:kfold_opt_lag+1].fillna(0)
        X_oos = indpro_oos_df.iloc[:,1:kfold_opt_lag+1].fillna(0)

        ridge_opt = Ridge(alpha=kfold_opt_alpha, normalize=True)
        ridge_opt.fit(X_insample, y_insample)
        ridge_opt_pred = ridge_opt.predict(X_oos)
        print('Out-of-sample RMSE for best Ridge K-fold model is ', sqrt(mean_squared_error(y_oos, ridge_opt_pred)))
        
        return ridge_opt_pred
    
    elif cv == 'poos':
#         alphas = np.linspace(0,4,6)
        ridge_poos_lags = np.zeros((len(maxlags), 3))

        for i,lag in enumerate(maxlags):
            print(' ')
            print('Lag: ', lag)

            ## For each lag length, create X and y datasets for ridge regression
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Choose hyperparameter: alpha for Ridge regression
            ridge_poos = np.zeros((len(alphas), tscv.n_splits))

            for j, a in enumerate(alphas):
                fold=0
                print(' ')
                print('Alpha: ', a)
                ridge_model = Ridge(a, normalize=True)
                ridge_model.set_params(alpha=a)

                for train_idx, test_idx in tscv.split(indpro_insample):
                    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                    y_train, y_test = y[train_idx], y[test_idx]

                    ridge_model.fit(X_train, y_train) # fit ridge on training data
                    ridge_pred = ridge_model.predict(X_test) # predict using test data

                    print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, ridge_pred)))

                    ridge_poos[j, fold] = sqrt(mean_squared_error(y_test, ridge_pred))
                    fold += 1

            poos_alpha_min_idx = np.mean(ridge_poos, axis=1).argmin()
            poos_alpha_min = alphas[poos_alpha_min_idx]
            print(' ')
            print('Optimal alpha for POOS CV is: ', poos_alpha_min)

            ridge_poos_lags[i, 0] = lag
            ridge_poos_lags[i, 1] = poos_alpha_min
            ridge_poos_lags[i, 2] = np.mean(ridge_poos, axis=1)[poos_alpha_min_idx]


        ## Optimal Ridge model
        poos_opt_lag = int(ridge_poos_lags[ridge_poos_lags[:,2].argmin(),0])
        poos_opt_alpha = ridge_poos_lags[ridge_poos_lags[:,2].argmin(),1]
        print('Optimal lag for POOS CV is: ', poos_opt_lag)

        indpro_insample_df = pd.DataFrame(indpro_insample)
        indpro_oos_df = pd.DataFrame(indpro_oos)
        for l in range(poos_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        y_oos = indpro_oos_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:poos_opt_lag+1].fillna(0)
        X_oos = indpro_oos_df.iloc[:,1:poos_opt_lag+1].fillna(0)

        ridge_opt = Ridge(alpha=poos_opt_alpha, normalize=True)
        ridge_opt.fit(X_insample, y_insample)
        ridge_opt_pred = ridge_opt.predict(X_oos)
        print('Out-of-sample RMSE for best Ridge POOS model is ', sqrt(mean_squared_error(y_oos, ridge_opt_pred)))
        
        return ridge_opt_pred
