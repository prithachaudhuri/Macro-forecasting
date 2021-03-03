def krrar_model_select(indpro_insample, indpro_oos, maxlags, alphas, gammas, cv, ss, tscv):
    """
    Inputs:
    indpro_insample: insample data for estimation
    indpro_oos: out-of-sample data for final evaluation
    maxlags: lag length for AR models
    alphas: Ridge penalty parameters
    gammas: RBF kernel tuning parameters
    cv: type of cross validation selected
    ss: indices for K-fold CV
    tscv: indices for pseudo out-of-sample CV
    
    Output:
    ridge_opt_pred: predictions for optimal Ridge AR model
    """
    
    indpro_insample_df = pd.DataFrame(indpro_insample)
    indpro_oos_df = pd.DataFrame(indpro_oos)
    
    if cv == 'kfold':
        krr_lags = np.zeros((len(maxlags), 4))

        for i,lag in enumerate(maxlags):
            print(' ')
            print('Lag: ', lag)

            ## For each lag length, create X and y datasets for ridge regression
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Choose hyperparameter: alpha for Ridge regression penalty, gamma for RBF tuning parameter
            krr_folds = np.zeros((len(alphas)*len(gammas), ss.n_splits+2))
            row = 0

            for j, a in enumerate(alphas):
                print(' ')
                print('Alpha: ', a)

                for k, g in enumerate(gammas):
                    fold=0
                    print('\n Gamma:', g)

                    krr_model = KernelRidge(alpha=a, kernel='rbf', gamma=g)

                    for train_idx, test_idx in ss.split(indpro_insample):
                        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                        y_train, y_test = y[train_idx], y[test_idx]

                        krr_model.fit(X_train, y_train) # fit ridge on training data
                        krr_pred = krr_model.predict(X_test) # predict using test data

                        print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, krr_pred)))

                        krr_folds[row, 0] = a
                        krr_folds[row, 1] = g
                        krr_folds[row, fold+2] = sqrt(mean_squared_error(y_test, krr_pred))
                        fold += 1

                    row+= 1
                    
            krr_para_min_idx = np.mean(krr_folds[:,2:6], axis=1).argmin()
            print(' ')

            krr_lags[i, 0] = lag
            krr_lags[i, 1] = krr_folds[krr_para_min_idx,0]
            krr_lags[i, 2] = krr_folds[krr_para_min_idx,1]
            krr_lags[i, 3] = np.mean(krr_folds[:,2:6], axis=1)[krr_para_min_idx]


        ## Optimal KRR model
        krr_opt_lag = int(krr_lags[krr_lags[:,3].argmin(),0])
        krr_opt_alpha = krr_lags[krr_lags[:,3].argmin(),1]
        krr_opt_gamma = krr_lags[krr_lags[:,3].argmin(),2]
        print('Optimal lag for K-fold CV is: ', krr_opt_lag)
        print('Optimal alpha for K-fold CV is: ', krr_opt_alpha)
        print('Optimal gamma for K-fold CV is: ', krr_opt_gamma)

        for l in range(krr_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        y_oos = indpro_oos_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:krr_opt_lag+1].fillna(0)
        X_oos = indpro_oos_df.iloc[:,1:krr_opt_lag+1].fillna(0)

        krr_opt = KernelRidge(alpha=krr_opt_alpha, kernel='rbf', gamma=krr_opt_gamma)
        krr_opt.fit(X_insample, y_insample)
        krr_opt_pred = krr_opt.predict(X_oos)
        print('Out-of-sample RMSE for best KRR K-fold model is ', sqrt(mean_squared_error(y_oos, krr_opt_pred)))
        
        return krr_opt_pred
    
    elif cv == 'poos':
        krr_lags = np.zeros((len(maxlags), 4))

        for i,lag in enumerate(maxlags):
            print(' ')
            print('Lag: ', lag)

            ## For each lag length, create X and y datasets for ridge regression
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Choose hyperparameter: alpha for Ridge regression penalty, gamma for RBF tuning parameter
            krr_folds = np.zeros((len(alphas)*len(gammas), tscv.n_splits+2))
            row = 0

            for j, a in enumerate(alphas):
                print(' ')
                print('Alpha: ', a)

                for k, g in enumerate(gammas):
                    fold=0
                    print('\n Gamma:', g)

                    krr_model = KernelRidge(alpha=a, kernel='rbf', gamma=g)

                    for train_idx, test_idx in tscv.split(indpro_insample):
                        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                        y_train, y_test = y[train_idx], y[test_idx]

                        krr_model.fit(X_train, y_train) # fit ridge on training data
                        krr_pred = krr_model.predict(X_test) # predict using test data

                        print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, krr_pred)))

                        krr_folds[row, 0] = a
                        krr_folds[row, 1] = g
                        krr_folds[row, fold+2] = sqrt(mean_squared_error(y_test, krr_pred))
                        fold += 1

                    row+= 1
                    
            krr_para_min_idx = np.mean(krr_folds[:,2:6], axis=1).argmin()
            
            krr_lags[i, 0] = lag
            krr_lags[i, 1] = krr_folds[krr_para_min_idx,0]
            krr_lags[i, 2] = krr_folds[krr_para_min_idx,1]
            krr_lags[i, 3] = np.mean(krr_folds[:,2:6], axis=1)[krr_para_min_idx]


        ## Optimal KRR model
        krr_opt_lag = int(krr_lags[krr_lags[:,3].argmin(),0])
        krr_opt_alpha = krr_lags[krr_lags[:,3].argmin(),1]
        krr_opt_gamma = krr_lags[krr_lags[:,3].argmin(),2]
        print(' ')
        print('Optimal lag for POOS CV is: ', krr_opt_lag)
        print('Optimal alpha for POOS CV is: ', krr_opt_alpha)
        print('Optimal gamma for POOS CV is: ', krr_opt_gamma)

        for l in range(krr_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        y_oos = indpro_oos_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:krr_opt_lag+1].fillna(0)
        X_oos = indpro_oos_df.iloc[:,1:krr_opt_lag+1].fillna(0)

        krr_opt = KernelRidge(alpha=krr_opt_alpha, kernel='rbf', gamma=krr_opt_gamma)
        krr_opt.fit(X_insample, y_insample)
        krr_opt_pred = krr_opt.predict(X_oos)
        print('Out-of-sample RMSE for best KRR POOS model is ', sqrt(mean_squared_error(y_oos, krr_opt_pred)))
        
        return krr_opt_pred