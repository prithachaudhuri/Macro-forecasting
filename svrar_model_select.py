def svrar_model_select(indpro_insample, indpro_oos, maxlags, alphas, gammas, epsilons, kernel, cv, ss, tscv):
    """
    Inputs:
    indpro_insample: insample data for estimation
    indpro_oos: out-of-sample data for final evaluation
    maxlags: lag length for AR models
    alphas: Ridge penalty parameters
    gammas: RBF kernel tuning parameters
    epsilons: size of insensitivity tube
    kernel: type of kernel for SVR, linear or RBF
    cv: type of cross validation selected
    ss: indices for K-fold CV
    tscv: indices for pseudo out-of-sample CV
    
    Output:
    svr_opt_pred: predictions for optimal Ridge AR model
    """
    
    indpro_insample_df = pd.DataFrame(indpro_insample)
    indpro_oos_df = pd.DataFrame(indpro_oos)
    
    if cv == 'kfold':
        svr_lags = np.zeros((len(maxlags), 5))

        for i,lag in enumerate(maxlags):
            print(' ')
            print('Lag: ', lag)

            ## For each lag length, create X and y datasets for svr regression
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Choose hyperparameter: alpha for Ridge regression penalty, gamma for RBF tuning parameter, epsilon for loss function
            svr_folds = np.zeros((len(alphas)*len(gammas)*len(epsilons), ss.n_splits+3))
            row = 0

            for j, a in enumerate(alphas):
                print(' ')
                print('Alpha: ', a)

                for k, g in enumerate(gammas):
                    print('\n Gamma:', g)

                    for m, e in enumerate(epsilons):
                        fold=0
                        print('\n Epsilon:', e)

                        svr_model = SVR(kernel=kernel, C=a, gamma=g, epsilon=e)

                        for train_idx, test_idx in ss.split(indpro_insample):
                            X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                            y_train, y_test = y[train_idx], y[test_idx]

                            svr_model.fit(X_train, y_train) # fit ridge on training data
                            svr_pred = svr_model.predict(X_test) # predict using test data

                            print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, svr_pred)))

                            svr_folds[row, 0] = a
                            svr_folds[row, 1] = g
                            svr_folds[row, 2] = e
                            svr_folds[row, fold+3] = sqrt(mean_squared_error(y_test, svr_pred))
                            fold += 1

                        row+= 1
                    
            svr_para_min_idx = np.mean(svr_folds[:,3:7], axis=1).argmin()
            print(' ')

            svr_lags[i, 0] = lag
            svr_lags[i, 1] = svr_folds[svr_para_min_idx,0]
            svr_lags[i, 2] = svr_folds[svr_para_min_idx,1]
            svr_lags[i, 3] = svr_folds[svr_para_min_idx,2]
            svr_lags[i, 4] = np.mean(svr_folds[:,3:7], axis=1)[svr_para_min_idx]


        ## Optimal SVR model
        svr_opt_lag = int(svr_lags[svr_lags[:,4].argmin(),0])
        svr_opt_alpha = svr_lags[svr_lags[:,4].argmin(),1]
        svr_opt_gamma = svr_lags[svr_lags[:,4].argmin(),2]
        svr_opt_epsilon = svr_lags[svr_lags[:,4].argmin(),3]
        print('Optimal lag for K-fold CV is: ', svr_opt_lag)
        print('Optimal alpha for K-fold CV is: ', svr_opt_alpha)
        print('Optimal gamma for K-fold CV is: ', svr_opt_gamma)
        print('Optimal gamma for K-fold CV is: ', svr_opt_epsilon)

        for l in range(svr_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        y_oos = indpro_oos_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:svr_opt_lag+1].fillna(0)
        X_oos = indpro_oos_df.iloc[:,1:svr_opt_lag+1].fillna(0)

        svr_opt = SVR(kernel=kernel, C=svr_opt_alpha, gamma=svr_opt_gamma, epsilon=svr_opt_epsilon)
        svr_opt.fit(X_insample, y_insample)
        svr_opt_pred = svr_opt.predict(X_oos)
        print('Out-of-sample RMSE for best SVR K-fold model is ', sqrt(mean_squared_error(y_oos, svr_opt_pred)))
        
        return svr_opt_pred
    
    elif cv == 'poos':
        svr_lags = np.zeros((len(maxlags), 5))

        for i,lag in enumerate(maxlags):
            print(' ')
            print('Lag: ', lag)

            ## For each lag length, create X and y datasets for svr regression
            for l in range(lag):
                indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)

            y = indpro_insample_df['INDPRO']
            X = indpro_insample_df.iloc[:,1:lag+1].fillna(0)

            ## Choose hyperparameter: alpha for Ridge regression penalty, gamma for RBF tuning parameter, epsilon for loss function
            svr_folds = np.zeros((len(alphas)*len(gammas)*len(epsilons), tscv.n_splits+3))
            row = 0

            for j, a in enumerate(alphas):
                print(' ')
                print('Alpha: ', a)

                for k, g in enumerate(gammas):
                    print('\n Gamma:', g)

                    for m, e in enumerate(epsilons):
                        fold=0
                        print('\n Epsilon:', e)

                        svr_model = SVR(kernel=kernel, C=a, gamma=g, epsilon=e)

                        for train_idx, test_idx in tscv.split(indpro_insample):
                            X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
                            y_train, y_test = y[train_idx], y[test_idx]

                            svr_model.fit(X_train, y_train) # fit ridge on training data
                            svr_pred = svr_model.predict(X_test) # predict using test data

                            print('Test RMSE for fold ', fold+1, ' is ', sqrt(mean_squared_error(y_test, svr_pred)))

                            svr_folds[row, 0] = a
                            svr_folds[row, 1] = g
                            svr_folds[row, 2] = e
                            svr_folds[row, fold+3] = sqrt(mean_squared_error(y_test, svr_pred))
                            fold += 1

                        row+= 1
                    
            svr_para_min_idx = np.mean(svr_folds[:,3:7], axis=1).argmin()
            print(' ')

            svr_lags[i, 0] = lag
            svr_lags[i, 1] = svr_folds[svr_para_min_idx,0]
            svr_lags[i, 2] = svr_folds[svr_para_min_idx,1]
            svr_lags[i, 3] = svr_folds[svr_para_min_idx,2]
            svr_lags[i, 4] = np.mean(svr_folds[:,3:7], axis=1)[svr_para_min_idx]


        ## Optimal SVR model
        svr_opt_lag = int(svr_lags[svr_lags[:,4].argmin(),0])
        svr_opt_alpha = svr_lags[svr_lags[:,4].argmin(),1]
        svr_opt_gamma = svr_lags[svr_lags[:,4].argmin(),2]
        svr_opt_epsilon = svr_lags[svr_lags[:,4].argmin(),3]
        print('Optimal lag for POOS CV is: ', svr_opt_lag)
        print('Optimal alpha for POOS CV is: ', svr_opt_alpha)
        print('Optimal gamma for POOS CV is: ', svr_opt_gamma)
        print('Optimal gamma for POOS CV is: ', svr_opt_epsilon)

        for l in range(svr_opt_lag):
            indpro_insample_df[f'lag_{l+1}'] = indpro_insample_df['INDPRO'].shift(l+1)
            indpro_oos_df[f'lag_{l+1}'] = indpro_oos_df['INDPRO'].shift(l+1)

        y_insample = indpro_insample_df['INDPRO']
        y_oos = indpro_oos_df['INDPRO']
        X_insample = indpro_insample_df.iloc[:,1:svr_opt_lag+1].fillna(0)
        X_oos = indpro_oos_df.iloc[:,1:svr_opt_lag+1].fillna(0)

        svr_opt = SVR(kernel=kernel, C=svr_opt_alpha, gamma=svr_opt_gamma, epsilon=svr_opt_epsilon)
        svr_opt.fit(X_insample, y_insample)
        svr_opt_pred = svr_opt.predict(X_oos)
        print('Out-of-sample RMSE for best SVR POOS model is ', sqrt(mean_squared_error(y_oos, svr_opt_pred)))
        
        return svr_opt_pred