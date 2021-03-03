def ar_model_select(data_insample, data_oos, maxlags, ic, ss, tscv):
    
    """
    This function chooses the optimal lag lenth criteria for the AR model given the user's choice for hyperparameter selection.
    
    Inputs:
    indpro_insample: insample data for estimation.
    indpro_oos: out-of-sample data for final evaluation.
    maxlags: lag length for AR models.
    ic: method to use for optimal lag length.
    ss: indices for K-fold CV.
    tscv: indices for pseudo out-of-sample CV.
    
    Output:
    ar_'ic'_pred: predictions for optimal AR model, depending on method chosen by user.
    """
    
    ### Selecting hyperparameters: Lag length
    
    ## Insample criteria: BIC and AIC
    results = np.zeros((len(maxlags), 3)) # matrix to store results, first column is lags, second column is BIC and 
                                          # third column is AIC
    
    for i, lag in enumerate(maxlags):
        # train AR model
        ar_model = AutoReg(data_insample, lags=lag)
        ar_model_fit = ar_model.fit()
        print('Lags: %s' % ar_model_fit.ar_lags[-1], 'BIC: %s' % ar_model_fit.bic, 'AIC: %s' % ar_model_fit.aic)
        results[i,0] =  ar_model_fit.ar_lags[-1]
        results[i,1] = ar_model_fit.bic
        results[i,2] = ar_model_fit.aic
        
        
    ## K-fold CV
    results_kfold = np.zeros((len(maxlags), ss.n_splits))
    fold = 0

    for i, lag in enumerate(maxlags):
        print('\n K-fold CV')
        print('\n Lag:', lag)
        fold = 0
        for train_idx, test_idx in ss.split(data_insample):
            train, test = data_insample[train_idx], data_insample[test_idx]

            model = AutoReg(train, lag)
            model_fit = model.fit()
            model_pred = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
            print('Test RMSE for fold ', fold+1, ' is: ', sqrt(mean_squared_error(test, model_pred)))
            results_kfold[i, fold] = sqrt(mean_squared_error(test, model_pred))
            fold +=1
            
            
    ## POOS CV
    maxlags = [1, 3, 9, 12]
    results_poos = np.zeros((len(maxlags), tscv.n_splits))
    fold = 0
    
    for i, lag in enumerate(maxlags):
        print('\n POOS CV')
        print('\n Lag:', lag)
        fold = 0
        for train_idx, test_idx in tscv.split(data_insample):
            train, test = data_insample[train_idx], data_insample[test_idx]

            model = AutoReg(train, lag)
            model_fit = model.fit()
            model_pred = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
            print('Test RMSE for fold ', fold+1, ' is: ', sqrt(mean_squared_error(test, model_pred)))
            results_poos[i, fold] = sqrt(mean_squared_error(test, model_pred))
            fold +=1
    
        
    #------------------------------------------------------------------------------------------------------#
    ### Prediction
    
    ## Optimal BIC Model
    if ic == 'bic':
        bic_min = np.amin(results[:,1]) # minimum BIC ignoring negative signs
        bic_min_idx = np.where(results[:,1] == np.amin(results[:,1])) # array index for min BIC
        bic_min_lags = int(results[bic_min_idx,0]) # optimal lag length, where BIC is minimum   
        print(' ')
        print('Optimal lag length for BIC: %s' % bic_min_lags)
        
        # train model
        ar_bic = AutoReg(data_insample, bic_min_lags)
        ar_bic_fit = ar_bic.fit()
        # predictions 
        ar_bic_pred = ar_bic_fit.predict(start=len(data_insample), end=len(data_insample)+len(data_oos)-1, dynamic=False)
        ar_bic_rmse = sqrt(mean_squared_error(data_oos, ar_bic_pred))
        print('Out-of-sample RMSE for best BIC model: %.3f' % ar_bic_rmse)
        
        return ar_bic_pred
    
    
    ## Optimal AIC Model
    elif ic == 'aic':
        aic_min = np.amin(results[:,2]) # minimum AIC ignoring negative signs
        aic_min_idx = np.where(results[:,2] == np.amin(results[:,2])) # array index for min AIC
        aic_min_lags = int(results[aic_min_idx,0]) # optimal lag length, where BIC is minimum    
        print(' ')
        print('Optimal lag length for AIC: %s' % aic_min_lags) 
        
        # train model
        ar_aic = AutoReg(data_insample, aic_min_lags)
        ar_aic_fit = ar_aic.fit()
        # predictions 
        ar_aic_pred = ar_aic_fit.predict(start=len(data_insample), end=len(data_insample)+len(data_oos)-1, dynamic=False)
        ar_aic_rmse = sqrt(mean_squared_error(data_oos, ar_aic_pred))
        print('Out-of-sample RMSE for best AIC model: %.3f' % ar_aic_rmse)
        
        return ar_aic_pred
    
    
    ## Optimal K-fold CV Model
    elif ic == 'kfold':
#         kfold_min = np.amin(np.mean(results_kfold, axis=1)) # min RMSE 
        kfold_min_idx = np.mean(results_kfold, axis=1).argmin() # array index for min RMSE
        kfold_min_lags = maxlags[kfold_min_idx] # optimal lag length, where RMSE in min
        print(' ')
        print('Optimal lag length for K-fold CV:', kfold_min_lags)
        
        # train model
        ar_kfold = AutoReg(data_insample, kfold_min_lags)
        ar_kfold_fit = ar_kfold.fit()
        # predictions
        ar_kfold_pred = ar_kfold_fit.predict(start=len(data_insample), end=len(data_insample)+len(data_oos)-1, dynamic=False)
        print('Out-of-sample RMSE for best K-fold CV model:', sqrt(mean_squared_error(data_oos, ar_kfold_pred)))
        
        return ar_kfold_pred
    
    
    ## Optimal POOS CV Model
    elif ic == 'poos':
        poos_min_idx = np.mean(results_poos, axis=1).argmin() # array index for min RMSE
        poos_min_lags = maxlags[poos_min_idx] # optimal lag length, where RMSE in min
        print(' ')
        print('Optimal lag length for POOS CV:', poos_min_lags)
        
        # train model
        ar_poos = AutoReg(data_insample, poos_min_lags)
        ar_poos_fit = ar_poos.fit()
        # predictions
        ar_poos_pred = ar_poos_fit.predict(start=len(data_insample), end=len(data_insample)+len(data_oos)-1, dynamic=False)
        print('Out-of-sample RMSE for best K-fold CV model:', sqrt(mean_squared_error(data_oos, ar_poos_pred)))
        
        return ar_poos_pred
        