import sklearn.linear_model as linear_model    

def linearRegressor(pred, gt, test, n_kpts=10, reg_pts=5):
        
    pred = pred.reshape(-1, n_kpts*2)
    gt = gt.reshape(-1, reg_pts*2)
    test = test.reshape(-1, n_kpts*2)
    # shape: (N, C)
    
    regr = linear_model.Ridge(alpha=0.0, fit_intercept=False)
    _ = regr.fit(pred, gt)
    y_predict = regr.predict(test)
    y_predict = y_predict.reshape(-1, reg_pts, 2)
    
    return y_predict