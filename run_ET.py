import Nick as n
import main as m

print "Importing Data..."
X, y = n.import_file('train.csv')
cvx, cvy = n.import_file('cv.csv')
testx, testy = n.import_file('test.csv')

for n_estimators in (10, 20, 30, 40, 50):

	ET = n.train_ET(X, y, n_estimators=n_estimators)

	preds_cv, _ = n.predict_model(ET, cvx, cvy)
	preds_test, _ = n.predict_model(ET, testx, testy)

	m.score_preds(preds_cv, cvy)

	m.save_preds(preds_cv, 'ET_%d_year_cv_preds_full.csv' % n_estimators)
	m.save_preds(preds_test, 'ET_%d_year_test_preds_full.csv' % n_estimators)
